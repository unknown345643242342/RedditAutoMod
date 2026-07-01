"""
duplicate_service.py
=====================
Heavy-ML duplicate-image detection microservice for the PokeLeaks Devvit mod tool.

This is the ONLY piece that stays in Python. It does the work Devvit can't:
perceptual hashing (pHash), ORB feature matching, and EfficientNet-B0 embedding
similarity. Everything Reddit-facing (triggers, mod actions, settings, timers,
spoiler/report logic) now lives in the Devvit app and talks to this service over
a small HTTP API.

Key differences from the old monolithic bot:
  * No asyncpraw. No Reddit credentials. This service never touches Reddit.
    (That kills the plaintext-credentials problem for good.)
  * No polling loops, no modqueue/stream/inbox workers, no mod-log scanning.
    Devvit drives everything via events and calls the endpoints below.
  * The index now stores the original post's METADATA (author/title/permalink/
    created_utc) because this service can no longer look it up on Reddit. Devvit
    passes that metadata in at index time.

Endpoints
  GET  /health                      -> {"status": "ok"}
  POST /check                       -> run duplicate detection, index if new
  POST /index                       -> store a post as an original (backfill, no dup check)
  POST /mod-removed                 -> flag a hash as moderator-removed (repost guard)
  POST /delete                      -> purge a submission from the index (user delete)

All POST bodies are JSON. If AUTH_TOKEN is set in the environment, every request
must carry a matching  X-Auth-Token  header (Devvit sends this from a secret).

Run:  pip install aiohttp numpy pillow imagehash opencv-python-headless torch torchvision
      python duplicate_service.py            # serves on 0.0.0.0:8080

Put this behind HTTPS (a reverse proxy such as Caddy/nginx, or a tunnel) and add
the resulting hostname to the Devvit app's http.domains allow-list.
"""

import os
import asyncio
from datetime import datetime

import aiohttp
from aiohttp import web
import numpy as np
from PIL import Image
import imagehash
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T

# =========================
# Config
# =========================
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")  # optional shared secret

# Detection thresholds (unchanged from the original bot)
HASH_DISTANCE = 3        # max pHash Hamming distance to treat as a hash match
HASH_AI_MIN = 0.50       # AI similarity required to confirm a hash match
ORB_SIM_MIN = 0.50       # ORB descriptor match ratio to consider a candidate
ORB_AI_MIN = 0.75        # AI similarity required to confirm an ORB match

# =========================
# Model / matcher globals (CPU, load once)
# =========================
device = "cpu"
efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet_model.eval()
efficientnet_model.to(device)
efficientnet_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# cv2.ORB is not thread-safe -> a fresh instance is created per call inside the
# worker thread. The brute-force matcher is stateless and safe to share.
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# =========================
# Per-subreddit in-memory index
# =========================
# subreddit -> {
#   'image_hashes':            { hash: (submission_id, created_utc) },
#   'orb_descriptors':         { submission_id: np.ndarray|None },
#   'ai_features':             { submission_id: torch.Tensor|None },
#   'moderator_removed_hashes': set(hash),
#   'metadata':                { submission_id: {author,title,permalink,created_utc} },
# }
subreddit_data = {}


def _store(subreddit_name):
    data = subreddit_data.get(subreddit_name)
    if data is None:
        data = {
            "image_hashes": {},
            "orb_descriptors": {},
            "ai_features": {},
            "moderator_removed_hashes": set(),
            "metadata": {},
        }
        subreddit_data[subreddit_name] = data
    return data


# =========================
# Heavy CPU math (runs in a worker thread)
# =========================
def _process_image_cpu(image_data):
    """Decode an image and compute (image, pHash, ORB descriptors, AI features).

    Pure CPU work; call via asyncio.to_thread() so it never blocks the loop.
    """
    local_orb = cv2.ORB_create()

    img = np.asarray(bytearray(image_data), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2 could not decode image bytes")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_value = str(imagehash.phash(Image.fromarray(gray)))

    def is_problematic_image(img_chk, white_threshold=0.7, text_threshold=0.05):
        gray_chk = cv2.cvtColor(img_chk, cv2.COLOR_BGR2GRAY) if len(img_chk.shape) == 3 else img_chk
        white_ratio = np.mean(gray_chk > 240)
        if white_ratio > white_threshold:
            return True
        edges = cv2.Canny(gray_chk, 100, 200)
        return np.mean(edges > 0) > text_threshold

    def get_orb_descriptors_conditional(img_chk):
        if is_problematic_image(img_chk):
            _, processed_img = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
            processed_img = cv2.Canny(processed_img, 100, 200)
        else:
            processed_img = gray
        _, des = local_orb.detectAndCompute(processed_img, None)
        return des

    def get_ai_features(img_chk):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img_chk, cv2.COLOR_BGR2RGB))
            img_tensor = efficientnet_transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = efficientnet_model(img_tensor)
                feat = feat / feat.norm(dim=1, keepdim=True)
            return feat
        except Exception as e:
            print(f"[AI] feature extraction error: {e}")
            return None

    descriptors = get_orb_descriptors_conditional(img)
    features = get_ai_features(img)
    return img, hash_value, descriptors, features


def _orb_similarity(desc1, desc2):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return 0.0
    matches = bf_matcher.match(desc1, desc2)
    return len(matches) / min(len(desc1), len(desc2))


def _ai_similarity(f1, f2):
    if f1 is not None and f2 is not None:
        return (f1 @ f2.T).item()
    return 0.0


def _fmt_date(utc_timestamp):
    return datetime.utcfromtimestamp(utc_timestamp).strftime("%Y-%m-%d %H:%M:%S")


# =========================
# Download + process
# =========================
async def _download_and_process(session, url):
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        resp.raise_for_status()
        image_data = await resp.read()
    return await asyncio.to_thread(_process_image_cpu, image_data)


def _original_payload(data, submission_id, status):
    meta = data["metadata"].get(submission_id, {})
    return {
        "submission_id": submission_id,
        "author": meta.get("author") or "[Deleted]",
        "title": meta.get("title") or "",
        "permalink": meta.get("permalink") or "",
        "created_utc": meta.get("created_utc"),
        "date": _fmt_date(meta["created_utc"]) if meta.get("created_utc") else "",
        "status": status,
    }


# =========================
# Core detection
# =========================
def _check_against_index(data, submission_id, created_utc, hash_value, descriptors, features):
    """Return (is_duplicate, method, original_payload) or (False, None, None)."""

    # 1) Repost of a moderator-removed image (matched by pHash within distance).
    for stored_hash, (orig_id, _orig_time) in list(data["image_hashes"].items()):
        if stored_hash not in data["moderator_removed_hashes"]:
            continue
        if hash_value == stored_hash or \
           (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= HASH_DISTANCE:
            return True, "mod-removed repost", _original_payload(data, orig_id, "Removed by Moderator")

    # 2) pHash duplicate (+ AI confirmation).
    matched_hash = None
    for stored_hash in data["image_hashes"].keys():
        if hash_value == stored_hash or \
           (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= HASH_DISTANCE:
            matched_hash = stored_hash
            break
    if matched_hash is not None:
        orig_id, orig_time = data["image_hashes"][matched_hash]
        # Only the newer post is a repost of the older original.
        if orig_id != submission_id and created_utc > orig_time:
            ai = _ai_similarity(features, data["ai_features"].get(orig_id))
            print(f"[MATCH] hash hit vs {orig_id} | AI={ai:.2f}")
            if ai > HASH_AI_MIN:
                status = "Removed by Moderator" if matched_hash in data["moderator_removed_hashes"] else "Active"
                return True, "hash + AI", _original_payload(data, orig_id, status)

    # 3) ORB duplicate (+ stricter AI confirmation).
    for orig_id, old_desc in data["orb_descriptors"].items():
        if orig_id == submission_id:
            continue
        if _orb_similarity(descriptors, old_desc) > ORB_SIM_MIN:
            ai = _ai_similarity(features, data["ai_features"].get(orig_id))
            if ai > ORB_AI_MIN:
                print(f"[MATCH] ORB hit vs {orig_id} | AI={ai:.2f}")
                old_hash = next((h for h, v in data["image_hashes"].items() if v[0] == orig_id), None)
                status = "Removed by Moderator" if old_hash and old_hash in data["moderator_removed_hashes"] else "Active"
                return True, "ORB + AI", _original_payload(data, orig_id, status)

    return False, None, None


def _store_original(data, submission_id, created_utc, hash_value, descriptors, features, meta):
    if hash_value not in data["image_hashes"]:
        data["image_hashes"][hash_value] = (submission_id, created_utc)
    data["orb_descriptors"][submission_id] = descriptors
    data["ai_features"][submission_id] = features
    data["metadata"][submission_id] = {
        "author": meta.get("author"),
        "title": meta.get("title"),
        "permalink": meta.get("permalink"),
        "created_utc": created_utc,
    }


# =========================
# HTTP handlers
# =========================
@web.middleware
async def auth_middleware(request, handler):
    if AUTH_TOKEN and request.path != "/health":
        if request.headers.get("X-Auth-Token", "") != AUTH_TOKEN:
            return web.json_response({"error": "unauthorized"}, status=401)
    return await handler(request)


async def handle_health(request):
    return web.json_response({"status": "ok"})


async def handle_check(request):
    body = await request.json()
    subreddit = body["subreddit"]
    submission_id = body["submission_id"]
    url = body["url"]
    created_utc = float(body["created_utc"])
    data = _store(subreddit)

    try:
        _, hash_value, descriptors, features = await _download_and_process(request.app["session"], url)
    except Exception as e:
        print(f"[r/{subreddit}] [CHECK] processing failed for {submission_id}: {e}")
        return web.json_response({"duplicate": False, "error": str(e)}, status=200)

    print(f"[r/{subreddit}] [CHECK] {submission_id} hash={hash_value}")
    is_dup, method, original = _check_against_index(
        data, submission_id, created_utc, hash_value, descriptors, features
    )
    if is_dup:
        return web.json_response({"duplicate": True, "method": method, "original": original})

    _store_original(data, submission_id, created_utc, hash_value, descriptors, features, body)
    print(f"[r/{subreddit}] [CHECK] stored new original {submission_id}")
    return web.json_response({"duplicate": False})


async def handle_index(request):
    """Backfill: store a post as an original without running duplicate detection."""
    body = await request.json()
    subreddit = body["subreddit"]
    submission_id = body["submission_id"]
    url = body["url"]
    created_utc = float(body["created_utc"])
    data = _store(subreddit)

    if submission_id in data["metadata"]:
        return web.json_response({"indexed": True, "skipped": "already indexed"})

    try:
        _, hash_value, descriptors, features = await _download_and_process(request.app["session"], url)
    except Exception as e:
        print(f"[r/{subreddit}] [INDEX] processing failed for {submission_id}: {e}")
        return web.json_response({"indexed": False, "error": str(e)}, status=200)

    _store_original(data, submission_id, created_utc, hash_value, descriptors, features, body)
    print(f"[r/{subreddit}] [INDEX] indexed {submission_id} hash={hash_value}")
    return web.json_response({"indexed": True})


async def handle_mod_removed(request):
    body = await request.json()
    subreddit = body["subreddit"]
    submission_id = body["submission_id"]
    data = _store(subreddit)

    flagged = None
    for h, (sid, _t) in data["image_hashes"].items():
        if sid == submission_id:
            flagged = h
            break
    if flagged and flagged not in data["moderator_removed_hashes"]:
        data["moderator_removed_hashes"].add(flagged)
        print(f"[r/{subreddit}] [MOD REMOVE] {submission_id} hash flagged for repost detection")
        return web.json_response({"flagged": True})
    return web.json_response({"flagged": False, "reason": "hash not indexed"})


async def handle_delete(request):
    body = await request.json()
    subreddit = body["subreddit"]
    submission_id = body["submission_id"]
    data = _store(subreddit)

    removed_hash = None
    for h, (sid, _t) in list(data["image_hashes"].items()):
        if sid == submission_id:
            removed_hash = h
            del data["image_hashes"][h]
            break
    data["orb_descriptors"].pop(submission_id, None)
    data["ai_features"].pop(submission_id, None)
    data["metadata"].pop(submission_id, None)
    if removed_hash:
        data["moderator_removed_hashes"].discard(removed_hash)
    print(f"[r/{subreddit}] [DELETE] purged {submission_id} from index")
    return web.json_response({"deleted": True})


# =========================
# App bootstrap
# =========================
async def _on_startup(app):
    app["session"] = aiohttp.ClientSession()
    print(f"[SYSTEM] duplicate_service started on {HOST}:{PORT} "
          f"(auth {'ON' if AUTH_TOKEN else 'OFF'})")


async def _on_cleanup(app):
    await app["session"].close()
    print("[SYSTEM] duplicate_service shut down")


def build_app():
    app = web.Application(middlewares=[auth_middleware], client_max_size=25 * 1024 * 1024)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/check", handle_check)
    app.router.add_post("/index", handle_index)
    app.router.add_post("/mod-removed", handle_mod_removed)
    app.router.add_post("/delete", handle_delete)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


if __name__ == "__main__":
    web.run_app(build_app(), host=HOST, port=PORT)
