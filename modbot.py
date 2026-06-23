# =============================================================================
# Standard library
# =============================================================================
import threading
import time
import traceback
from datetime import datetime, timezone

# =============================================================================
# Third-party
# =============================================================================
import cv2
import imagehash
import numpy as np
import praw
import prawcore.exceptions
import requests
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================
SUBREDDIT_NAME = 'PokeLeaks'

IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png', 'gif')

# Thresholds: minimum report count required to trigger remove or approve.
APPROVE_THRESHOLDS = {
    'This is misinformation': 1,
    'This is spam':           1,
}
COMMENT_REMOVE_THRESHOLDS = {
    'No Linking to Downloadable Content in Posts or Comments':          1,
    'No ROMs, ISOs, or Game Files Sharing or Requests':                 1,
    'No insults or harassment of other subreddit members in the comments': 1,
}
SUBMISSION_REMOVE_THRESHOLDS = {
    'Users Are Responsible for the Content They Post':                                          2,
    'Discussion-Only for Leaks, Not Distribution':                                             2,
    'No Linking to Downloadable Content in Posts or Comments':                                 1,
    'No ROMs, ISOs, or Game Files Sharing or Requests':                                        2,
    'Theories, Questions, Speculations must be commented in the Theory/Speculation/Question Megathread': 2,
    'Content Must Relate to Pokémon Leaks or News':                                            2,
    'Content must not contain any profanities, vulgarity, sexual content, slurs, be appropriate in nature': 2,
    'Post title should include sourcing and must be transparent':                               2,
    'Posts with spoilers must have required spoiler flair, indicate spoiler alert in title, and be vague': 3,
    'No reposting of posts already up on the subreddit':                                       2,
    'No Self Advertisements or Promotion':                                                     2,
    'No Memes, Fan Art, or Joke Posts':                                                        2,
}

# Similarity thresholds for duplicate detection.
HASH_AI_SCORE_THRESHOLD = 0.50   # minimum AI cosine score after a pHash match
ORB_MATCH_THRESHOLD     = 0.50   # minimum ORB descriptor match ratio
ORB_AI_SCORE_THRESHOLD  = 0.75   # minimum AI cosine score after an ORB match

# Moderation timing constants.
ONE_REPORT_GRACE_PERIOD = 3600   # seconds before a 1-report post is auto-approved

# Removal-check polling intervals by post age (seconds).
RECENT_AGE_LIMIT    =  3_600    # < 1 hour  → check every 30 s
MEDIUM_AGE_LIMIT    = 86_400    # < 1 day   → check every 5 min
RECENT_INTERVAL     =     30
MEDIUM_INTERVAL     =    300
OLD_INTERVAL        =  1_800

# Cache size caps.
MAX_PROCESSED_MODQUEUE = 5_000
MAX_PROCESSED_LOG      = 1_000

# Initial scan limit (Reddit caps at ~1000 in practice; 20000 is aspirational).
INITIAL_SCAN_LIMIT = 20_000

# Removal-checker batch size before a courtesy sleep.
REMOVAL_BATCH_SIZE  = 10
REMOVAL_BATCH_SLEEP = 60        # seconds

# =============================================================================
# Shared Reddit instance
# =============================================================================
_reddit      = None
_reddit_lock = threading.Lock()

def get_reddit():
    """Return one shared PRAW instance; lazily created, thread-safe."""
    global _reddit
    with _reddit_lock:
        if _reddit is None:
            _reddit = praw.Reddit(
                client_id     = 'jl-I3OHYH2_VZMC1feoJMQ',
                client_secret = 'TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
                username      = 'PokeLeakBot3',
                password      = 'testbot1',
                user_agent    = 'testbot',
            )
    return _reddit

# =============================================================================
# Logging / exception handling
# =============================================================================
def handle_exception(e, context=""):
    """Log all exceptions; silently drop 429 rate-limits."""
    if isinstance(e, prawcore.exceptions.ResponseException):
        if getattr(e, "response", None) and e.response.status_code == 429:
            suffix = f" in {context}" if context else ""
            print(f"[RATE LIMIT] 429 suppressed{suffix}.")
            return
    prefix = f"[ERROR {context}]" if context else "[ERROR]"
    print(f"{prefix} {type(e).__name__}: {e}")
    traceback.print_exc()

# =============================================================================
# Loop runner — the sole crash-recovery mechanism.
# Worker functions provide only their body logic; they must not loop themselves.
# =============================================================================
def loop_forever(fn, interval, context=""):
    """Call fn() repeatedly, sleeping interval seconds between calls.
    Exceptions are logged and execution continues."""
    while True:
        try:
            fn()
        except Exception as e:
            handle_exception(e, context=context)
        time.sleep(interval)

# =============================================================================
# Shared helpers
# =============================================================================
def is_image_url(url):
    """Return True when url points at a supported image format."""
    return url.endswith(IMAGE_EXTENSIONS)

def _apply_threshold_action(item, thresholds, action):
    """Apply remove or approve to item if its top report reason meets its threshold.
    Returns True when the action is taken."""
    reports = getattr(item, "user_reports", None)
    if not reports:
        return False
    reason, count = reports[0][0], reports[0][1]
    if reason not in thresholds or count < thresholds[reason]:
        return False

    label = (getattr(item, 'body', None) or getattr(item, 'title', ''))[:80]
    try:
        if action == 'remove':
            item.mod.remove()
            print(f'[{SUBREDDIT_NAME}] Removed  ({reason}, ×{count}): "{label}"')
        elif action == 'approve':
            item.mod.approve()
            print(f'[{SUBREDDIT_NAME}] Approved ({reason}, ×{count}): "{label}"')
        return True
    except prawcore.exceptions.ServerError as e:
        handle_exception(e, context=f"threshold {action}")
    return False

# =============================================================================
# Worker: modqueue report dispatcher
# =============================================================================
def handle_modqueue_reports():
    """Dispatch every modqueue item to remove/approve based on report reason.
    Also handles the 1-report / 1-hour grace-period timer for borderline posts."""
    reddit = get_reddit()
    timers = {}

    def body():
        now = time.time()
        for item in reddit.subreddit(SUBREDDIT_NAME).mod.modqueue(limit=100):
            if isinstance(item, praw.models.Comment):
                (_apply_threshold_action(item, COMMENT_REMOVE_THRESHOLDS, 'remove') or
                 _apply_threshold_action(item, APPROVE_THRESHOLDS,         'approve'))

            elif isinstance(item, praw.models.Submission):
                if _apply_threshold_action(item, APPROVE_THRESHOLDS, 'approve'):
                    timers.pop(item.id, None)
                    continue
                if _apply_threshold_action(item, SUBMISSION_REMOVE_THRESHOLDS, 'remove'):
                    timers.pop(item.id, None)
                    continue

                if getattr(item, "num_reports", 0) == 1:
                    if item.id not in timers:
                        timers[item.id] = now
                        print(f"[{SUBREDDIT_NAME}] Timer started for {item.id}")
                    elif now - timers[item.id] >= ONE_REPORT_GRACE_PERIOD:
                        try:
                            item.mod.approve()
                            print(f"[{SUBREDDIT_NAME}] Approved {item.id} "
                                  f"(1 report, grace period elapsed)")
                        except prawcore.exceptions.ServerError as e:
                            handle_exception(e, context="1-report approval")
                        timers.pop(item.id, None)
                    else:
                        remaining = int(timers[item.id] + ONE_REPORT_GRACE_PERIOD - now)
                        print(f"[{SUBREDDIT_NAME}] {item.id} — {remaining}s until auto-approve")

    loop_forever(body, interval=60, context="handle_modqueue_reports")

# =============================================================================
# Worker: re-approve previously-approved reported posts
# =============================================================================
def monitor_reported_posts():
    """Re-approve any post that has been approved before but is re-reported."""
    reddit = get_reddit()

    def body():
        for post in reddit.subreddit(SUBREDDIT_NAME).mod.reports():
            if getattr(post, "approved", False):
                post.mod.approve()
                print(f"[{SUBREDDIT_NAME}] Re-approved post {post.id}")

    loop_forever(body, interval=60, context="monitor_reported_posts")

# =============================================================================
# Worker: spoiler-tag enforcement
# =============================================================================
def handle_spoiler_status():
    """Re-apply spoiler tags removed by non-moderators."""
    reddit    = get_reddit()
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    previous  = {}   # submission_id → last-seen spoiler bool

    def body():
        for submission in subreddit.new():
            if submission.id not in previous:
                previous[submission.id] = submission.spoiler
                continue
            if previous[submission.id] == submission.spoiler:
                continue

            try:
                is_mod = submission.author in subreddit.moderator()
            except Exception:
                is_mod = False

            if not submission.spoiler and not is_mod:
                try:
                    submission.mod.spoiler()
                    print(f"[{SUBREDDIT_NAME}] Re-marked {submission.id} as spoiler "
                          f"(non-mod removed it)")
                except prawcore.exceptions.ServerError as e:
                    handle_exception(e, context="spoiler re-mark")
            else:
                print(f"[{SUBREDDIT_NAME}] Spoiler change on {submission.id} "
                      f"by mod — leaving as-is")

            previous[submission.id] = submission.spoiler

    loop_forever(body, interval=30, context="handle_spoiler_status")

# =============================================================================
# Worker: duplicate image detection
# =============================================================================
def run_pokemon_duplicate_bot():
    """Detect and remove duplicate image posts across all moderated subreddits.
    Automatically accepts moderator invites and sets up new subreddits on the fly."""

    reddit              = get_reddit()
    subreddit_data      = {}
    subreddit_data_lock = threading.Lock()

    # -------------------------------------------------------------------------
    # AI / CV models — loaded once, shared across all subreddits
    # -------------------------------------------------------------------------
    device       = "cpu"
    efficientnet = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    ).eval().to(device)
    en_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    orb_detector = cv2.ORB_create()
    bf_matcher   = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # -------------------------------------------------------------------------
    # Image utilities
    # -------------------------------------------------------------------------
    def fetch_image_array(url):
        """Download url and return a BGR numpy array."""
        raw = requests.get(url, timeout=10).content
        return cv2.imdecode(np.asarray(bytearray(raw), dtype=np.uint8), cv2.IMREAD_COLOR)

    def get_ai_features(img):
        """Return a normalised EfficientNet feature vector for a BGR image, or None on error."""
        try:
            pil    = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            tensor = en_transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = efficientnet(tensor)
            return feat / feat.norm(dim=1, keepdim=True)
        except Exception as e:
            handle_exception(e, context="get_ai_features")
            return None

    def is_problematic_image(img, white_thr=0.7, edge_thr=0.05):
        """Return True for near-white or heavily-edged images that confuse plain ORB."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(gray > 240) > white_thr:
            return True
        return np.mean(cv2.Canny(gray, 100, 200) > 0) > edge_thr

    def get_orb_descriptors(img):
        """Return ORB descriptors, using edge-preprocessed input for problematic images."""
        if is_problematic_image(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, gray = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
            processed = cv2.Canny(gray, 100, 200)
        else:
            processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = orb_detector.detectAndCompute(processed, None)
        return descriptors

    def orb_similarity(d1, d2):
        """Return the ratio of ORB matches to the smaller descriptor set."""
        if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
            return 0
        return len(bf_matcher.match(d1, d2)) / min(len(d1), len(d2))

    def ai_similarity(f1, f2):
        """Return the cosine similarity between two EfficientNet feature vectors."""
        if f1 is not None and f2 is not None:
            return (f1 @ f2.T).item()
        return 0

    def load_and_process_image(url):
        """Fetch an image and return (phash_str, orb_descriptors, ai_features)."""
        img  = fetch_image_array(url)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h    = str(imagehash.phash(Image.fromarray(gray)))
        return h, get_orb_descriptors(img), get_ai_features(img)

    def format_age(utc_ts):
        """Return a human-readable age string for a UTC timestamp."""
        delta      = datetime.now(timezone.utc) - datetime.fromtimestamp(utc_ts, tz=timezone.utc)
        days, secs = delta.days, delta.seconds
        if days:
            return f"{days} day{'s' if days != 1 else ''} ago"
        if secs >= 3600:
            h = secs // 3600
            return f"{h} hour{'s' if h != 1 else ''} ago"
        if secs >= 60:
            m = secs // 60
            return f"{m} minute{'s' if m != 1 else ''} ago"
        return f"{secs} second{'s' if secs != 1 else ''} ago"

    # -------------------------------------------------------------------------
    # Cache helpers
    # -------------------------------------------------------------------------
    def get_cached_features(data, submission_id):
        """Return cached AI features for a submission, fetching them if absent."""
        if submission_id not in data['ai_features']:
            sub = reddit.submission(id=submission_id)
            data['ai_features'][submission_id] = get_ai_features(fetch_image_array(sub.url))
        return data['ai_features'][submission_id]

    def store_new_original(data, hash_value, submission, descriptors, features):
        """Insert a confirmed original into all three caches."""
        data['image_hashes'][hash_value]       = (submission.id, submission.created_utc)
        data['orb_descriptors'][submission.id] = descriptors
        data['ai_features'][submission.id]     = features

    def evict_caches(data, submission_id, hash_value=None):
        """Remove a submission from all caches (e.g. after user deletion)."""
        if hash_value and hash_value in data['image_hashes']:
            del data['image_hashes'][hash_value]
        data['orb_descriptors'].pop(submission_id, None)
        data['ai_features'].pop(submission_id, None)
        data['last_checked'].pop(submission_id, None)

    # -------------------------------------------------------------------------
    # Duplicate info builders
    # -------------------------------------------------------------------------
    def _original_submission_info(original_id, status):
        """Fetch the original PRAW submission and return the 6-tuple for comments."""
        orig = reddit.submission(id=original_id)
        date = datetime.fromtimestamp(
            orig.created_utc, tz=timezone.utc
        ).strftime('%Y-%m-%d %H:%M:%S')
        return orig.author.name, orig.title, date, orig.created_utc, status, orig.permalink

    def build_info_from_hash(data, original_id, matched_hash):
        """Build duplicate comment info when the match was via pHash."""
        status = ("Removed by Moderator"
                  if matched_hash in data['moderator_removed_hashes'] else "Active")
        return _original_submission_info(original_id, status)

    def build_info_from_id(data, original_id):
        """Build duplicate comment info when the match was via ORB (hash not in hand)."""
        old_hash = next(
            (h for h, v in data['image_hashes'].items() if v[0] == original_id), None
        )
        status = ("Removed by Moderator"
                  if old_hash and old_hash in data['moderator_removed_hashes'] else "Active")
        return _original_submission_info(original_id, status)

    # -------------------------------------------------------------------------
    # Duplicate removal
    # -------------------------------------------------------------------------
    def remove_and_comment(sub_name, submission, author, title, date, utc, status, permalink,
                           method):
        """Remove a duplicate submission and post a sticky comment identifying the original.
        Skips already-approved posts; retries the comment up to 3 times."""
        if submission.approved:
            return
        submission.mod.remove()
        body = (
            "> **Duplicate detected**\n\n"
            "| Original Author | Title | Date | Age | Status |\n"
            "|:---------------:|:-----:|:----:|:---:|:------:|\n"
            f"| {author} | [{title}]({permalink}) | {date} "
            f"| {format_age(utc)} | {status} |"
        )
        for attempt in range(3):
            try:
                c = submission.reply(body)
                c.mod.distinguish(sticky=True)
                print(f"[r/{sub_name}] Duplicate removed+commented ({method}): {submission.url}")
                return
            except Exception as e:
                handle_exception(e, context=f"remove_and_comment attempt {attempt + 1}")
                time.sleep(1)

    # -------------------------------------------------------------------------
    # Duplicate checkers
    # -------------------------------------------------------------------------
    def check_hash_duplicate(data, submission, hash_value, new_features):
        """Return (True, info_tuple) if a pHash+AI duplicate is found, else (False, None)."""
        for stored_hash in data['image_hashes']:
            if hash_value == stored_hash or \
               (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= 3:
                original_id, original_time = data['image_hashes'][stored_hash]
                if submission.id == original_id or submission.created_utc <= original_time:
                    break
                score = ai_similarity(new_features, get_cached_features(data, original_id))
                print(f"[dup] Hash match — AI similarity: {score:.2f}")
                if score > HASH_AI_SCORE_THRESHOLD:
                    return True, build_info_from_hash(data, original_id, stored_hash)
                break
        return False, None

    def check_orb_duplicate(data, submission, descriptors, new_features):
        """Return (True, info_tuple) if an ORB+AI duplicate is found, else (False, None)."""
        for old_id, old_desc in data['orb_descriptors'].items():
            if orb_similarity(descriptors, old_desc) > ORB_MATCH_THRESHOLD:
                score = ai_similarity(new_features, get_cached_features(data, old_id))
                if score > ORB_AI_SCORE_THRESHOLD:
                    print(f"[dup] ORB match — AI similarity: {score:.2f}")
                    return True, build_info_from_id(data, old_id)
        return False, None

    # -------------------------------------------------------------------------
    # Core pipeline
    # -------------------------------------------------------------------------
    def process_submission(sub_name, data, submission, context="stream"):
        """Run the full duplicate-detection pipeline for one submission.
        context is 'stream' for new posts or 'modqueue' for pending approvals."""
        try:
            hash_value, descriptors, new_features = load_and_process_image(submission.url)

            # Repost of a mod-removed image
            if hash_value in data['moderator_removed_hashes']:
                original_id = data['image_hashes'][hash_value][0]
                info = build_info_from_hash(data, original_id, hash_value)
                remove_and_comment(sub_name, submission, *info, method="mod-removed repost")
                return True

            # pHash duplicate check
            found, info = check_hash_duplicate(data, submission, hash_value, new_features)
            if found:
                remove_and_comment(sub_name, submission, *info, method="hash+AI")
                return True

            # ORB duplicate check
            found, info = check_orb_duplicate(data, submission, descriptors, new_features)
            if found:
                remove_and_comment(sub_name, submission, *info, method="ORB+AI")
                return True

            # Confirmed original — store it
            if hash_value not in data['image_hashes']:
                store_new_original(data, hash_value, submission, descriptors, new_features)
                print(f"[r/{sub_name}] Stored new original: {submission.url}")

            if context == "modqueue" and not submission.approved:
                submission.mod.approve()
                print(f"[r/{sub_name}] Original approved from modqueue: {submission.url}")

            return False

        except Exception as e:
            handle_exception(e, context=f"process_submission [{sub_name}]")
            return False

    # -------------------------------------------------------------------------
    # Subreddit setup
    # -------------------------------------------------------------------------
    def setup_subreddit(sub_name):
        """Initialise data structures for a new subreddit and index existing posts."""
        print(f"[r/{sub_name}] Setting up bot...")
        data = {
            'subreddit':                      reddit.subreddit(sub_name),
            'image_hashes':                   {},
            'orb_descriptors':                {},
            'moderator_removed_hashes':       set(),
            'processed_modqueue_submissions': set(),
            'ai_features':                    {},
            'current_time':                   int(time.time()),
            'processed_log_items':            set(),
            'last_checked':                   {},
        }
        with subreddit_data_lock:
            subreddit_data[sub_name] = data

        print(f"[r/{sub_name}] Starting initial scan (up to {INITIAL_SCAN_LIMIT} posts)...")
        try:
            for submission in data['subreddit'].new(limit=INITIAL_SCAN_LIMIT):
                if isinstance(submission, praw.models.Submission) and is_image_url(submission.url):
                    try:
                        h, desc, feat = load_and_process_image(submission.url)
                        if h not in data['image_hashes']:
                            store_new_original(data, h, submission, desc, feat)
                    except Exception as e:
                        handle_exception(e, context=f"initial scan [{sub_name}]")
        except Exception as e:
            handle_exception(e, context=f"initial scan outer [{sub_name}]")

        print(f"[r/{sub_name}] Ready — {len(data['image_hashes'])} images indexed.")

    # -------------------------------------------------------------------------
    # Subreddit iteration helpers
    # -------------------------------------------------------------------------
    def get_subreddit_names():
        """Return a snapshot of currently-known subreddit names (thread-safe)."""
        with subreddit_data_lock:
            return list(subreddit_data.keys())

    def for_each_subreddit(fn):
        """Call fn(name, data) for every known subreddit, logging per-subreddit errors."""
        for name in get_subreddit_names():
            try:
                fn(name, subreddit_data[name])
            except Exception as e:
                handle_exception(e, context=f"for_each_subreddit [{name}]")

    # -------------------------------------------------------------------------
    # Worker bodies
    # -------------------------------------------------------------------------
    def _body_check_for_invites():
        for message in reddit.inbox.unread(limit=None):
            if "invitation to moderate" not in message.subject.lower():
                continue
            sub_name = message.subreddit.display_name
            print(f"[invite] Received mod invite for r/{sub_name}")
            try:
                message.subreddit.mod.accept_invite()
                print(f"[invite] Accepted invite for r/{sub_name}")
                setup_subreddit(sub_name)
            except Exception as e:
                handle_exception(e, context=f"accept_invite [{sub_name}]")
            message.mark_read()

        for sub in reddit.user.moderator_subreddits(limit=None):
            name = sub.display_name
            if name not in subreddit_data:
                print(f"[invite] Already moderating r/{name} — setting up")
                setup_subreddit(name)

    def _body_mod_log_monitor():
        def handle(name, data):
            for entry in data['subreddit'].mod.log(action='removelink', limit=50):
                if entry.id in data['processed_log_items']:
                    continue
                data['processed_log_items'].add(entry.id)
                removed_id = entry.target_fullname.replace('t3_', '')
                for h, (sid, _) in list(data['image_hashes'].items()):
                    if sid == removed_id and h not in data['moderator_removed_hashes']:
                        data['moderator_removed_hashes'].add(h)
                        print(f"[r/{name}] Mod-removed {removed_id} — hash retained.")
                        break
                if len(data['processed_log_items']) > MAX_PROCESSED_LOG:
                    data['processed_log_items'].clear()
        for_each_subreddit(handle)

    def _body_removal_checker():
        def handle(name, data):
            now = time.time()
            recent, medium, old = [], [], []
            for h, (sid, created) in list(data['image_hashes'].items()):
                if h in data['moderator_removed_hashes']:
                    continue
                age  = now - created
                last = data['last_checked'].get(sid, 0)
                if   age <  RECENT_AGE_LIMIT and now - last >= RECENT_INTERVAL:
                    recent.append((h, sid))
                elif age <  MEDIUM_AGE_LIMIT and now - last >= MEDIUM_INTERVAL:
                    medium.append((h, sid))
                elif age >= MEDIUM_AGE_LIMIT and now - last >= OLD_INTERVAL:
                    old.append((h, sid))

            checked = 0
            for h, sid in recent + medium[:20] + old[:10]:
                try:
                    if reddit.submission(id=sid).author is None:
                        evict_caches(data, sid, hash_value=h)
                        print(f"[r/{name}] User-deleted {sid} — evicted from cache.")
                    else:
                        data['last_checked'][sid] = now
                    checked += 1
                    if checked >= REMOVAL_BATCH_SIZE:
                        time.sleep(REMOVAL_BATCH_SLEEP)
                        checked = 0
                except Exception as e:
                    handle_exception(e, context=f"removal_checker [{name}] {sid}")
                    data['last_checked'][sid] = now
        for_each_subreddit(handle)

    def _body_modqueue_worker():
        def handle(name, data):
            items = sorted(
                data['subreddit'].mod.modqueue(only='submission', limit=None),
                key=lambda x: x.created_utc,
            )
            for sub in items:
                if not isinstance(sub, praw.models.Submission):
                    continue
                if getattr(sub, 'num_reports', 0) > 0:
                    print(f"[r/{name}] Skipping reported submission: {sub.url}")
                    evict_caches(data, sub.id)
                    continue
                if is_image_url(sub.url):
                    process_submission(name, data, sub, context="modqueue")
                    data['processed_modqueue_submissions'].add(sub.id)
            if len(data['processed_modqueue_submissions']) > MAX_PROCESSED_MODQUEUE:
                data['processed_modqueue_submissions'].clear()
        for_each_subreddit(handle)

    def _body_stream_worker():
        def handle(name, data):
            cycle_start = int(time.time())
            for sub in data['subreddit'].new(limit=10):
                if not isinstance(sub, praw.models.Submission):
                    continue
                if sub.created_utc <= data['current_time']:
                    continue
                if sub.id in data['processed_modqueue_submissions']:
                    continue
                if is_image_url(sub.url):
                    process_submission(name, data, sub, context="stream")
            data['current_time'] = cycle_start
        for_each_subreddit(handle)

    # -------------------------------------------------------------------------
    # Launch worker threads
    # -------------------------------------------------------------------------
    workers = [
        (_body_check_for_invites, 60,  "check_for_invites"),
        (_body_mod_log_monitor,   30,  "mod_log_monitor"),
        (_body_removal_checker,   60,  "removal_checker"),
        (_body_modqueue_worker,   15,  "modqueue_worker"),
        (_body_stream_worker,     20,  "stream_worker"),
    ]
    for body, interval, ctx in workers:
        threading.Thread(
            target=loop_forever,
            args=(body, interval, ctx),
            daemon=True,
        ).start()
        print(f"[STARTED] {ctx}")

    print("[bot] Multi-subreddit duplicate bot running.")
    while True:
        time.sleep(20)

# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    top_level_workers = [
        handle_modqueue_reports,
        monitor_reported_posts,
        handle_spoiler_status,
        run_pokemon_duplicate_bot,
    ]
    for func in top_level_workers:
        threading.Thread(target=func, daemon=True, name=func.__name__).start()
        print(f"[STARTED] {func.__name__}")

    while True:
        time.sleep(30)
