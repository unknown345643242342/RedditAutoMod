import asyncio
import aiohttp
import asyncpraw
import asyncprawcore.exceptions
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import imagehash
import cv2
import traceback
import torch
import torchvision.models as models
import torchvision.transforms as T

# =========================
# Shared Async Globals & Configs
# =========================
# Shared clients initialized in main()
reddit = None
http_session = None

subreddit_data = {}

device = "cpu"
efficientnet_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet_model.eval()
efficientnet_model.to(device)
efficientnet_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
orb_detector = cv2.ORB_create()
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# =========================
# Crash-proof runner
# =========================
async def safe_run(target, *args, **kwargs):
    """
    Keeps an async target function running forever.
    """
    while True:
        try:
            await target(*args, **kwargs)
        except Exception as e:
            print(f"[FATAL] {target.__name__} crashed: {e}")
            traceback.print_exc()
            await asyncio.sleep(10)  # brief cooldown before retrying

def handle_exception(e):
    if isinstance(e, asyncprawcore.exceptions.ResponseException) and getattr(e, "response", None) and e.response.status == 429:
        print("Rate limited by Reddit API. Ignoring error.")

# =========================
# Accept Moderator Invites & Register Subreddits
# =========================
async def accept_invites_and_register():
    while True:
        try:
            # Accept any pending moderator invites
            async for message in reddit.inbox.unread(limit=None):
                if "invitation to moderate" in message.subject.lower():
                    if hasattr(message, 'subreddit') and message.subreddit:
                        try:
                            await message.subreddit.mod.accept_invite()
                            print(f"[REGISTRY] Accepted invite to r/{message.subreddit.display_name}")
                            await message.mark_read()
                        except Exception as e:
                            print(f"[REGISTRY] Failed to accept invite for r/{message.subreddit.display_name}: {e}")

            # Register any newly moderated subreddits with the duplicate bot
            async for sub in reddit.user.moderator_subreddits(limit=None):
                if sub.display_name not in subreddit_data:
                    print(f"[REGISTRY] Registering new subreddit: r/{sub.display_name}")
                    asyncio.create_task(setup_subreddit(sub.display_name))

        except Exception as e:
            print(f"[REGISTRY] Sync error: {e}")

        await asyncio.sleep(300)

# =========================
# Heavy CPU Math Isolation
# =========================
def _process_image_cpu(image_data):
    """Runs purely in a background thread to prevent blocking the async loop"""
    img = np.asarray(bytearray(image_data), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_value = str(imagehash.phash(Image.fromarray(gray)))

    def is_problematic_image(img_chk, white_threshold=0.7, text_threshold=0.05):
        white_ratio = np.mean(gray > 240)
        if white_ratio > white_threshold: return True
        edges = cv2.Canny(gray, 100, 200)
        return np.mean(edges > 0) > text_threshold

    def get_orb_descriptors_conditional(img_chk):
        if is_problematic_image(img_chk):
            _, processed_img = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
            processed_img = cv2.Canny(processed_img, 100, 200)
        else:
            processed_img = cv2.cvtColor(img_chk, cv2.COLOR_BGR2GRAY)
        kp, des = orb_detector.detectAndCompute(processed_img, None)
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
            print(f"AI feature extraction error: {e}")
            return None

    descriptors = get_orb_descriptors_conditional(img)
    features = get_ai_features(img)
    return img, hash_value, descriptors, features

# =========================
# Shared Duplicate Bot logic
# =========================
async def setup_subreddit(subreddit_name):
    print(f"\n=== Setting up bot for r/{subreddit_name} ===")

    subreddit = await reddit.subreddit(subreddit_name)

    data = {
        'subreddit': subreddit,
        'image_hashes': {},
        'orb_descriptors': {},
        'moderator_removed_hashes': set(),
        'processed_modqueue_submissions': set(),
        'approved_by_moderator': set(),
        'ai_features': {},
        'current_time': int(time.time()),
        'processed_log_items': set(),
        'last_checked': {}
    }

    subreddit_data[subreddit_name] = data

    def orb_similarity(desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0
        matches = bf_matcher.match(desc1, desc2)
        return len(matches) / min(len(desc1), len(desc2))

    def format_age(utc_timestamp):
        now = datetime.now(timezone.utc)
        created = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
        delta = now - created
        days = delta.days
        seconds = delta.seconds
        if days > 0: return f"{days} day{'s' if days != 1 else ''} ago"
        elif seconds >= 3600: return f"{seconds // 3600} hour{'s' if seconds // 3600 != 1 else ''} ago"
        elif seconds >= 60: return f"{seconds // 60} minute{'s' if seconds // 60 != 1 else ''} ago"
        else: return f"{seconds} second{'s' if seconds != 1 else ''} ago"

    async def post_comment(submission, original_post_author, original_post_title, original_post_date, original_post_utc, original_status, original_post_permalink):
        max_retries = 3
        retries = 0
        age_text = format_age(original_post_utc)
        while retries < max_retries:
            try:
                comment_text = (
                    "> **Duplicate detected**\n\n"
                    "| Original Author | Title | Date | Age | Status |\n"
                    "|:---------------:|:-----:|:----:|:---:|:------:|\n"
                    f"| {original_post_author} | [{original_post_title}]({original_post_permalink}) | {original_post_date} | {age_text} | {original_status} |"
                )
                comment = await submission.reply(comment_text)
                await comment.mod.distinguish(sticky=True)
                print(f"[r/{subreddit_name}] Duplicate removed and comment posted: ", submission.url)
                return True
            except Exception as e:
                handle_exception(e)
                retries += 1
                await asyncio.sleep(1)
        return False

    async def load_and_process_image(url):
        async with http_session.get(url, timeout=10) as response:
            image_data = await response.read()

        img, hash_value, descriptors, features = await asyncio.to_thread(_process_image_cpu, image_data)
        print(f"[r/{subreddit_name}] Generated hash: {hash_value}")
        return img, hash_value, descriptors, features

    async def get_cached_ai_features(submission_id):
        if submission_id in data['ai_features']:
            return data['ai_features'][submission_id]

        old_submission = await reddit.submission(id=submission_id)
        await old_submission.load()
        async with http_session.get(old_submission.url, timeout=10) as response:
            old_image_data = await response.read()

        _, _, _, old_features = await asyncio.to_thread(_process_image_cpu, old_image_data)
        data['ai_features'][submission_id] = old_features
        return old_features

    def calculate_ai_similarity(features1, features2):
        if features1 is not None and features2 is not None:
            return (features1 @ features2.T).item()
        return 0

    async def check_hash_duplicate(submission, hash_value, new_features):
        matched_hash = None
        for stored_hash in data['image_hashes'].keys():
            if hash_value == stored_hash or (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= 3:
                matched_hash = stored_hash
                break

        if matched_hash is None:
            return False, None, None, None, None, None, None

        original_id, original_time = data['image_hashes'][matched_hash]
        if submission.id == original_id or submission.created_utc <= original_time:
            return False, None, None, None, None, None, None

        original_submission = await reddit.submission(id=original_id)
        await original_submission.load()
        original_features = await get_cached_ai_features(original_submission.id)

        ai_score = calculate_ai_similarity(new_features, original_features)
        print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f}")

        if ai_score > 0.50:
            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            original_status = "Removed by Moderator" if matched_hash in data['moderator_removed_hashes'] else "Active"
            author_name = original_submission.author.name if original_submission.author else "[Deleted]"
            return True, author_name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink

        return False, None, None, None, None, None, None

    async def check_orb_duplicate(submission, descriptors, new_features):
        for old_id, old_desc in data['orb_descriptors'].items():
            sim = orb_similarity(descriptors, old_desc)
            if sim > 0.50:
                old_features = await get_cached_ai_features(old_id)
                ai_score = calculate_ai_similarity(new_features, old_features)

                if ai_score > 0.75:
                    print(f"[r/{subreddit_name}] ORB duplicate found! AI similarity: {ai_score:.2f}")
                    original_submission = await reddit.submission(id=old_id)
                    await original_submission.load()
                    original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    old_hash = next((h for h, v in data['image_hashes'].items() if v[0] == old_id), None)
                    original_status = "Removed by Moderator" if old_hash and old_hash in data['moderator_removed_hashes'] else "Active"
                    author_name = original_submission.author.name if original_submission.author else "[Deleted]"

                    return True, author_name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        return False, None, None, None, None, None, None

    async def handle_duplicate(submission, detection_method, author, title, date, utc, status, permalink):
        if not submission.approved:
            await submission.mod.remove()
            await post_comment(submission, author, title, date, utc, status, permalink)
            print(f"[r/{subreddit_name}] Duplicate removed by {detection_method}: {submission.url}")
        return True

    async def handle_moderator_removed_repost(submission, hash_value):
        if hash_value in data['moderator_removed_hashes'] and not submission.approved:
            await submission.mod.remove()
            original_submission = await reddit.submission(id=data['image_hashes'][hash_value][0])
            await original_submission.load()
            author_name = original_submission.author.name if original_submission.author else "[Deleted]"
            await post_comment(
                submission,
                author_name,
                original_submission.title,
                datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                original_submission.created_utc,
                "Removed by Moderator",
                original_submission.permalink
            )
            print(f"[r/{subreddit_name}] Repost of a moderator-removed image removed: ", submission.url)
            return True
        return False

    async def process_submission_for_duplicates(submission, context="stream"):
        try:
            _, hash_value, descriptors, new_features = await load_and_process_image(submission.url)
            data['ai_features'][submission.id] = new_features

            if await handle_moderator_removed_repost(submission, hash_value):
                return True

            is_duplicate, author, title, date, utc, status, permalink = await check_hash_duplicate(
                submission, hash_value, new_features
            )
            if is_duplicate:
                return await handle_duplicate(submission, "hash + AI", author, title, date, utc, status, permalink)

            is_duplicate, author, title, date, utc, status, permalink = await check_orb_duplicate(
                submission, descriptors, new_features
            )
            if is_duplicate:
                return await handle_duplicate(submission, "ORB + AI", author, title, date, utc, status, permalink)

            if hash_value not in data['image_hashes']:
                data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                data['orb_descriptors'][submission.id] = descriptors
                print(f"[r/{subreddit_name}] Stored new original: {submission.url}")

            if context == "modqueue" and not submission.approved:
                await submission.mod.approve()
                print(f"[r/{subreddit_name}] Original submission approved: ", submission.url)

            return False

        except Exception as e:
            handle_exception(e)
            return False

    data['process_submission'] = process_submission_for_duplicates

    print(f"[r/{subreddit_name}] Starting initial scan...")
    try:
        async for submission in subreddit.new(limit=20):
            if isinstance(submission, asyncpraw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                print(f"[r/{subreddit_name}] Indexing submission (initial scan): ", submission.url)
                try:
                    _, hash_value, descriptors, features = await load_and_process_image(submission.url)
                    if hash_value not in data['image_hashes']:
                        data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                        data['orb_descriptors'][submission.id] = descriptors
                        data['ai_features'][submission.id] = features
                except Exception as e:
                    handle_exception(e)
    except Exception as e:
        handle_exception(e)

    print(f"[r/{subreddit_name}] Initial scan complete. Indexed {len(data['image_hashes'])} images.")
    print(f"[r/{subreddit_name}] Bot setup complete!\n")

# =========================
# Shared Duplicate Bot Workers
# =========================
async def run_pokemon_duplicate_bot():

    async def shared_mod_log_monitor():
        while True:
            try:
                subreddits = list(subreddit_data.keys())
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        subreddit = data['subreddit']

                        async for log_entry in subreddit.mod.log(action='removelink', limit=50):
                            if log_entry.id in data['processed_log_items']:
                                continue

                            data['processed_log_items'].add(log_entry.id)
                            removed_submission_id = log_entry.target_fullname.replace('t3_', '')

                            hash_to_process = None
                            for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                                if submission_id == removed_submission_id:
                                    hash_to_process = hash_value
                                    break

                            if hash_to_process and hash_to_process not in data['moderator_removed_hashes']:
                                data['moderator_removed_hashes'].add(hash_to_process)
                                print(f"[r/{subreddit_name}] [MOD REMOVE] Submission {removed_submission_id} removed by moderator. Hash kept for future duplicate detection.")

                            if len(data['processed_log_items']) > 1000:
                                data['processed_log_items'].clear()

                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in mod log check: {e}")
            except Exception as e:
                print(f"Error in shared mod log monitor: {e}")
            await asyncio.sleep(30)

    async def shared_removal_checker():
        while True:
            try:
                subreddits = list(subreddit_data.keys())
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        current_check_time = time.time()
                        checked_this_cycle = 0

                        recent_submissions = []
                        medium_submissions = []
                        old_submissions = []

                        for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                            if hash_value in data['moderator_removed_hashes']: continue
                            age = current_check_time - creation_time
                            last_check = data['last_checked'].get(submission_id, 0)

                            if age < 3600 and current_check_time - last_check >= 30:
                                recent_submissions.append((hash_value, submission_id))
                            elif age < 86400 and current_check_time - last_check >= 300:
                                medium_submissions.append((hash_value, submission_id))
                            elif age >= 86400 and current_check_time - last_check >= 1800:
                                old_submissions.append((hash_value, submission_id))

                        all_to_check = recent_submissions + medium_submissions[:20] + old_submissions[:10]

                        for hash_value, submission_id in all_to_check:
                            try:
                                original_submission = await reddit.submission(id=submission_id)
                                await original_submission.load()
                                original_author = getattr(original_submission, "author", None)

                                if original_author is None:
                                    data['image_hashes'].pop(hash_value, None)
                                    data['orb_descriptors'].pop(submission_id, None)
                                    data['ai_features'].pop(submission_id, None)
                                    data['last_checked'].pop(submission_id, None)
                                    print(f"[r/{subreddit_name}] [USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                                else:
                                    data['last_checked'][submission_id] = current_check_time

                                checked_this_cycle += 1
                                if checked_this_cycle >= 10:
                                    await asyncio.sleep(60)
                                    checked_this_cycle = 0

                            except Exception as e:
                                print(f"[r/{subreddit_name}] Error checking submission {submission_id}: {e}")
                                data['last_checked'][submission_id] = current_check_time
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in removal check: {e}")
            except Exception as e:
                print(f"Error in shared removal checker: {e}")
            await asyncio.sleep(60)

    async def shared_modqueue_worker():
        while True:
            try:
                subreddits = list(subreddit_data.keys())
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        subreddit = data['subreddit']

                        modqueue_submissions = []
                        async for s in subreddit.mod.modqueue(only='submission', limit=None):
                            modqueue_submissions.append(s)

                        modqueue_submissions = sorted(modqueue_submissions, key=lambda x: getattr(x, 'created_utc', 0))

                        for submission in modqueue_submissions:
                            if not isinstance(submission, asyncpraw.models.Submission): continue
                            print(f"[r/{subreddit_name}] Scanning Mod Queue: ", submission.url)

                            if getattr(submission, "num_reports", 0) > 0:
                                print(f"[r/{subreddit_name}] Skipping reported image: ", submission.url)
                                data['image_hashes'] = {k: v for k, v in data['image_hashes'].items() if v[0] != submission.id}
                                data['orb_descriptors'].pop(submission.id, None)
                                data['ai_features'].pop(submission.id, None)
                                continue

                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                await data['process_submission'](submission, context="modqueue")
                                data['processed_modqueue_submissions'].add(submission.id)
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in modqueue worker: {e}")
            except Exception as e:
                print(f"Error in shared modqueue worker: {e}")
            await asyncio.sleep(15)

    async def shared_stream_worker():
        while True:
            try:
                subreddits = list(subreddit_data.keys())
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        subreddit = data['subreddit']

                        async for submission in subreddit.new(limit=10):
                            if submission.created_utc > data['current_time'] and isinstance(submission, asyncpraw.models.Submission):
                                if submission.id in data['processed_modqueue_submissions']:
                                    continue

                                print(f"[r/{subreddit_name}] Scanning new image/post: ", submission.url)

                                if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                    await data['process_submission'](submission, context="stream")

                        data['current_time'] = int(time.time())
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in stream worker: {e}")
            except Exception as e:
                print(f"Error in shared stream worker: {e}")
            await asyncio.sleep(20)

    # Dispatch shared async workers
    asyncio.create_task(shared_mod_log_monitor())
    asyncio.create_task(shared_removal_checker())
    asyncio.create_task(shared_modqueue_worker())
    asyncio.create_task(shared_stream_worker())

    print("=== Multi-subreddit duplicate bot started ===")
    print("Running with lightweight async coroutines for all subreddits")
    while True:
        await asyncio.sleep(3600)  # Keep the main duplicate bot logic alive

# =========================
# Main: initialize connections and gather loops
# =========================
async def main():
    global reddit, http_session

    # Initialize a single shared Reddit instance
    reddit = asyncpraw.Reddit(
        client_id='',
        client_secret='',
        username='',
        password='',
        user_agent=''
    )

    # Initialize a single shared HTTP session for downloads
    http_session = aiohttp.ClientSession()

    print("[SYSTEM] Starting all background services concurrently...")

    try:
        await asyncio.gather(
            safe_run(run_pokemon_duplicate_bot),
            safe_run(accept_invites_and_register)
        )
    finally:
        # Graceful cleanup on shutdown
        await http_session.close()
        await reddit.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Bot shutdown initiated by user.")
