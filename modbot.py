import praw
import prawcore.exceptions
import requests
import re
import time
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import imagehash
import cv2
import threading
import traceback
import torch
import torchvision.models as models
import torchvision.transforms as T

# =========================
# Constants
# =========================
DATE_FMT = '%Y-%m-%d %H:%M:%S'

# =========================
# Modqueue action thresholds
# =========================
APPROVE_THRESHOLDS = {
    'This is misinformation': 1,
    'This is spam': 1,
}

COMMENT_REMOVE_THRESHOLDS = {
    'No Linking to Downloadable Content in Posts or Comments': 1,
    'No ROMs, ISOs, or Game Files Sharing or Requests': 1,
    'No insults or harassment of other subreddit members in the comments': 1,
}

SUBMISSION_REMOVE_THRESHOLDS = {
    'Users Are Responsible for the Content They Post': 2,
    'Discussion-Only for Leaks, Not Distribution': 2,
    'No Linking to Downloadable Content in Posts or Comments': 1,
    'No ROMs, ISOs, or Game Files Sharing or Requests': 2,
    'Theories, Questions, Speculations must be commented in the Theory/Speculation/Question Megathread': 2,
    'Content Must Relate to Pokémon Leaks or News': 2,
    'Content must not contain any profanities, vulgarity, sexual content, slurs, be appropriate in nature': 2,
    'Post title should include sourcing and must be transparent': 2,
    'Posts with spoilers must have required spoiler flair, indicate spoiler alert in title, and be vague': 3,
    'No reposting of posts already up on the subreddit': 2,
    'No Self Advertisements or Promotion': 2,
    'No Memes, Fan Art, or Joke Posts': 2,
}

# =========================
# Reddit init & Thread Safety
# =========================
def initialize_reddit():
    return praw.Reddit(
        client_id='jl-I3OHYH2_VZMC1feoJMQ',
        client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
        username='PokeLeakBot3',
        password='testbot1',
        user_agent='testbot',
    )

_thread_local = threading.local()

def get_reddit():
    """Returns a thread-local PRAW instance to prevent Session collisions."""
    if not hasattr(_thread_local, 'reddit'):
        _thread_local.reddit = initialize_reddit()
    return _thread_local.reddit

# =========================
# Error handling
# =========================
def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and getattr(e, "response", None) and e.response.status_code == 429:
        print("[WARN] Rate limited by Reddit API.")
    else:
        print(f"[ERROR] {type(e).__name__}: {e}")

# =========================
# Crash-proof runner
# =========================
def safe_run(target, *args, **kwargs):
    while True:
        try:
            target(*args, **kwargs)
        except Exception as e:
            print(f"[FATAL] {target.__name__} crashed: {e}")
            traceback.print_exc()
            time.sleep(10)

# =========================
# Module-level image utility
# =========================
def _fetch_image(url):
    """Fetch an image URL and return it as an OpenCV BGR array."""
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status() # Throws exception on 404/500
        
        img = cv2.imdecode(np.asarray(bytearray(resp.content), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None (invalid or corrupted image format).")
        return img
    except Exception as e:
        print(f"[WARN] Failed to fetch or decode image {url}: {e}")
        return None

# =========================
# Module-level data eviction
# =========================
def _evict_submission(data, submission_id, hash_value=None):
    """Remove a submission's entries from all per-subreddit data stores."""
    if hash_value is None:
        hash_value = next(
            (h for h, (sid, _) in data['image_hashes'].items() if sid == submission_id),
            None,
        )
    if hash_value is not None:
        data['image_hashes'].pop(hash_value, None)
    data['orb_descriptors'].pop(submission_id, None)
    data['ai_features'].pop(submission_id, None)
    data['last_checked'].pop(submission_id, None)

# =========================
# Workers
# =========================

def monitor_reported_posts():
    reddit = get_reddit()
    # Uses "mod" to check reports across all moderated subreddits
    subreddit = reddit.subreddit("mod")
    while True:
        try:
            for post in subreddit.mod.reports():
                if getattr(post, "approved", False):
                    post.mod.approve()
                    print(f"[r/{post.subreddit.display_name}] Post {post.id} has been approved again")
        except Exception as e:
            handle_exception(e)
            time.sleep(60)


def handle_modqueue_items():
    """Timer-based auto-approval for posts with exactly one report after one hour."""
    reddit = get_reddit()
    timers = {}
    while True:
        try:
            for item in reddit.subreddit("mod").mod.modqueue():
                if getattr(item, "num_reports", 0) == 1 and item.id not in timers:
                    timers[item.id] = time.time()
                    print(f"[r/{item.subreddit.display_name}] Starting timer for post {item.id} (created {getattr(item, 'created_utc', time.time())})...")

                if item.id in timers:
                    if time.time() - timers[item.id] >= 3600:
                        try:
                            item.mod.approve()
                            print(f"[r/{item.subreddit.display_name}] Approved post {item.id} with one report")
                            del timers[item.id]
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
                    else:
                        time_remaining = int(timers[item.id] + 3600 - time.time())
                        if time_remaining % 300 == 0:  # Only print every 5 minutes to avoid log spam
                            print(f"[r/{item.subreddit.display_name}] Time remaining for post {item.id}: {time_remaining} seconds")
        except Exception as e:
            handle_exception(e)
            time.sleep(60)


def handle_spoiler_status():
    reddit = get_reddit()
    subreddit = reddit.subreddit("mod")
    previous_spoiler_status = {}
    mod_cache = {}
    
    while True:
        try:
            for submission in subreddit.new(limit=100):
                sub_name = submission.subreddit.display_name
                
                # Dynamically cache moderators per subreddit to prevent rate limits
                if sub_name not in mod_cache:
                    mod_cache[sub_name] = {mod.name for mod in submission.subreddit.moderator()}

                if submission.id not in previous_spoiler_status:
                    previous_spoiler_status[submission.id] = submission.spoiler
                    continue

                if previous_spoiler_status[submission.id] != submission.spoiler:
                    # Guard against deleted users (author is None)
                    is_moderator = bool(submission.author and submission.author.name in mod_cache[sub_name])

                    if not submission.spoiler:
                        if not is_moderator:
                            try:
                                print(f'[r/{sub_name}] Post {submission.id} unmarked as spoiler by non-mod. Re-marking.')
                                submission.mod.spoiler()
                            except prawcore.exceptions.ServerError as se:
                                handle_exception(se)
                        else:
                            print(f'[r/{sub_name}] Post {submission.id} unmarked as spoiler by a moderator. Leaving as-is.')
                    previous_spoiler_status[submission.id] = submission.spoiler
        except Exception as e:
            handle_exception(e)
            time.sleep(30)


def process_modqueue():
    """
    Consolidated replacement for all modqueue approval/removal functions.
    """
    reddit = get_reddit()
    while True:
        try:
            for item in reddit.subreddit("mod").mod.modqueue(limit=100):
                user_reports = getattr(item, "user_reports", None)
                if not user_reports:
                    continue
                reason = user_reports[0][0]
                count = user_reports[0][1]
                sub_name = item.subreddit.display_name
                try:
                    if reason in APPROVE_THRESHOLDS and count >= APPROVE_THRESHOLDS[reason]:
                        item.mod.approve()
                        label = getattr(item, 'title', None) or getattr(item, 'body', str(item))
                        print(f'[r/{sub_name}] Item "{label}" approved: {count}× "{reason}"')
                    elif isinstance(item, praw.models.Comment) and reason in COMMENT_REMOVE_THRESHOLDS and count >= COMMENT_REMOVE_THRESHOLDS[reason]:
                        item.mod.remove()
                        print(f'[r/{sub_name}] Comment "{item.body}" removed: {count}× "{reason}"')
                    elif isinstance(item, praw.models.Submission) and reason in SUBMISSION_REMOVE_THRESHOLDS and count >= SUBMISSION_REMOVE_THRESHOLDS[reason]:
                        item.mod.remove()
                        print(f'[r/{sub_name}] Submission "{item.title}" removed: {count}× "{reason}"')
                except prawcore.exceptions.ServerError as se:
                    handle_exception(se)
        except Exception as e:
            handle_exception(e)
            time.sleep(60)


# =========================
# Duplicate detection bot
# =========================
def run_pokemon_duplicate_bot():
    subreddit_data = {}
    subreddit_data_lock = threading.Lock()

    # --- AI models: initialized once for the whole bot ---
    device = "cpu"
    efficientnet_model = models.efficientnet_b0(pretrained=True)
    efficientnet_model.eval()
    efficientnet_model.to(device)
    efficientnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    orb_detector = cv2.ORB_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # --- Shared helpers ---

    def get_ai_features(img):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_tensor = efficientnet_transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = efficientnet_model(img_tensor)
                feat = feat / feat.norm(dim=1, keepdim=True)
            return feat
        except Exception as e:
            print(f"[AI] Feature extraction error: {e}")
            return None

    def _get_orb_input(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if np.mean(gray > 240) > 0.7:
            _, masked = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
            return cv2.Canny(masked, 100, 200)

        if np.mean(cv2.Canny(gray, 100, 200) > 0) > 0.05:
            _, masked = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
            return cv2.Canny(masked, 100, 200)

        return gray

    def get_orb_descriptors(img):
        _, des = orb_detector.detectAndCompute(_get_orb_input(img), None)
        return des

    def orb_similarity(desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0
        matches = bf_matcher.match(desc1, desc2)
        return len(matches) / min(len(desc1), len(desc2))

    def calculate_ai_similarity(features1, features2):
        if features1 is not None and features2 is not None:
            return (features1 @ features2.T).item()
        return 0

    def format_age(utc_timestamp):
        delta = datetime.now(timezone.utc) - datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
        days, seconds = delta.days, delta.seconds
        if days > 0:
            return f"{days} day{'s' if days != 1 else ''} ago"
        if seconds >= 3600:
            h = seconds // 3600
            return f"{h} hour{'s' if h != 1 else ''} ago"
        if seconds >= 60:
            m = seconds // 60
            return f"{m} minute{'s' if m != 1 else ''} ago"
        return f"{seconds} second{'s' if seconds != 1 else ''} ago"

    def _snapshot_subreddits():
        """Thread-safe snapshot of current (name, data) pairs."""
        with subreddit_data_lock:
            return list(subreddit_data.items())

    # --- Per-subreddit setup ---

    def setup_subreddit(subreddit_name):
        print(f"\n=== Setting up bot for r/{subreddit_name} ===")
        reddit = get_reddit()
        subreddit = reddit.subreddit(subreddit_name)
        
        data = {
            'subreddit_name': subreddit_name,
            'image_hashes': {},
            'orb_descriptors': {},
            'moderator_removed_hashes': set(),
            'processed_modqueue_submissions': set(),
            'approved_by_moderator': set(),
            'ai_features': {},
            'current_time': int(time.time()),
            'processed_log_items': set(),
            'last_checked': {},
        }
        with subreddit_data_lock:
            subreddit_data[subreddit_name] = data

        def load_and_process_image(url):
            img = _fetch_image(url)
            if img is None:
                raise ValueError(f"Unprocessable image from {url}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hash_value = str(imagehash.phash(Image.fromarray(gray)))
            print(f"[r/{subreddit_name}] Generated hash: {hash_value}")
            return img, hash_value, get_orb_descriptors(img), get_ai_features(img)

        def get_cached_ai_features(submission_id):
            if submission_id in data['ai_features']:
                return data['ai_features'][submission_id]
            old_submission = get_reddit().submission(id=submission_id)
            features = get_ai_features(_fetch_image(old_submission.url))
            data['ai_features'][submission_id] = features
            return features

        def _build_original_info(original_submission, hash_value=None):
            return (
                original_submission.author.name if original_submission.author else "[Deleted]",
                original_submission.title,
                datetime.utcfromtimestamp(original_submission.created_utc).strftime(DATE_FMT),
                original_submission.created_utc,
                "Removed by Moderator" if hash_value and hash_value in data['moderator_removed_hashes'] else "Active",
                original_submission.permalink,
            )

        def post_comment(submission, original_post_author, original_post_title,
                         original_post_date, original_post_utc, original_status,
                         original_post_permalink):
            age_text = format_age(original_post_utc)
            for _ in range(3):
                try:
                    comment_text = (
                        "> **Duplicate detected**\n\n"
                        "| Original Author | Title | Date | Age | Status |\n"
                        "|:---------------:|:-----:|:----:|:---:|:------:|\n"
                        f"| {original_post_author} | [{original_post_title}]({original_post_permalink}) "
                        f"| {original_post_date} | {age_text} | {original_status} |"
                    )
                    comment = submission.reply(comment_text)
                    comment.mod.distinguish(sticky=True)
                    print(f"[r/{subreddit_name}] Duplicate comment posted: {submission.url}")
                    return True
                except Exception as e:
                    handle_exception(e)
                    time.sleep(1)
            return False

        def check_hash_duplicate(submission, hash_value, new_features):
            matched_hash = next(
                (h for h in data['image_hashes']
                 if hash_value == h or (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(h)) <= 3),
                None,
            )
            if matched_hash is None:
                return False, None, None, None, None, None, None

            original_id, original_time = data['image_hashes'][matched_hash]
            if submission.id == original_id or submission.created_utc <= original_time:
                return False, None, None, None, None, None, None

            original_submission = get_reddit().submission(id=original_id)
            ai_score = calculate_ai_similarity(new_features, get_cached_ai_features(original_id))
            print(f"[r/{subreddit_name}] Hash match. AI similarity: {ai_score:.2f}")

            if ai_score > 0.50:
                return (True,) + _build_original_info(original_submission, matched_hash)
            return False, None, None, None, None, None, None

        def check_orb_duplicate(submission, descriptors, new_features):
            for old_id, old_desc in data['orb_descriptors'].items():
                if orb_similarity(descriptors, old_desc) > 0.50:
                    ai_score = calculate_ai_similarity(new_features, get_cached_ai_features(old_id))
                    if ai_score > 0.75:
                        print(f"[r/{subreddit_name}] ORB duplicate. AI similarity: {ai_score:.2f}")
                        original_submission = get_reddit().submission(id=old_id)
                        old_hash = next((h for h, v in data['image_hashes'].items() if v[0] == old_id), None)
                        return (True,) + _build_original_info(original_submission, old_hash)
            return False, None, None, None, None, None, None

        def handle_duplicate(submission, detection_method, author, title, date, utc, status, permalink):
            if not submission.approved:
                submission.mod.remove()
                post_comment(submission, author, title, date, utc, status, permalink)
                print(f"[r/{subreddit_name}] Duplicate removed by {detection_method}: {submission.url}")
            return True

        def handle_moderator_removed_repost(submission, hash_value):
            if hash_value in data['moderator_removed_hashes'] and not submission.approved:
                submission.mod.remove()
                original_submission = get_reddit().submission(id=data['image_hashes'][hash_value][0])
                post_comment(submission, *_build_original_info(original_submission, hash_value))
                print(f"[r/{subreddit_name}] Repost of mod-removed image removed: {submission.url}")
                return True
            return False

        def process_submission_for_duplicates(submission, context="stream"):
            try:
                _, hash_value, descriptors, new_features = load_and_process_image(submission.url)
                data['ai_features'][submission.id] = new_features

                if handle_moderator_removed_repost(submission, hash_value):
                    return True

                checks = [
                    ("hash + AI", lambda: check_hash_duplicate(submission, hash_value, new_features)),
                    ("ORB + AI",  lambda: check_orb_duplicate(submission, descriptors, new_features)),
                ]
                for method, check_fn in checks:
                    is_dup, author, title, date, utc, status, permalink = check_fn()
                    if is_dup:
                        return handle_duplicate(submission, method, author, title, date, utc, status, permalink)

                if hash_value not in data['image_hashes']:
                    data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                    data['orb_descriptors'][submission.id] = descriptors
                    data['ai_features'][submission.id] = new_features
                    print(f"[r/{subreddit_name}] Stored new original: {submission.url}")

                if context == "modqueue" and not submission.approved:
                    submission.mod.approve()
                    print(f"[r/{subreddit_name}] Original approved from modqueue: {submission.url}")

                return False
            except Exception as e:
                handle_exception(e)
                return False

        data['process_submission'] = process_submission_for_duplicates

        print(f"[r/{subreddit_name}] Starting initial scan...")
        try:
            for submission in subreddit.new(limit=20000):
                if isinstance(submission, praw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    print(f"[r/{subreddit_name}] Indexing (initial scan): {submission.url}")
                    try:
                        _, hash_value, descriptors, features = load_and_process_image(submission.url)
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

    # --- Shared worker threads ---

    def shared_mod_log_monitor():
        while True:
            try:
                for subreddit_name, data in _snapshot_subreddits():
                    try:
                        local_sub = get_reddit().subreddit(data['subreddit_name'])
                        for log_entry in local_sub.mod.log(action='removelink', limit=50):
                            if log_entry.id in data['processed_log_items']:
                                continue
                            data['processed_log_items'].add(log_entry.id)
                            removed_id = log_entry.target_fullname.replace('t3_', '')
                            hash_to_flag = next(
                                (h for h, (sid, _) in data['image_hashes'].items() if sid == removed_id),
                                None,
                            )
                            if hash_to_flag and hash_to_flag not in data['moderator_removed_hashes']:
                                data['moderator_removed_hashes'].add(hash_to_flag)
                                print(f"[r/{subreddit_name}] [MOD REMOVE] {removed_id} flagged for duplicate detection.")
                            if len(data['processed_log_items']) > 1000:
                                data['processed_log_items'].clear()
                    except Exception as e:
                        handle_exception(e)
            except Exception as e:
                handle_exception(e)
            time.sleep(30)

    def shared_removal_checker():
        while True:
            try:
                for subreddit_name, data in _snapshot_subreddits():
                    try:
                        now = time.time()
                        recent, medium, old = [], [], []
                        for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                            if hash_value in data['moderator_removed_hashes']:
                                continue
                            age = now - creation_time
                            last = data['last_checked'].get(submission_id, 0)
                            if age < 3600 and now - last >= 30:
                                recent.append((hash_value, submission_id))
                            elif age < 86400 and now - last >= 300:
                                medium.append((hash_value, submission_id))
                            elif age >= 86400 and now - last >= 1800:
                                old.append((hash_value, submission_id))

                        checked = 0
                        for hash_value, submission_id in recent + medium[:20] + old[:10]:
                            try:
                                if get_reddit().submission(id=submission_id).author is None:
                                    _evict_submission(data, submission_id, hash_value)
                                    print(f"[r/{subreddit_name}] [USER DELETE] {submission_id} removed from index.")
                                else:
                                    data['last_checked'][submission_id] = now
                                checked += 1
                                if checked >= 10:
                                    time.sleep(60)
                                    checked = 0
                            except Exception as e:
                                handle_exception(e)
                                data['last_checked'][submission_id] = now
                    except Exception as e:
                        handle_exception(e)
            except Exception as e:
                handle_exception(e)
            time.sleep(60)

    def shared_modqueue_worker():
        while True:
            try:
                for subreddit_name, data in _snapshot_subreddits():
                    try:
                        local_sub = get_reddit().subreddit(data['subreddit_name'])
                        submissions = sorted(
                            (s for s in local_sub.mod.modqueue(only='submissions', limit=None)
                             if isinstance(s, praw.models.Submission)),
                            key=lambda x: x.created_utc,
                        )
                        for submission in submissions:
                            print(f"[r/{subreddit_name}] Scanning modqueue: {submission.url}")
                            if submission.num_reports > 0:
                                print(f"[r/{subreddit_name}] Skipping reported image: {submission.url}")
                                _evict_submission(data, submission.id)
                                continue
                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                data['process_submission'](submission, context="modqueue")
                                data['processed_modqueue_submissions'].add(submission.id)
                    except Exception as e:
                        handle_exception(e)
            except Exception as e:
                handle_exception(e)
            time.sleep(15)

    def shared_stream_worker():
        while True:
            try:
                for subreddit_name, data in _snapshot_subreddits():
                    try:
                        local_sub = get_reddit().subreddit(data['subreddit_name'])
                        for submission in local_sub.new(limit=10):
                            if (submission.created_utc > data['current_time']
                                    and isinstance(submission, praw.models.Submission)
                                    and submission.id not in data['processed_modqueue_submissions']):
                                print(f"[r/{subreddit_name}] Scanning new post: {submission.url}")
                                if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                    data['process_submission'](submission, context="stream")
                        data['current_time'] = int(time.time())
                    except Exception as e:
                        handle_exception(e)
            except Exception as e:
                handle_exception(e)
            time.sleep(20)

    def _spawn_setup(name):
        if name in subreddit_data:
            return
        def _run():
            try:
                setup_subreddit(name)
            except Exception as e:
                print(f"[setup] Error initializing {name}: {e}")
        threading.Thread(target=_run, daemon=True).start()
        print(f"[invite] Setup thread started for r/{name}")

    def check_for_invites():
        """Checks the bot's inbox for moderator invitations and accepts them."""
        reddit = get_reddit()
        while True:
            try:
                # 1. Iterate through unread messages in the bot's inbox
                for message in reddit.inbox.unread(limit=50):
                    # 2. Verify the item is a private message
                    if isinstance(message, praw.models.Message) and "invitation to moderate" in message.subject.lower():
                        # 3. Ensure PRAW attached the Subreddit object to the message
                        if message.subreddit:
                            try:
                                # Accept the invitation directly
                                message.subreddit.mod.accept_invite()
                                print(f"✅ Accepted invite to moderate r/{message.subreddit.display_name}")
                                message.mark_read()
                                # Trigger duplicate scan setup for the new subreddit
                                _spawn_setup(message.subreddit.display_name)
                            except Exception as e:
                                print(f"Failed to accept invite for r/{message.subreddit.display_name}: {e}")
                
                # Check current moderated subs as a backup
                for subreddit in reddit.user.me().moderated():
                    if subreddit.display_name not in subreddit_data:
                        _spawn_setup(subreddit.display_name)

            except Exception as e:
                handle_exception(e)
            
            # Check every 5 minutes
            time.sleep(300)

    for target in (check_for_invites, shared_mod_log_monitor, shared_removal_checker,
                   shared_modqueue_worker, shared_stream_worker):
        threading.Thread(target=target, daemon=True).start()

    print("=== Multi-subreddit bot started ===")
    print("Running with 5 shared worker threads for all subreddits")
    print("Monitoring for mod invites...")
    while True:
        time.sleep(20)


# =========================
# Main: start threads via safe_run
# =========================
if __name__ == "__main__":
    threads = {}

    def add_thread(name, func, *args, **kwargs):
        t = threading.Thread(target=safe_run, args=(func,) + args, kwargs=kwargs, daemon=True)
        t.start()
        threads[name] = t
        print(f"[STARTED] {name}")

    add_thread('modqueue_timer_thread',   handle_modqueue_items)
    add_thread('reported_posts_thread',   monitor_reported_posts)
    add_thread('spoiler_status_thread',   handle_spoiler_status)
    add_thread('modqueue_actions_thread', process_modqueue)
    add_thread('duplicate_bot_thread',    run_pokemon_duplicate_bot)

    while True:
        time.sleep(30)
