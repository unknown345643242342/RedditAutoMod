
import praw
import prawcore.exceptions
import requests
import time
from datetime import datetime
import numpy as np
from PIL import Image
import imagehash
import cv2
import threading
import traceback
import pytesseract
import easyocr
import openai
from openai import OpenAI
from torchvision import transforms
import torch
import torchvision.models as models
import torchvision.transforms as T
import hashlib
import difflib as _difflib
from io import BytesIO
from datetime import datetime, timezone

# =========================
# Crash-proof runner
# =========================
def safe_run(target, *args, **kwargs):
    """
    Keeps a target function running forever.
    If the function raises, log the error, sleep briefly, and run it again.
    """
    while True:
        try:
            target(*args, **kwargs)
        except Exception as e:
            print(f"[FATAL] {target.__name__} crashed: {e}")
            traceback.print_exc()
            time.sleep(5)  # brief cooldown before retrying

# =========================
# Reddit init + error handler
# =========================
def initialize_reddit():
    return praw.Reddit(
        client_id='jl-I3OHYH2_VZMC1feoJMQ',
        client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
        username='PokeLeakBot3',
        password='testbot1',
        user_agent='testbot'
    )

def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and getattr(e, "response", None) and e.response.status_code == 429:
        print("Rate limited by Reddit API. Ignoring error.")
    else:
        print(f"Exception encountered: {str(e)}")

# =========================
# Global storage for all subreddits
# =========================
subreddit_data = {}
subreddit_data_lock = threading.Lock()

# =========================
# Moderator invite acceptor
# =========================
def accept_moderator_invites():
    """Continuously monitors and accepts moderator invites"""
    reddit = initialize_reddit()
    processed_invites = set()
    
    while True:
        try:
            for invite in reddit.subreddit('mod').moderator.invitations():
                invite_id = f"{invite.subreddit.display_name}_{invite.id}"
                
                if invite_id not in processed_invites:
                    try:
                        invite.accept()
                        subreddit_name = invite.subreddit.display_name
                        print(f"[INVITE ACCEPTED] Accepted moderator invite for r/{subreddit_name}")
                        
                        # Initialize subreddit monitoring
                        with subreddit_data_lock:
                            if subreddit_name not in subreddit_data:
                                subreddit_data[subreddit_name] = {
                                    'image_hashes': {},
                                    'orb_descriptors': {},
                                    'moderator_removed_hashes': set(),
                                    'processed_modqueue_submissions': set(),
                                    'approved_by_moderator': set(),
                                    'ai_features': {}
                                }
                                # Start monitoring this subreddit
                                threading.Thread(
                                    target=safe_run,
                                    args=(run_pokemon_duplicate_bot, subreddit_name),
                                    daemon=True
                                ).start()
                                print(f"[STARTED] Monitoring r/{subreddit_name}")
                        
                        processed_invites.add(invite_id)
                    except Exception as e:
                        print(f"[ERROR] Failed to accept invite for r/{invite.subreddit.display_name}: {e}")
            
            time.sleep(30)  # Check for invites every 30 seconds
            
        except Exception as e:
            handle_exception(e)
            time.sleep(30)

# =========================
# Workers
def run_pokemon_duplicate_bot(subreddit_name):
    reddit = initialize_reddit()
    subreddit = reddit.subreddit(subreddit_name)
    
    # Get this subreddit's data dictionaries
    with subreddit_data_lock:
        sub_data = subreddit_data[subreddit_name]
    
    image_hashes = sub_data['image_hashes']
    orb_descriptors = sub_data['orb_descriptors']
    moderator_removed_hashes = sub_data['moderator_removed_hashes']
    processed_modqueue_submissions = sub_data['processed_modqueue_submissions']
    approved_by_moderator = sub_data['approved_by_moderator']
    ai_features = sub_data['ai_features']
    current_time = int(time.time())

    # --- Tiny AI similarity model ---
    device = "cpu"
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.eval()
    resnet_model.to(device)
    resnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Helper functions ---
    def get_ai_features(img):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_tensor = resnet_transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = resnet_model(img_tensor)
                feat = feat / feat.norm(dim=1, keepdim=True)
            return feat
        except Exception as e:
            print(f"[r/{subreddit_name}] AI feature extraction error:", e)
            return None

    def is_problematic_image(img, white_threshold=0.7, text_threshold=0.05):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_ratio = np.mean(gray > 240)
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.mean(edges > 0)
        return white_ratio > white_threshold or edge_ratio > text_threshold

    def preprocess_image_for_orb(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def get_orb_descriptors_conditional(img):
        if is_problematic_image(img):
            processed_img = preprocess_image_for_orb(img)
        else:
            processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(processed_img, None)
        return des

    def orb_similarity(desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        return len(matches) / min(len(desc1), len(desc2))

    def format_age(utc_timestamp):
        now = datetime.now(timezone.utc)
        created = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
        delta = now - created
        days = delta.days
        seconds = delta.seconds
        if days > 0:
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif seconds >= 3600:
            hours = seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds >= 60:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return f"{seconds} second{'s' if seconds != 1 else ''} ago"

    def post_comment(submission, original_post_author, original_post_title, original_post_date, original_post_utc, original_status, original_post_permalink):
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
                comment = submission.reply(comment_text)
                comment.mod.distinguish(sticky=True)
                print(f"[r/{subreddit_name}] Duplicate removed and comment posted: ", submission.url)
                return True
            except Exception as e:
                handle_exception(e)
                retries += 1
                time.sleep(1)
        return False

    # --- NEW: Consolidated helper functions ---
    def load_and_process_image(url):
        """Load image from URL and compute hash, descriptors, and AI features"""
        image_data = requests.get(url).content
        img = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
        descriptors = get_orb_descriptors_conditional(img)
        features = get_ai_features(img)
        return img, hash_value, descriptors, features

    def get_cached_ai_features(submission_id):
        """Get AI features from cache or compute them"""
        if submission_id in ai_features:
            return ai_features[submission_id]
        
        old_submission = reddit.submission(id=submission_id)
        old_image_data = requests.get(old_submission.url).content
        old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        old_features = get_ai_features(old_img)
        ai_features[submission_id] = old_features
        return old_features

    def calculate_ai_similarity(features1, features2):
        """Calculate AI similarity score between two feature vectors"""
        if features1 is not None and features2 is not None:
            return (features1 @ features2.T).item()
        return 0

    def check_hash_duplicate(submission, hash_value, new_features):
        """Check if submission is a hash-based duplicate"""
        if hash_value not in image_hashes:
            return False, None, None, None, None, None, None
        
        original_id, original_time = image_hashes[hash_value]
        
        # Skip if same submission or older
        if submission.id == original_id or submission.created_utc <= original_time:
            return False, None, None, None, None, None, None
        
        original_submission = reddit.submission(id=original_id)
        original_features = get_cached_ai_features(original_submission.id)
        ai_score = calculate_ai_similarity(new_features, original_features)
        
        print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f}")
        
        if ai_score > 0.70:
            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            original_status = "Removed by Moderator" if hash_value in moderator_removed_hashes else "Active"
            return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def check_orb_duplicate(submission, descriptors, new_features):
        """Check if submission is an ORB-based duplicate"""
        for old_id, old_desc in orb_descriptors.items():
            sim = orb_similarity(descriptors, old_desc)
            
            if sim > 0.30:
                old_features = get_cached_ai_features(old_id)
                ai_score = calculate_ai_similarity(new_features, old_features)
                
                if ai_score > 0.70:
                    original_submission = reddit.submission(id=old_id)
                    original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    old_hash = next((h for h, v in image_hashes.items() if v[0] == old_id), None)
                    original_status = "Removed by Moderator" if old_hash and old_hash in moderator_removed_hashes else "Active"
                    
                    return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def handle_duplicate(submission, is_hash_dup, detection_method, author, title, date, utc, status, permalink):
        """Remove duplicate and post comment if not approved"""
        if not submission.approved:
            submission.mod.remove()
            post_comment(submission, author, title, date, utc, status, permalink)
            print(f"[r/{subreddit_name}] Duplicate removed by {detection_method}: {submission.url}")
        return True

    def handle_moderator_removed_repost(submission, hash_value):
        """Handle reposts of moderator-removed images"""
        if hash_value in moderator_removed_hashes and not submission.approved:
            submission.mod.remove()
            original_submission = reddit.submission(id=image_hashes[hash_value][0])
            post_comment(
                submission,
                original_submission.author.name,
                original_submission.title,
                datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                original_submission.created_utc,
                "Removed by Moderator",
                original_submission.permalink
            )
            print(f"[r/{subreddit_name}] Repost of a moderator-removed image removed: ", submission.url)
            return True
        return False

    def process_submission_for_duplicates(submission, context="stream"):
        """Main duplicate detection logic - works for both mod queue and stream"""
        try:
            img, hash_value, descriptors, new_features = load_and_process_image(submission.url)
            ai_features[submission.id] = new_features
            
            # Check for moderator-removed reposts first
            if handle_moderator_removed_repost(submission, hash_value):
                return True
            
            # Check hash-based duplicates
            is_duplicate, author, title, date, utc, status, permalink = check_hash_duplicate(
                submission, hash_value, new_features
            )
            if is_duplicate:
                return handle_duplicate(submission, True, "hash + AI", author, title, date, utc, status, permalink)
            
            # Check ORB-based duplicates
            is_duplicate, author, title, date, utc, status, permalink = check_orb_duplicate(
                submission, descriptors, new_features
            )
            if is_duplicate:
                return handle_duplicate(submission, False, "ORB + AI", author, title, date, utc, status, permalink)
            
            # Not a duplicate - approve if in mod queue and store data
            if context == "modqueue" and not submission.approved:
                submission.mod.approve()
                print(f"[r/{subreddit_name}] Original submission approved: ", submission.url)
            
            if hash_value not in image_hashes:
                image_hashes[hash_value] = (submission.id, submission.created_utc)
                orb_descriptors[submission.id] = descriptors
                ai_features[submission.id] = new_features
            
            return False
            
        except Exception as e:
            handle_exception(e)
            return False

    def check_removed_original_posts():
        """Monitor for immediate removal detection using dual approach"""
        processed_log_items = set()
        last_checked = {}
        
        # Thread for monitoring mod log (immediate mod removals)
        def monitor_mod_log():
            while True:
                try:
                    for log_entry in subreddit.mod.stream.log(action='removelink', skip_existing=True):
                        if log_entry.id in processed_log_items:
                            continue
                        
                        processed_log_items.add(log_entry.id)
                        removed_submission_id = log_entry.target_fullname.replace('t3_', '')
                        
                        # Find the hash for this submission
                        hash_to_process = None
                        for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                            if submission_id == removed_submission_id:
                                hash_to_process = hash_value
                                break
                        
                        if hash_to_process and hash_to_process not in moderator_removed_hashes:
                            moderator_removed_hashes.add(hash_to_process)
                            print(f"[r/{subreddit_name}] [MOD REMOVE] Submission {removed_submission_id} removed by moderator. Hash kept for future duplicate detection.")
                        
                        # Limit processed log items
                        if len(processed_log_items) > 1000:
                            processed_log_items.clear()
                    
                except Exception as e:
                    print(f"[r/{subreddit_name}] Error in mod log monitor: {e}")
                    time.sleep(5)
        
        # Start mod log monitor in separate thread
        threading.Thread(target=monitor_mod_log, daemon=True).start()
        
        # Main thread: prioritized check for user deletions
        while True:
            try:
                current_check_time = time.time()
                checked_this_cycle = 0
                
                # Separate submissions into priority tiers based on age
                recent_submissions = []  # < 1 hour old
                medium_submissions = []  # 1-24 hours old
                old_submissions = []     # > 24 hours old
                
                for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                    # Skip if already marked as mod-removed
                    if hash_value in moderator_removed_hashes:
                        continue
                    
                    age = current_check_time - creation_time
                    last_check = last_checked.get(submission_id, 0)
                    
                    # Determine check interval based on age
                    if age < 3600:  # Less than 1 hour old
                        check_interval = 30  # Check every 30 seconds
                        if current_check_time - last_check >= check_interval:
                            recent_submissions.append((hash_value, submission_id))
                    elif age < 86400:  # 1-24 hours old
                        check_interval = 300  # Check every 5 minutes
                        if current_check_time - last_check >= check_interval:
                            medium_submissions.append((hash_value, submission_id))
                    else:  # Older than 24 hours
                        check_interval = 1800  # Check every 30 minutes
                        if current_check_time - last_check >= check_interval:
                            old_submissions.append((hash_value, submission_id))
                
                # Process in priority order: recent first, then medium, then old
                all_to_check = recent_submissions + medium_submissions[:20] + old_submissions[:10]
                
                for hash_value, submission_id in all_to_check:
                    try:
                        original_submission = reddit.submission(id=submission_id)
                        original_author = original_submission.author
                        
                        # Check if user deleted their post
                        if original_author is None:
                            if hash_value in image_hashes:
                                del image_hashes[hash_value]
                            if submission_id in orb_descriptors:
                                del orb_descriptors[submission_id]
                            if submission_id in ai_features:
                                del ai_features[submission_id]
                            if submission_id in last_checked:
                                del last_checked[submission_id]
                            print(f"[r/{subreddit_name}] [USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                        else:
                            last_checked[submission_id] = current_check_time
                        
                        checked_this_cycle += 1
                        
                        # Rate limiting: check 10 at a time, then pause
                        if checked_this_cycle >= 10:
                            time.sleep(2)
                            checked_this_cycle = 0
                        
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error checking submission {submission_id}: {e}")
                        last_checked[submission_id] = current_check_time
                
            except Exception as e:
                handle_exception(e)
            
            time.sleep(5)  # Short 5-second pause between cycles
    threading.Thread(target=check_removed_original_posts, daemon=True).start()

    # --- Initial scan ---
    try:
        print(f"[r/{subreddit_name}] Starting initial index scan...")
        for submission in subreddit.new(limit=300):
            if isinstance(submission, praw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                print(f"[r/{subreddit_name}] Indexing submission (initial scan): ", submission.url)
                try:
                    img, hash_value, descriptors, features = load_and_process_image(submission.url)
                    if hash_value not in image_hashes:
                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                        orb_descriptors[submission.id] = descriptors
                        ai_features[submission.id] = features
                except Exception as e:
                    handle_exception(e)
        print(f"[r/{subreddit_name}] Initial index scan complete.")
    except Exception as e:
        handle_exception(e)

    # --- Mod Queue worker ---
    def modqueue_worker():
        nonlocal image_hashes, orb_descriptors, moderator_removed_hashes, processed_modqueue_submissions, ai_features
        while True:
            try:
                modqueue_submissions = subreddit.mod.modqueue(only='submission', limit=None)
                modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                for submission in modqueue_submissions:
                    if not isinstance(submission, praw.models.Submission):
                        continue
                    
                    print(f"[r/{subreddit_name}] Scanning Mod Queue: ", submission.url)
                    
                    if submission.num_reports > 0:
                        print(f"[r/{subreddit_name}] Skipping reported image: ", submission.url)
                        image_hashes = {k: v for k, v in image_hashes.items() if v[0] != submission.id}
                        orb_descriptors.pop(submission.id, None)
                        ai_features.pop(submission.id, None)
                        continue
                    
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        is_duplicate = process_submission_for_duplicates(submission, context="modqueue")
                        # Always add to processed set so it doesn't get reprocessed in stream
                        processed_modqueue_submissions.add(submission.id)

            except Exception as e:
                handle_exception(e)
            time.sleep(2)

    threading.Thread(target=modqueue_worker, daemon=True).start()

    # --- Stream new submissions ---
    while True:
        try:
            for submission in subreddit.stream.submissions(skip_existing=True):
                if submission.created_utc > current_time and isinstance(submission, praw.models.Submission):
                    # Skip originals already approved in mod queue
                    if submission.id in processed_modqueue_submissions:
                        continue

                    print(f"[r/{subreddit_name}] Scanning new image/post: ", submission.url)
                    
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        process_submission_for_duplicates(submission, context="stream")

            current_time = int(time.time())
        except Exception as e:
            handle_exception(e)
            
# =========================
# Main: start threads via safe_run
# =========================
if __name__ == "__main__":
    threads = {}

    def add_thread(name, func, *args, **kwargs):
        t = threading.Thread(target=safe_run, args=(func,)+args, kwargs=kwargs, daemon=True)
        t.start()
        threads[name] = t
        print(f"[STARTED] {name}")
    
    # Start the moderator invite acceptor
    add_thread('accept_moderator_invites_thread', accept_moderator_invites)
    
    # Start monitoring for the initial subreddit (PokeLeaks)
    with subreddit_data_lock:
        subreddit_data['PokeLeaks'] = {
            'image_hashes': {},
            'orb_descriptors': {},
            'moderator_removed_hashes': set(),
            'processed_modqueue_submissions': set(),
            'approved_by_moderator': set(),
            'ai_features': {}
        }
    add_thread('run_pokemon_duplicate_bot_thread', run_pokemon_duplicate_bot, 'PokeLeaks')

    # Keep the main thread alive indefinitely so daemon threads keep running.
    while True:
        time.sleep(30)
