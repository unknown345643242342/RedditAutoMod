import asyncpraw
import asyncpraw.exceptions
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
import openai
from openai import OpenAI
from torchvision import transforms
import torch
import torchvision.models as models
import torchvision.transforms as T
import hashlib
import difflib as _difflib
from datetime import datetime, timezone
import asyncio

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
            asyncio.run(target(*args, **kwargs))
        except Exception as e:
            print(f"[FATAL] {target.__name__} crashed: {e}")
            traceback.print_exc()
            time.sleep(10)  # brief cooldown before retrying

# =========================
# Reddit init + error handler
# =========================
def initialize_reddit():
    return asyncpraw.Reddit(
        client_id='jl-I3OHYH2_VZMC1feoJMQ',
        client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
        username='PokeLeakBot3',
        password='testbot1',
        user_agent='testbot'
    )

def handle_exception(e):
    if isinstance(e, asyncpraw.exceptions.AsyncPRAWException):
        print("Rate limited by Reddit API. Ignoring error.")

# =========================
# Workers
# =========================
async def monitor_reported_posts():
    reddit = initialize_reddit()
    subreddit = await reddit.subreddit("PokeLeaks")
    while True:
        try:
            async for post in subreddit.mod.reports():
                # If already approved previously, re-approve (idempotent)
                if getattr(post, "approved", False):
                    await post.mod.approve()
                    print(f"Post {post.id} has been approved again")
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)
    await reddit.close()

async def handle_modqueue_items():
    reddit = initialize_reddit()
    timers = {}

    while True:
        try:
            subreddit = await reddit.subreddit('PokeLeaks')
            async for item in subreddit.mod.modqueue():
                if getattr(item, "num_reports", 0) == 1 and item.id not in timers:
                    created_time = getattr(item, "created_utc", time.time())
                    timers[item.id] = time.time()
                    print(f"Starting timer for post {item.id} (created {created_time})...")

                if item.id in timers:
                    start_time = timers[item.id]
                    time_diff = time.time() - start_time
                    if time_diff >= 3600:
                        try:
                            await item.mod.approve()
                            print(f"Approved post {item.id} with one report")
                            del timers[item.id]
                        except Exception as se:
                            handle_exception(se)
                    else:
                        # NOTE: As written originally, this comparison doesn't change.
                        # Keeping logic intact; just protecting against crashes.
                        new_reports = getattr(item, "report_reasons", None)
                        if new_reports != getattr(item, "report_reasons", None):
                            print(f"New reports for post {item.id}, leaving post in mod queue")
                            del timers[item.id]
                        else:
                            time_remaining = int(start_time + 3600 - time.time())
                            print(f"Time remaining for post {item.id}: {time_remaining} seconds")
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)
    await reddit.close()

async def handle_spoiler_status():
    reddit = initialize_reddit()
    subreddit = await reddit.subreddit('PokeLeaks')
    previous_spoiler_status = {}

    while True:
        try:
            async for submission in subreddit.new():
                if submission.id not in previous_spoiler_status:
                    previous_spoiler_status[submission.id] = submission.spoiler
                    continue

                if previous_spoiler_status[submission.id] != submission.spoiler:
                    # Check if the change was made by a moderator
                    try:
                        mods = await subreddit.moderator()
                        is_moderator = submission.author in mods
                    except Exception:
                        is_moderator = False  # be safe if something weird happens

                    if not submission.spoiler:
                        if not is_moderator:
                            try:
                                print(f'Post {submission.id} unmarked as spoiler by non-mod. Re-marking.')
                                await submission.mod.spoiler()
                            except Exception as se:
                                handle_exception(se)
                        else:
                            print(f'Post {submission.id} unmarked as spoiler by a moderator. Leaving as-is.')
                    previous_spoiler_status[submission.id] = submission.spoiler
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(30)
    await reddit.close()

async def handle_user_reports_and_removal():
    reddit = initialize_reddit()
    subreddit = await reddit.subreddit("PokeLeaks")
    thresholds = {
        'No Linking to Downloadable Content in Posts or Comments': 1,
        'No ROMs, ISOs, or Game Files Sharing or Requests': 1,
        'No insults or harassment of other subreddit members in the comments': 1
    }

    while True:
        try:
            async for comment in subreddit.mod.modqueue(limit=100):
                if isinstance(comment, asyncpraw.models.Comment) and getattr(comment, "user_reports", None):
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            await comment.mod.remove()
                            print(f'Comment "{comment.body}" removed due to {count} reports for reason: {reason}')
                        except Exception as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)
    await reddit.close()

async def handle_submissions_based_on_user_reports():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            subreddit = await reddit.subreddit('PokeLeaks')
            async for post in subreddit.mod.modqueue(limit=100):
                if isinstance(post, asyncpraw.models.Submission) and getattr(post, "user_reports", None):
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            await post.mod.approve()
                            print(f'post "{post.title}" approved due to {count} reports for reason: {reason}')
                        except Exception as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)
    await reddit.close()

async def handle_posts_based_on_removal():
    reddit = initialize_reddit()
    thresholds = {
        'Users Are Responsible for the Content They Post': 2,
        'Discussion-Only for Leaks, Not Distribution': 2,
        'No Linking to Downloadable Content in Posts or Comments': 1,
        'No ROMs, ISOs, or Game Files Sharing or Requests': 2,
        'Theories, Questions, Speculations must be commented in the Theory/Speculation/Question Megathread': 2,
        'Content Must Relate to PokÃ©mon Leaks or News': 2,
        'Content must not contain any profanities, vulgarity, sexual content, slurs, be appropriate in nature': 2,
        'Post title should include sourcing and must be transparent': 2,
        'Posts with spoilers must have required spoiler flair, indicate spoiler alert in title, and be vague': 3,
        'No reposting of posts already up on the subreddit': 2,
        'No Self Advertisements or Promotion': 2,
        'No Memes, Fan Art, or Joke Posts': 2
    }

    while True:
        try:
            subreddit = await reddit.subreddit('PokeLeaks')
            async for post in subreddit.mod.modqueue(limit=100):
                if isinstance(post, asyncpraw.models.Submission) and getattr(post, "user_reports", None):
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            await post.mod.remove()
                            print(f'Submission "{post.title}" removed due to {count} reports for reason: {reason}')
                        except Exception as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)
    await reddit.close()

async def handle_comments_based_on_approval():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            subreddit = await reddit.subreddit('PokeLeaks')
            async for comment in subreddit.mod.modqueue(limit=100):
                if getattr(comment, "user_reports", None):
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            await comment.mod.approve()
                            print(f'Comment "{comment.body}" approved due to {count} reports for reason: {reason}')
                        except Exception as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)
    await reddit.close()

async def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    
    # Global dictionary to store per-subreddit data
    subreddit_data = {}
    
    async def setup_subreddit(subreddit_name):
        """Initialize data structures and monitoring for a specific subreddit"""
        print(f"\n=== Setting up bot for r/{subreddit_name} ===")
        
        subreddit = await reddit.subreddit(subreddit_name)
        
        # Create dedicated dictionaries for this subreddit
        data = {
            'subreddit': subreddit,
            'image_hashes': {},
            'orb_descriptors': {},
            'moderator_removed_hashes': set(),
            'processed_modqueue_submissions': set(),
            'approved_by_moderator': set(),
            'ai_features': {},
            'repost_history': {},  # Track repost attempts: hash -> list of (author, title, date, utc, permalink)
            'current_time': int(time.time())
        }
        
        subreddit_data[subreddit_name] = data
        
        # --- AI model (shared across all subreddits) ---
        device = "cpu"
        efficientnet_model = models.efficientnet_b0(pretrained=True)
        efficientnet_model.eval()
        efficientnet_model.to(device)
        efficientnet_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Reuse ORB detector
        orb_detector = cv2.ORB_create()
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # --- Helper functions for this subreddit ---
        def get_ai_features(img):
            try:
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_tensor = efficientnet_transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = efficientnet_model(img_tensor)
                    feat = feat / feat.norm(dim=1, keepdim=True)
                return feat
            except Exception as e:
                print(f"[r/{subreddit_name}] AI feature extraction error:", e)
                return None

        def is_problematic_image(img, white_threshold=0.7, text_threshold=0.05):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            white_ratio = np.mean(gray > 240)
            
            if white_ratio > white_threshold:
                return True
            
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.mean(edges > 0)
            return edge_ratio > text_threshold

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
            kp, des = orb_detector.detectAndCompute(processed_img, None)
            return des

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

        async def post_comment(submission, original_post_author, original_post_title, original_post_date, original_post_utc, original_status, original_post_permalink, matched_hash):
            max_retries = 3
            retries = 0
            age_text = format_age(original_post_utc)
            
            # Build repost history section (excluding current submission)
            repost_section = ""
            if matched_hash in data['repost_history'] and len(data['repost_history'][matched_hash]) > 0:
                # Filter out current submission from history
                previous_reposts = [r for r in data['repost_history'][matched_hash] if r[4] != submission.permalink]
                
                if len(previous_reposts) > 0:
                    repost_section = "\n\n---\n\n**Previous repost attempts:**\n\n"
                    repost_section += "| Author | Title | Date | Age | Status |\n"
                    repost_section += "|:------:|:-----:|:----:|:---:|:------:|\n"
                    
                    for repost in previous_reposts:
                        repost_author, repost_title, repost_date, repost_utc, repost_permalink, repost_status = repost
                        repost_age = format_age(repost_utc)
                        repost_section += f"| {repost_author} | [{repost_title}]({repost_permalink}) | {repost_date} | {repost_age} | {repost_status} |\n"
            
            while retries < max_retries:
                try:
                    comment_text = (
                        "> **Duplicate detected**\n\n"
                        "| Original Author | Title | Date | Age | Status |\n"
                        "|:---------------:|:-----:|:----:|:---:|:------:|\n"
                        f"| {original_post_author} | [{original_post_title}]({original_post_permalink}) | {original_post_date} | {age_text} | {original_status} |"
                        f"{repost_section}"
                    )
                    comment = await submission.reply(comment_text)
                    await comment.mod.distinguish(sticky=True)
                    await comment.mod.lock()
                    print(f"[r/{subreddit_name}] Duplicate removed and comment posted (locked): ", submission.url)
                    return True
                except Exception as e:
                    handle_exception(e)
                    retries += 1
                    await asyncio.sleep(1)
            return False

        def add_to_repost_history(matched_hash, submission):
            """Add current submission to repost history"""
            if matched_hash not in data['repost_history']:
                data['repost_history'][matched_hash] = []
            
            repost_data = (
                submission.author.name,
                submission.title,
                datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                submission.created_utc,
                submission.permalink,
                "Removed (Repost)"  # Status for reposts
            )
            data['repost_history'][matched_hash].append(repost_data)

        def load_and_process_image(url):
            """Load image from URL and compute hash, descriptors, AI features"""
            image_data = requests.get(url, timeout=10).content
            img = np.asarray(bytearray(image_data), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hash_value = str(imagehash.phash(Image.fromarray(gray)))
            print(f"[r/{subreddit_name}] Generated hash: {hash_value}")
            
            descriptors = get_orb_descriptors_conditional(img)
            features = get_ai_features(img)
                
            return img, hash_value, descriptors, features

        async def get_cached_ai_features(submission_id):
            """Get AI features from cache or compute them"""
            if submission_id in data['ai_features']:
                return data['ai_features'][submission_id]
            
            old_submission = await reddit.submission(id=submission_id)
            old_image_data = requests.get(old_submission.url, timeout=10).content
            old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
            old_features = get_ai_features(old_img)
            data['ai_features'][submission_id] = old_features
            return old_features

        def calculate_ai_similarity(features1, features2):
            """Calculate AI similarity score between two feature vectors"""
            if features1 is not None and features2 is not None:
                return (features1 @ features2.T).item()
            return 0

        async def check_hash_duplicate(submission, hash_value, new_features):
            """Check if submission is a hash-based duplicate"""
            matched_hash = None
            for stored_hash in data['image_hashes'].keys():
                if hash_value == stored_hash or (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= 4:
                    matched_hash = stored_hash
                    break
            
            if matched_hash is None:
                return False, None, None, None, None, None, None, None
            
            original_id, original_time = data['image_hashes'][matched_hash]
            
            if submission.id == original_id or submission.created_utc <= original_time:
                return False, None, None, None, None, None, None, None
            
            original_submission = await reddit.submission(id=original_id)
            original_features = await get_cached_ai_features(original_submission.id)
            
            ai_score = calculate_ai_similarity(new_features, original_features)
            
            print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f}")
            
            if ai_score > 0.50:
                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                original_status = "Removed by Moderator" if matched_hash in data['moderator_removed_hashes'] else "Active"
                return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink, matched_hash
            
            return False, None, None, None, None, None, None, None

        async def check_orb_duplicate(submission, descriptors, new_features):
            """Check if submission is an ORB-based duplicate"""
            for old_id, old_desc in data['orb_descriptors'].items():
                sim = orb_similarity(descriptors, old_desc)
                
                if sim > 0.50:
                    old_features = await get_cached_ai_features(old_id)
                    
                    ai_score = calculate_ai_similarity(new_features, old_features)
                    
                    if ai_score > 0.75:
                        print(f"[r/{subreddit_name}] ORB duplicate found! AI similarity: {ai_score:.2f}")
                        
                        original_submission = await reddit.submission(id=old_id)
                        original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                        old_hash = next((h for h, v in data['image_hashes'].items() if v[0] == old_id), None)
                        original_status = "Removed by Moderator" if old_hash and old_hash in data['moderator_removed_hashes'] else "Active"
                        
                        return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink, old_hash
            
            return False, None, None, None, None, None, None, None

        async def handle_duplicate(submission, is_hash_dup, detection_method, author, title, date, utc, status, permalink, matched_hash):
            """Remove duplicate and post comment if not approved"""
            if not submission.approved:
                # Post comment first (before adding to history so current post isn't included)
                await post_comment(submission, author, title, date, utc, status, permalink, matched_hash)
                # Then add to history for future reposts
                add_to_repost_history(matched_hash, submission)
                await submission.mod.remove()
                print(f"[r/{subreddit_name}] Duplicate removed by {detection_method}: {submission.url}")
            return True

        async def handle_moderator_removed_repost(submission, hash_value):
            """Handle reposts of moderator-removed images"""
            if hash_value in data['moderator_removed_hashes'] and not submission.approved:
                original_submission = await reddit.submission(id=data['image_hashes'][hash_value][0])
                # Post comment first (before adding to history)
                await post_comment(
                    submission,
                    original_submission.author.name,
                    original_submission.title,
                    datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    original_submission.created_utc,
                    "Removed by Moderator",
                    original_submission.permalink,
                    hash_value
                )
                # Then add to history
                add_to_repost_history(hash_value, submission)
                await submission.mod.remove()
                print(f"[r/{subreddit_name}] Repost of a moderator-removed image removed: ", submission.url)
                return True
            return False

        async def process_submission_for_duplicates(submission, context="stream"):
            """Main duplicate detection logic - works for both mod queue and stream"""
            try:
                img, hash_value, descriptors, new_features = load_and_process_image(submission.url)
                data['ai_features'][submission.id] = new_features
                
                if await handle_moderator_removed_repost(submission, hash_value):
                    return True
                
                is_duplicate, author, title, date, utc, status, permalink, matched_hash = await check_hash_duplicate(
                    submission, hash_value, new_features
                )
                if is_duplicate:
                    return await handle_duplicate(submission, True, "hash + AI", author, title, date, utc, status, permalink, matched_hash)
                
                is_duplicate, author, title, date, utc, status, permalink, matched_hash = await check_orb_duplicate(
                    submission, descriptors, new_features
                )
                if is_duplicate:
                    return await handle_duplicate(submission, False, "ORB + AI", author, title, date, utc, status, permalink, matched_hash)
                
                if hash_value not in data['image_hashes']:
                    data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                    data['orb_descriptors'][submission.id] = descriptors
                    data['ai_features'][submission.id] = new_features
                    print(f"[r/{subreddit_name}] Stored new original: {submission.url}")
                
                if context == "modqueue" and not submission.approved:
                    await submission.mod.approve()
                    print(f"[r/{subreddit_name}] Original submission approved: ", submission.url)
                
                return False
                
            except Exception as e:
                handle_exception(e)
                return False

        async def check_removed_original_posts():
            """Monitor for immediate removal detection using dual approach"""
            processed_log_items = set()
            last_checked = {}
            
            async def monitor_mod_log():
                while True:
                    try:
                        async for log_entry in subreddit.mod.stream.log(action='removelink', skip_existing=True):
                            if log_entry.id in processed_log_items:
                                continue
                            
                            processed_log_items.add(log_entry.id)
                            removed_submission_id = log_entry.target_fullname.replace('t3_', '')
                            
                            hash_to_process = None
                            for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                                if submission_id == removed_submission_id:
                                    hash_to_process = hash_value
                                    break
                            
                            if hash_to_process and hash_to_process not in data['moderator_removed_hashes']:
                                data['moderator_removed_hashes'].add(hash_to_process)
                                print(f"[r/{subreddit_name}] [MOD REMOVE] Submission {removed_submission_id} removed by moderator. Hash kept for future duplicate detection.")
                            
                            if len(processed_log_items) > 1000:
                                processed_log_items.clear()
                        
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in mod log monitor: {e}")
                        await asyncio.sleep(5)
            
            asyncio.create_task(monitor_mod_log())
            
            while True:
                try:
                    current_check_time = time.time()
                    checked_this_cycle = 0
                    
                    recent_submissions = []
                    medium_submissions = []
                    old_submissions = []
                    
                    for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                        if hash_value in data['moderator_removed_hashes']:
                            continue
                        
                        age = current_check_time - creation_time
                        last_check = last_checked.get(submission_id, 0)
                        
                        if age < 3600:
                            check_interval = 30
                            if current_check_time - last_check >= check_interval:
                                recent_submissions.append((hash_value, submission_id))
                        elif age < 86400:
                            check_interval = 300
                            if current_check_time - last_check >= check_interval:
                                medium_submissions.append((hash_value, submission_id))
                        else:
                            check_interval = 1800
                            if current_check_time - last_check >= check_interval:
                                old_submissions.append((hash_value, submission_id))
                    
                    all_to_check = recent_submissions + medium_submissions[:20] + old_submissions[:10]
                    
                    for hash_value, submission_id in all_to_check:
                        try:
                            original_submission = await reddit.submission(id=submission_id)
                            original_author = original_submission.author
                            
                            if original_author is None:
                                if hash_value in data['image_hashes']:
                                    del data['image_hashes'][hash_value]
                                if submission_id in data['orb_descriptors']:
                                    del data['orb_descriptors'][submission_id]
                                if submission_id in data['ai_features']:
                                    del data['ai_features'][submission_id]
                                if submission_id in last_checked:
                                    del last_checked[submission_id]
                                if hash_value in data['repost_history']:
                                    del data['repost_history'][hash_value]
                                print(f"[r/{subreddit_name}] [USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                            else:
                                last_checked[submission_id] = current_check_time
                            
                            checked_this_cycle += 1
                            
                            if checked_this_cycle >= 10:
                                await asyncio.sleep(60)
                                checked_this_cycle = 0
                            
                        except Exception as e:
                            print(f"[r/{subreddit_name}] Error checking submission {submission_id}: {e}")
                            last_checked[submission_id] = current_check_time
                    
                except Exception as e:
                    handle_exception(e)
                
                await asyncio.sleep(60)
        
        asyncio.create_task(check_removed_original_posts())

        # --- Initial scan ---
        print(f"[r/{subreddit_name}] Starting initial scan...")
        try:
            async for submission in subreddit.new(limit=10000):
                if isinstance(submission, asyncpraw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    print(f"[r/{subreddit_name}] Indexing submission (initial scan): ", submission.url)
                    try:
                        img, hash_value, descriptors, features = load_and_process_image(submission.url)
                        if hash_value not in data['image_hashes']:
                            data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                            data['orb_descriptors'][submission.id] = descriptors
                            data['ai_features'][submission.id] = features
                    except Exception as e:
                        handle_exception(e)
        except Exception as e:
            handle_exception(e)
        
        print(f"[r/{subreddit_name}] Initial scan complete. Indexed {len(data['image_hashes'])} images.")

        # --- Mod Queue worker ---
        async def modqueue_worker():
            while True:
                try:
                    modqueue_submissions = []
                    async for sub in subreddit.mod.modqueue(only='submission', limit=None):
                        modqueue_submissions.append(sub)
                    modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                    for submission in modqueue_submissions:
                        if not isinstance(submission, asyncpraw.models.Submission):
                            continue
                        
                        print(f"[r/{subreddit_name}] Scanning Mod Queue: ", submission.url)
                        
                        if submission.num_reports > 0:
                            print(f"[r/{subreddit_name}] Skipping reported image: ", submission.url)
                            data['image_hashes'] = {k: v for k, v in data['image_hashes'].items() if v[0] != submission.id}
                            data['orb_descriptors'].pop(submission.id, None)
                            data['ai_features'].pop(submission.id, None)
                            continue
                        
                        if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                            is_duplicate = await process_submission_for_duplicates(submission, context="modqueue")
                            data['processed_modqueue_submissions'].add(submission.id)

                except Exception as e:
                    handle_exception(e)
                await asyncio.sleep(15)

        asyncio.create_task(modqueue_worker())

        # --- Stream new submissions ---
        async def stream_worker():
            while True:
                try:
                    async for submission in subreddit.stream.submissions(skip_existing=True):
                        if submission.created_utc > data['current_time'] and isinstance(submission, asyncpraw.models.Submission):
                            if submission.id in data['processed_modqueue_submissions']:
                                continue

                            print(f"[r/{subreddit_name}] Scanning new image/post: ", submission.url)
                            
                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                await process_submission_for_duplicates(submission, context="stream")

                    data['current_time'] = int(time.time())
                except Exception as e:
                    handle_exception(e)
                await asyncio.sleep(20)
        
        asyncio.create_task(stream_worker())
        
        print(f"[r/{subreddit_name}] Bot fully operational!\n")

    # --- Accept invites and setup subreddits ---
    async def check_for_invites():
        """Check for moderator invites and automatically accept them"""
        while True:
            try:
                # Check unread messages for mod invites
                async for message in reddit.inbox.unread(limit=None):
                    if "invitation to moderate" in message.subject.lower():
                        subreddit_name = message.subreddit.display_name
                        print(f"\n*** Found mod invite for r/{subreddit_name} ***")
                        try:
                            await message.subreddit.mod.accept_invite()
                            print(f"✅ Accepted mod invite for r/{subreddit_name}")
                            asyncio.create_task(setup_subreddit(subreddit_name))
                        except Exception as e:
                            print(f"Error accepting invite for r/{subreddit_name}: {e}")
                        await message.mark_read()
            
                # Also check for already accepted subreddits
                async for subreddit in reddit.user.moderator_subreddits(limit=None):
                    subreddit_name = subreddit.display_name
                    if subreddit_name not in subreddit_data:
                        print(f"\n*** Already moderating r/{subreddit_name}, setting up bot ***")
                        asyncio.create_task(setup_subreddit(subreddit_name))
            
            except Exception as e:
                print(f"Error checking for invites: {e}")
        
            await asyncio.sleep(60)
    
    # Start the invite checker
    asyncio.create_task(check_for_invites())
    
    # Keep main coroutine alive
    print("=== Multi-subreddit duplicate bot started ===")
    print("Monitoring for mod invites...")
    while True:
        await asyncio.sleep(10)  # Keep alive
        
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

    add_thread('modqueue_thread', handle_modqueue_items)
    add_thread('reported_posts_thread', monitor_reported_posts)
    add_thread('spoiler_status_thread', handle_spoiler_status)
    add_thread('user_reports_removal_thread', handle_user_reports_and_removal)
    add_thread('submissions_based_on_user_reports_thread', handle_submissions_based_on_user_reports)
    add_thread('posts_based_on_removal_thread', handle_posts_based_on_removal)
    add_thread('comments_based_on_approval_thread', handle_comments_based_on_approval)
    add_thread('run_pokemon_duplicate_bot_thread', run_pokemon_duplicate_bot)

    # Keep the main thread alive indefinitely so daemon threads keep running.
    while True:
        time.sleep(30)
