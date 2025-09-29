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
        client_id='',
        client_secret='',
        username='',
        password='',
        user_agent=''
    )

def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and getattr(e, "response", None) and e.response.status_code == 429:
        print("Rate limited by Reddit API. Ignoring error.")
    else:
        print(f"Exception encountered: {str(e)}")

# =========================
# Workers
# =========================
def monitor_reported_posts():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit("PokeLeaks")
    while True:
        try:
            for post in subreddit.mod.reports():
                # If already approved previously, re-approve (idempotent)
                if getattr(post, "approved", False):
                    post.mod.approve()
                    print(f"Post {post.id} has been approved again")
        except Exception as e:
            handle_exception(e)
            time.sleep(5)

def handle_modqueue_items():
    reddit = initialize_reddit()
    timers = {}

    while True:
        try:
            for item in reddit.subreddit('PokeLeaks').mod.modqueue():
                if getattr(item, "num_reports", 0) == 1 and item.id not in timers:
                    created_time = getattr(item, "created_utc", time.time())
                    timers[item.id] = time.time()
                    print(f"Starting timer for post {item.id} (created {created_time})...")

                if item.id in timers:
                    start_time = timers[item.id]
                    time_diff = time.time() - start_time
                    if time_diff >= 3600:
                        try:
                            item.mod.approve()
                            print(f"Approved post {item.id} with one report")
                            del timers[item.id]
                        except prawcore.exceptions.ServerError as se:
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
            time.sleep(5)

def handle_spoiler_status():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    previous_spoiler_status = {}

    while True:
        try:
            for submission in subreddit.new():
                if submission.id not in previous_spoiler_status:
                    previous_spoiler_status[submission.id] = submission.spoiler
                    continue

                if previous_spoiler_status[submission.id] != submission.spoiler:
                    # Check if the change was made by a moderator
                    try:
                        is_moderator = submission.author in subreddit.moderator()
                    except Exception:
                        is_moderator = False  # be safe if something weird happens

                    if not submission.spoiler:
                        if not is_moderator:
                            try:
                                print(f'Post {submission.id} unmarked as spoiler by non-mod. Re-marking.')
                                submission.mod.spoiler()
                            except prawcore.exceptions.ServerError as se:
                                handle_exception(se)
                        else:
                            print(f'Post {submission.id} unmarked as spoiler by a moderator. Leaving as-is.')
                    previous_spoiler_status[submission.id] = submission.spoiler
        except Exception as e:
            handle_exception(e)
            time.sleep(5)

def handle_user_reports_and_removal():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit("PokeLeaks")
    thresholds = {
        'No Linking to Downloadable Content in Posts or Comments': 1,
        'No ROMs, ISOs, or Game Files Sharing or Requests': 1,
        'No insults or harassment of other subreddit members in the comments': 1
    }

    while True:
        try:
            for comment in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
                if isinstance(comment, praw.models.Comment) and getattr(comment, "user_reports", None):
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            comment.mod.remove()
                            print(f'Comment "{comment.body}" removed due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            time.sleep(5)

def handle_submissions_based_on_user_reports():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            for post in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
                if isinstance(post, praw.models.Submission) and getattr(post, "user_reports", None):
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            post.mod.approve()
                            print(f'post "{post.title}" approved due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            time.sleep(5)

def handle_posts_based_on_removal():
    reddit = initialize_reddit()
    thresholds = {
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
        'No Memes, Fan Art, or Joke Posts': 2
    }

    while True:
        try:
            for post in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
                if isinstance(post, praw.models.Submission) and getattr(post, "user_reports", None):
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            post.mod.remove()
                            print(f'Submission "{post.title}" removed due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            time.sleep(5)

def handle_comments_based_on_approval():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            for comment in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
                if getattr(comment, "user_reports", None):
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            comment.mod.approve()
                            print(f'Comment "{comment.body}" approved due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            time.sleep(5)

def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    image_hashes = {}
    orb_descriptors = {}  # Store ORB descriptors
    moderator_removed_hashes = set()  # Track images removed by moderators
    processed_modqueue_submissions = set()
    approved_by_moderator = set()  # Track submissions approved by moderators
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
    def ai_similarity(img1, img2):
        try:
            img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            img1_tensor = resnet_transform(img1_pil).unsqueeze(0).to(device)
            img2_tensor = resnet_transform(img2_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feat1 = resnet_model(img1_tensor)
                feat2 = resnet_model(img2_tensor)
                feat1 = feat1 / feat1.norm(dim=1, keepdim=True)
                feat2 = feat2 / feat2.norm(dim=1, keepdim=True)
                similarity = (feat1 @ feat2.T).item()
            return similarity
        except Exception as e:
            print("AI similarity error:", e)
            return 0

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

    def post_comment(submission, original_post_author, original_post_title, original_post_link, original_post_date):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                comment = submission.reply(
                    ">| Title | Date and Time |\n"
                    ">|:---:|:---:|\n"
                    ">| [{}]({}) | {} |\n\n"
                    ">[Link to Original Post]({})"
                    .format(original_post_title, original_post_link, original_post_date, original_post_link)
                )
                comment.mod.distinguish(sticky=True)
                print("Duplicate removed and comment posted: ", submission.url)
                return True
            except Exception as e:
                handle_exception(e)
                retries += 1
                time.sleep(1)
        return False

    def check_removed_original_posts():
        while True:
            try:
                for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                    original_submission = reddit.submission(id=submission_id)
                    original_author = original_submission.author
                    banned_by_moderator = original_submission.banned_by
                    if banned_by_moderator is not None:
                        moderator_removed_hashes.add(hash_value)
                        del image_hashes[hash_value]
                        if submission_id in orb_descriptors:
                            del orb_descriptors[submission_id]
                    elif original_author is None:
                        del image_hashes[hash_value]
                        if submission_id in orb_descriptors:
                            del orb_descriptors[submission_id]
            except Exception as e:
                handle_exception(e)
            time.sleep(10)

    # --- Start original post checker thread ---
    threading.Thread(target=check_removed_original_posts, daemon=True).start()

    # --- Initial scan of subreddit posts ---
    try:
        for submission in subreddit.new(limit=500):
            if isinstance(submission, praw.models.Submission):
                print("Indexing submission (initial scan): ", submission.url)
                if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    try:
                        image_data = requests.get(submission.url).content
                        img = np.asarray(bytearray(image_data), dtype=np.uint8)
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
                        descriptors = get_orb_descriptors_conditional(img)
                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                        orb_descriptors[submission.id] = descriptors
                    except Exception as e:
                        handle_exception(e)
    except Exception as e:
        handle_exception(e)

    # --- Mod Queue scanning thread ---
    def modqueue_worker():
        nonlocal image_hashes, orb_descriptors, moderator_removed_hashes, processed_modqueue_submissions
        while True:
            try:
                modqueue_submissions = subreddit.mod.modqueue(only='submission', limit=None)
                modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                for submission in modqueue_submissions:
                    if isinstance(submission, praw.models.Submission):
                        print("Scanning Mod Queue: ", submission.url)
                        if submission.num_reports > 0:
                            print("Skipping reported image: ", submission.url)
                            image_hashes = {k: v for k, v in image_hashes.items() if v[0] != submission.id}
                            if submission.id in orb_descriptors:
                                del orb_descriptors[submission.id]
                            continue
                        if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                            try:
                                image_data = requests.get(submission.url).content
                                img = np.asarray(bytearray(image_data), dtype=np.uint8)
                                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                                hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
                                descriptors = get_orb_descriptors_conditional(img)

                                # Moderator removed check
                                if hash_value in moderator_removed_hashes:
                                    if not submission.approved:
                                        submission.mod.remove()
                                        print("Repost of a moderator-removed image removed: ", submission.url)
                                    continue

                                is_duplicate_orb = False  # Prevent double removal

                                # Hash duplicates
                                if hash_value in image_hashes:
                                    original_submission_id, original_time = image_hashes[hash_value]
                                    original_submission = reddit.submission(id=original_submission_id)
                                    original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                                    original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                                    original_post_title = original_submission.title
                                    original_post_author = original_submission.author
                                    if submission.id != original_submission_id and submission.created_utc > original_time:
                                        try:
                                            original_image_data = requests.get(original_submission.url).content
                                            original_img = np.asarray(bytearray(original_image_data), dtype=np.uint8)
                                            original_img = cv2.imdecode(original_img, cv2.IMREAD_COLOR)
                                            ai_score = ai_similarity(img, original_img)
                                            print(f"Hash match detected in Mod Queue. AI similarity: {ai_score:.2f}")
                                            if ai_score > 0.70:
                                                if not submission.approved:
                                                    submission.mod.remove()
                                                    print("Duplicate removed by hash + AI: ", submission.url)
                                                if post_comment(submission, original_post_author.name, original_post_title, original_post_link, original_post_date):
                                                    processed_modqueue_submissions.add(submission.id)
                                                is_duplicate_orb = True  # skip ORB + AI
                                        except Exception as e:
                                            handle_exception(e)
                                else:
                                    image_hashes[hash_value] = (submission.id, submission.created_utc)
                                    orb_descriptors[submission.id] = descriptors

                                # ORB + AI similarity for non-hash duplicates
                                if not is_duplicate_orb:
                                    for old_id, old_desc in orb_descriptors.items():
                                        sim = orb_similarity(descriptors, old_desc)
                                        if sim > 0.30:
                                            original_submission = reddit.submission(id=old_id)
                                            original_post_author = original_submission.author
                                            original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                                            original_post_title = original_submission.title
                                            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                                            try:
                                                original_image_data = requests.get(original_submission.url).content
                                                original_img = np.asarray(bytearray(original_image_data), dtype=np.uint8)
                                                original_img = cv2.imdecode(original_img, cv2.IMREAD_COLOR)
                                                ai_score = ai_similarity(img, original_img)
                                                print(f"ORB similarity in Mod Queue: {sim:.2f}, AI similarity: {ai_score:.2f}")
                                                if ai_score > 0.70:
                                                    if not submission.approved:
                                                        submission.mod.remove()
                                                        print("Duplicate removed by ORB + AI in Mod Queue: ", submission.url)
                                                    post_comment(submission, original_post_author.name, original_post_title, original_post_link, original_post_date)
                                                is_duplicate_orb = True
                                            except Exception as e:
                                                handle_exception(e)
                                        if is_duplicate_orb:
                                            break

                                # Approve if not duplicate
                                if not is_duplicate_orb:
                                    if not submission.approved:
                                        submission.mod.approve()
                                        print("Original submission approved: ", submission.url)
                                    image_hashes[hash_value] = (submission.id, submission.created_utc)
                                    orb_descriptors[submission.id] = descriptors

                            except Exception as e:
                                handle_exception(e)
            except Exception as e:
                handle_exception(e)
            time.sleep(2)

    threading.Thread(target=modqueue_worker, daemon=True).start()

    # --- Main thread: stream new submissions ---
    while True:
        try:
            for submission in subreddit.stream.submissions(skip_existing=True):
                if submission.created_utc > current_time and isinstance(submission, praw.models.Submission):
                    print("Scanning new image/post: ", submission.url)
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        try:
                            image_data = requests.get(submission.url).content
                            img = np.asarray(bytearray(image_data), dtype=np.uint8)
                            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                            hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
                            descriptors = get_orb_descriptors_conditional(img)

                            # Moderator removed check
                            if hash_value in moderator_removed_hashes:
                                if not submission.approved:
                                    submission.mod.remove()
                                    print("Repost of a moderator-removed image removed: ", submission.url)
                                continue

                            is_duplicate_orb = False  # Flag to prevent double removal

                            # Hash duplicates
                            if hash_value in image_hashes:
                                original_submission_id, original_time = image_hashes[hash_value]
                                original_submission = reddit.submission(id=original_submission_id)
                                original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                                original_post_title = original_submission.title
                                original_post_author = original_submission.author
                                if submission.id != original_submission_id and submission.created_utc > original_time:
                                    try:
                                        original_image_data = requests.get(original_submission.url).content
                                        original_img = np.asarray(bytearray(original_image_data), dtype=np.uint8)
                                        original_img = cv2.imdecode(original_img, cv2.IMREAD_COLOR)
                                        ai_score = ai_similarity(img, original_img)
                                        print(f"Hash match detected. AI similarity: {ai_score:.2f}")
                                        if ai_score > 0.70:
                                            if not submission.approved:
                                                submission.mod.remove()
                                                print("Duplicate removed by hash + AI: ", submission.url)
                                            post_comment(submission, original_post_author.name, original_post_title, original_post_link, original_post_date)
                                            is_duplicate_orb = True  # skip ORB + AI
                                    except Exception as e:
                                        handle_exception(e)
                            else:
                                image_hashes[hash_value] = (submission.id, submission.created_utc)
                                orb_descriptors[submission.id] = descriptors

                            # ORB + AI similarity for non-hash duplicates
                            if not is_duplicate_orb:
                                for old_id, old_desc in orb_descriptors.items():
                                    sim = orb_similarity(descriptors, old_desc)
                                    if sim > 0.30:
                                        original_submission = reddit.submission(id=old_id)
                                        original_post_author = original_submission.author
                                        original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                                        original_post_title = original_submission.title
                                        original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                                        try:
                                            original_image_data = requests.get(original_submission.url).content
                                            original_img = np.asarray(bytearray(original_image_data), dtype=np.uint8)
                                            original_img = cv2.imdecode(original_img, cv2.IMREAD_COLOR)
                                            ai_score = ai_similarity(img, original_img)
                                            print(f"ORB similarity: {sim:.2f}, AI similarity: {ai_score:.2f}")
                                            if ai_score > 0.70:
                                                if not submission.approved:
                                                    submission.mod.remove()
                                                    print("Duplicate removed by ORB + AI: ", submission.url)
                                                post_comment(submission, original_post_author.name, original_post_title, original_post_link, original_post_date)
                                            is_duplicate_orb = True
                                        except Exception as e:
                                            handle_exception(e)
                                    if is_duplicate_orb:
                                        break

                            # Approve if not duplicate
                            if not is_duplicate_orb:
                                if not submission.approved:
                                    submission.mod.approve()
                                    print("Original submission approved: ", submission.url)
                                image_hashes[hash_value] = (submission.id, submission.created_utc)
                                orb_descriptors[submission.id] = descriptors

                        except Exception as e:
                            handle_exception(e)
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

