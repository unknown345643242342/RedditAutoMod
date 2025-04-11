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
from praw.models import Submission, Comment

# === Constants === #
MAX_REQUESTS_PER_MINUTE = 100
SUBREDDIT_NAME = "PokeLeaks"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif')

THRESHOLDS_COMMENTS_APPROVAL = {'This is misinformation': 1, 'This is spam': 1}
THRESHOLDS_COMMENTS_REMOVAL = {
    'Comments complaining about Riddler Khu will be removed. Ignore the post if you are going to complain': 1,
    'No insults or harassment of other subreddit members in the comments': 1,
    'No asking about or sharing of XCI or NSP files': 1
}
THRESHOLDS_SUBMISSIONS_APPROVAL = {'This is misinformation': 1, 'This is spam': 1}
THRESHOLDS_SUBMISSIONS_REMOVAL = {
    'Posts must be about leaks, riddles, news, and rumours about Pokémon content': 2,
    'Post must not contain any profanities, vulgarity, sexual content, slurs, be appropriate in nature': 2,
    'No reposting of posts already up on the subreddit': 1,
    'No memes allowed': 2,
    'No joke submissions or shitposts': 2,
    'No Self Advertisements/Promotion': 2,
    'No Fan art': 2,
    'Theories, Questions, Speculations must be commented in the Theory/Speculation/Question Megathread': 2,
    'Posts with spoilers must have required spoiler flair, indicate spoiler alert in title, and be vague': 3,
    'Retarded': 1
}

# === Globals === #
api_requests_count = 0
minute_start_time = time.time()
reddit = None


# === Utility Functions === #
def make_api_request():
    global api_requests_count, minute_start_time
    current_time = time.time()
    if current_time - minute_start_time >= 60:
        minute_start_time = current_time
        api_requests_count = 0
    if api_requests_count >= MAX_REQUESTS_PER_MINUTE:
        time.sleep(65 - (current_time - minute_start_time))
    api_requests_count += 1


def initialize_reddit():
    return praw.Reddit(
        client_id='jl-I3OHYH2_VZMC1feoJMQ',
        client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
        username='PokeLeakBot3',
        password='testbot1',
        user_agent='testbot'
    )


def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and getattr(e.response, 'status_code', None) == 429:
        print("Rate limited by Reddit API.")
    else:
        print(f"Exception: {e}")


def exceeds_threshold(user_reports, thresholds):
    if not user_reports:
        return False
    reason, count = user_reports[0]
    return reason in thresholds and count >= thresholds[reason]


def get_image_hash_from_url(url):
    try:
        image_data = requests.get(url).content
        img = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return str(imagehash.phash(Image.fromarray(gray)))
    except Exception as e:
        handle_exception(e)
        return None


def post_comment(submission, title, author, link, date):
    for _ in range(3):
        try:
            comment = submission.reply(
                f">| Title | Date and Time |
"
                f">|:---:|:---:|
"
                f">| [{title}]({link}) | {date} |

"
                f">[Link to Original Post]({link})"
            )
            comment.mod.distinguish(sticky=True)
            return True
        except Exception as e:
            handle_exception(e)
            time.sleep(1)
    return False


# === Moderation Handlers === #
def monitor_reported_posts():
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    while True:
        make_api_request()
        try:
            for post in subreddit.mod.reports():
                if post.approved:
                    post.mod.approve()
        except Exception as e:
            handle_exception(e)


def handle_modqueue_items():
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    timers = {}
    while True:
        make_api_request()
        try:
            for item in subreddit.mod.modqueue():
                if item.num_reports == 1 and item.id not in timers:
                    timers[item.id] = time.time()
                elif item.id in timers:
                    if time.time() - timers[item.id] >= 3600:
                        try:
                            item.mod.approve()
                            del timers[item.id]
                        except Exception as e:
                            handle_exception(e)
        except Exception as e:
            handle_exception(e)


def handle_spoiler_status():
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    prev_spoiler = {}
    while True:
        make_api_request()
        try:
            for sub in subreddit.new():
                if sub.id not in prev_spoiler:
                    prev_spoiler[sub.id] = sub.spoiler
                    continue
                if prev_spoiler[sub.id] != sub.spoiler and not sub.spoiler:
                    if sub.author not in subreddit.moderator():
                        sub.mod.spoiler()
                prev_spoiler[sub.id] = sub.spoiler
        except Exception as e:
            handle_exception(e)


def process_modqueue_items(thresholds, action='remove', kind=Comment):
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    while True:
        make_api_request()
        try:
            for item in subreddit.mod.modqueue(limit=100):
                if isinstance(item, kind) and exceeds_threshold(item.user_reports, thresholds):
                    try:
                        getattr(item.mod, action)()
                    except Exception as e:
                        handle_exception(e)
        except Exception as e:
            handle_exception(e)


def run_duplicate_bot():
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    image_hashes = {}
    removed_hashes = set()

    def remove_reposts_loop():
        while True:
            make_api_request()
            for hash_val, (sub_id, _) in list(image_hashes.items()):
                try:
                    original = reddit.submission(id=sub_id)
                    if original.banned_by or original.author is None:
                        removed_hashes.add(hash_val)
                        del image_hashes[hash_val]
                except Exception as e:
                    handle_exception(e)

    threading.Thread(target=remove_reposts_loop, daemon=True).start()

    while True:
        make_api_request()
        try:
            for sub in subreddit.new(limit=100):
                if isinstance(sub, Submission) and sub.url.endswith(IMAGE_EXTENSIONS):
                    hash_val = get_image_hash_from_url(sub.url)
                    if not hash_val:
                        continue
                    if hash_val in removed_hashes and not sub.approved:
                        sub.mod.remove()
                        continue
                    if hash_val in image_hashes:
                        orig_id, orig_time = image_hashes[hash_val]
                        if sub.created_utc > orig_time and not sub.approved:
                            orig = reddit.submission(id=orig_id)
                            date = datetime.utcfromtimestamp(orig.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                            post_comment(sub, orig.title, orig.author.name, f"https://www.reddit.com{orig.permalink}", date)
                            sub.mod.remove()
                    else:
                        image_hashes[hash_val] = (sub.id, sub.created_utc)
        except Exception as e:
            handle_exception(e)


def monitor_threads(thread_map):
    while True:
        for name, (thread, target) in thread_map.items():
            if not thread.is_alive():
                new_thread = threading.Thread(target=target)
                thread_map[name] = (new_thread, target)
                new_thread.start()
        time.sleep(10)


# === Main Execution === #
if __name__ == "__main__":
    reddit = initialize_reddit()
    threads = {
        'reports': (threading.Thread(target=monitor_reported_posts), monitor_reported_posts),
        'modqueue': (threading.Thread(target=handle_modqueue_items), handle_modqueue_items),
        'spoilers': (threading.Thread(target=handle_spoiler_status), handle_spoiler_status),
        'comment_removal': (threading.Thread(target=lambda: process_modqueue_items(THRESHOLDS_COMMENTS_REMOVAL, 'remove', Comment)), lambda: process_modqueue_items(THRESHOLDS_COMMENTS_REMOVAL, 'remove', Comment)),
        'comment_approval': (threading.Thread(target=lambda: process_modqueue_items(THRESHOLDS_COMMENTS_APPROVAL, 'approve', Comment)), lambda: process_modqueue_items(THRESHOLDS_COMMENTS_APPROVAL, 'approve', Comment)),
        'submission_removal': (threading.Thread(target=lambda: process_modqueue_items(THRESHOLDS_SUBMISSIONS_REMOVAL, 'remove', Submission)), lambda: process_modqueue_items(THRESHOLDS_SUBMISSIONS_REMOVAL, 'remove', Submission)),
        'submission_approval': (threading.Thread(target=lambda: process_modqueue_items(THRESHOLDS_SUBMISSIONS_APPROVAL, 'approve', Submission)), lambda: process_modqueue_items(THRESHOLDS_SUBMISSIONS_APPROVAL, 'approve', Submission)),
        'duplicates': (threading.Thread(target=run_duplicate_bot), run_duplicate_bot),
    }

    for thread, _ in threads.values():
        thread.start()

    monitor_threads(threads)
