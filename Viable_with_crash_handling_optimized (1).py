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
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
from openai import OpenAI

# Define the threshold for maximum API requests per minute
MAX_REQUESTS_PER_MINUTE = 100

# Track the number of API requests made within the current minute
api_requests_count = 0
minute_start_time = time.time()

# Thread-safe tracking for per-thread API usage
api_requests_by_thread = defaultdict(int)
api_throttle_lock = threading.Lock()

# Function to make API requests with automatic throttling and logging
def make_api_request():
    global api_requests_count, minute_start_time
    thread_name = threading.current_thread().name

    with api_throttle_lock:
        current_time = time.time()

        # Start new minute
        if current_time - minute_start_time >= 60:
            print("\n⏱️ NEW MINUTE STARTED — resetting counters")
            print(f"🔁 Total API Requests Last Minute: {api_requests_count}")
            for tname, count in api_requests_by_thread.items():
                print(f"  • {tname}: {count} requests")
            print("———————————————————————————————")

            minute_start_time = current_time
            api_requests_count = 0
            for key in api_requests_by_thread:
                api_requests_by_thread[key] = 0  # reset per-thread count

        # Throttle if needed
        if api_requests_count >= MAX_REQUESTS_PER_MINUTE:
            throttle_duration = 60 - (current_time - minute_start_time) + 5
            print(f"[{thread_name}] 🚫 API limit hit ({api_requests_count} total). Throttling {int(throttle_duration)}s...")
            time.sleep(throttle_duration)
            minute_start_time = time.time()
            api_requests_count = 0
            for key in api_requests_by_thread:
                api_requests_by_thread[key] = 0

        # Log usage
        api_requests_count += 1
        api_requests_by_thread[thread_name] += 1

        print(f"[{thread_name}] ✅ API request #{api_requests_by_thread[thread_name]} (global #{api_requests_count}) @ {datetime.now().strftime('%H:%M:%S')}")

def initialize_reddit():
    return praw.Reddit(client_id='jl-I3OHYH2_VZMC1feoJMQ',
                      client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
                      username='PokeLeakBot3',
                      password='testbot1',
                      user_agent='testbot')

# Function to handle exceptions without waiting
def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and e.response.status_code == 429:
        print("Rate limited by Reddit API. Ignoring error.")
    else:
        print(f"Exception encountered: {str(e)}")

# Function to monitor reported posts
def monitor_reported_posts():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit("PokeLeaks")
    try:
        while True:
            try:
                make_api_request()
                for post in subreddit.mod.reports():
                    if post.approved:
                        post.mod.approve()
                        print(f"Post {post.id} has been approved again")
            except Exception as e:
                handle_exception(e)
    except Exception as e:
        handle_exception(e)

# Function to handle reported items in the mod queue
def handle_modqueue_items():
    reddit = initialize_reddit()
    timers = {}

    try:
        while True:
            try:
                make_api_request()
                for item in reddit.subreddit('PokeLeaks').mod.modqueue():
                    if item.num_reports == 1 and item.id not in timers:
                        created_time = item.created_utc
                        timers[item.id] = time.time()
                        print(f"Starting timer for post {item.id}...")
    
                    if item.id in timers:
                        start_time = timers[item.id]
                        time_diff = time.time() - start_time
                        if time_diff >= 3600:
                            try:
                                item.mod.approve()
                                print(f"Approved post {item.id} with one report")
                                del timers[item.id]
                            except prawcore.exceptions.ServerError as e:
                                handle_exception(e)
                        else:
                            new_reports = item.report_reasons
                            if new_reports != item.report_reasons:
                                print(f"New reports for post {item.id}, leaving post in mod queue")
                                del timers[item.id]
                            else:
                                time_remaining = int(start_time + 3600 - time.time())
                                print(f"Time remaining for post {item.id}: {time_remaining} seconds")
            except Exception as e:
                handle_exception(e)
    except Exception as e:
        handle_exception(e)

# Function to handle marking/unmarking posts as spoilers
def handle_spoiler_status():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    previous_spoiler_status = {}

    while True:
        try:
            make_api_request()
            for submission in subreddit.new():
                if submission.id not in previous_spoiler_status:
                    previous_spoiler_status[submission.id] = submission.spoiler
                    continue
                if previous_spoiler_status[submission.id] != submission.spoiler:
                    # Check if the change was made by a moderator
                    is_moderator = submission.author in subreddit.moderator()
                    if not submission.spoiler:
                        if not is_moderator:
                            try:
                                print(f'Post {submission.id} has been unmarked as a spoiler by a non-moderator. Remarking as spoiler again.')
                                submission.mod.spoiler()
                            except prawcore.exceptions.ServerError as e:
                                handle_exception(e)
                        else:
                            print(f'Post {submission.id} has been unmarked as a spoiler by a moderator. Will not re-spoiler the post.')
                    previous_spoiler_status[submission.id] = submission.spoiler
    
        except Exception as e:
            handle_exception(e)
# Function to handle user reports and removal
def handle_user_reports_and_removal():
    try:
        reddit = initialize_reddit()
        subreddit = reddit.subreddit("PokeLeaks")
        # Set up a dictionary of removal reasons to their report threshold
        thresholds = {'Comments complaining about Riddler Khu will be removed. Ignore the post if you are going to complain': 1, 'No insults or harassment of other subreddit members in the comments': 1, 'No asking about or sharing of XCI or NSP files': 1}

        while True:
            try:
                make_api_request()
                # Check the mod queue for new activities
                for comment in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
                    # Check if the item is a comment and has any user reports
                    if isinstance(comment, praw.models.Comment) and comment.user_reports:
                        # Check if the report reason is in the threshold dict
                        if comment.user_reports[0][0] in thresholds:
                            # Check if the number of reports for that reason exceeds the threshold
                            if comment.user_reports[0][1] >= thresholds[comment.user_reports[0][0]]:
                                # Remove the comment
                                try:
                                    comment.mod.remove()
                                    print(f'Comment "{comment.body}" removed due to {comment.user_reports[0][1]} reports for reason: {comment.user_reports[0][0]}')
                                except prawcore.exceptions.ServerError as e:
                                    handle_exception(e)
            except Exception as e:
                handle_exception(e)
    except Exception as e:
        handle_exception(e)

# Function to handle removal of submissions based on user reports
def handle_submissions_based_on_user_reports():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            make_api_request()
            for post in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
                if isinstance(post, praw.models.Submission) and post.user_reports:
                    if post.user_reports[0][0] in thresholds:
                        if post.user_reports[0][1] >= thresholds[post.user_reports[0][0]]:
                            try:
                                post.mod.approve()
                                print(f'post "{post.title}" approved due to {post.user_reports[0][1]} reports for reason: {post.user_reports[0][0]}')
                            except prawcore.exceptions.ServerError as e:
                                handle_exception(e)
    
        except Exception as e:
            handle_exception(e)
def handle_posts_based_on_removal():
    reddit = initialize_reddit()
    thresholds = {'Posts must be about leaks, riddles, news, and rumours about Pokémon content': 2, 'Post must not contain any profanities, vulgarity, sexual content, slurs, be appropriate in nature': 2, 'No reposting of posts already up on the subreddit': 1, 'No memes allowed': 2, 'No joke submissions or shitposts': 2, 'No Self Advertisements/Promotion': 2, 'No Fan art': 2, 'Theories, Questions, Speculations must be commented in the Theory/Speculation/Question Megathread': 2, 'Posts with spoilers must have required spoiler flair, indicate spoiler alert in title, and be vague': 3, 'Retarded': 1}

    while True:
        try:
            make_api_request()
            # Check the mod queue for new activities
            for post in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
                if isinstance(post, praw.models.Submission) and post.user_reports:
                # Check if the item has any user reports
                    if post.user_reports:
                        # Check if the report reason is in the threshold dict
                        if post.user_reports[0][0] in thresholds:
                            # Check if the number of reports for that reason exceed the threshold
                            if post.user_reports[0][1] >= thresholds[post.user_reports[0][0]]:
                                # Remove the submission
                                try:
                                    post.mod.remove()
                                    print(f'Submission "{post.title}" removed due to {post.user_reports[0][1]} reports for reason: {post.user_reports[0][0]}')
                                except prawcore.exceptions.ServerError as e:
                                    handle_exception(e)
    
        except Exception as e:
            handle_exception(e)
def handle_comments_based_on_approval():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            make_api_request()
            # Check the mod queue for new activities
            for comment in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
                # Check if the item has any user reports
                if comment.user_reports:
                    # Check if the report reason is in the threshold dict
                    if comment.user_reports[0][0] in thresholds:
                        # Check if the number of reports for that reason exceeds the threshold
                        if comment.user_reports[0][1] >= thresholds[comment.user_reports[0][0]]:
                            # Remove the comment
                            try:
                                comment.mod.approve()
                                print(f'Comment "{comment.body}" approved due to {comment.user_reports[0][1]} reports for reason: {comment.user_reports[0][0]}')
                            except prawcore.exceptions.ServerError as e:
                                handle_exception(e)
    
        except Exception as e:
            handle_exception(e)
def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    image_hashes = {}
    moderator_removed_hashes = set()  # Track images removed by moderators
    processed_modqueue_submissions = set()
    approved_by_moderator = set()  # Keep track of submissions approved by moderators
    current_time = int(time.time())

    def post_comment(submission, original_post_author, original_post_title, original_post_link, original_post_date):
        max_retries = 3  # Number of times to retry posting the comment
        retries = 0
        while retries < max_retries:
            try:
                comment = submission.reply(
                    ">| Title | Date and Time |\n"
                    ">|:---:|:---:|\n"
                    ">| [{}]({}) | {} |\n\n"
                    ">[Link to Original Post]({})"
                    .format(original_post_title, original_post_link,
                            original_post_date, original_post_link)
                )
                comment.mod.distinguish(sticky=True)
                print("Duplicate removed and comment posted: ", submission.url)
                return True  # Comment successfully posted
            except Exception as e:
                handle_exception(e)
                retries += 1
                time.sleep(1)  # Wait for a few seconds before retrying
        return False  # Comment posting failed after retries

    def check_removed_original_posts():
        while True:
            try:
                make_api_request()
                try:
                    for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                        original_submission = reddit.submission(id=submission_id)
                        original_author = original_submission.author
                        banned_by_moderator = original_submission.banned_by
    
                        # Check if the post was removed by a moderator
                        if banned_by_moderator is not None:
                            moderator_removed_hashes.add(hash_value)  # Add hash to the blocked list
                            del image_hashes[hash_value]  # Remove the hash from active tracking
                        elif original_author is None:  # User deleted their post
                            del image_hashes[hash_value]
                except Exception as e:
                    handle_exception(e)
    
            except Exception as e:
                handle_exception(e)
    # Start a separate thread to continuously check for removed original posts
    original_post_checker_thread = threading.Thread(target=check_removed_original_posts)
    original_post_checker_thread.start()

    try:
        for submission in subreddit.new(limit=99999):
            if isinstance(submission, praw.models.Submission):
                print("Scanning image/post: ", submission.url)
                if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    try:
                        image_data = requests.get(submission.url).content
                        img = np.asarray(bytearray(image_data), dtype=np.uint8)
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        hash_value = str(imagehash.phash(Image.fromarray(gray)))

                        # Block reposts of moderator-removed images
                        if hash_value in moderator_removed_hashes:
                            if not submission.approved:
                                submission.mod.remove()
                                print("Repost of a moderator-removed image removed: ", submission.url)
                                continue  # Skip further processing

                        # Check for duplicates
                        if hash_value in image_hashes:
                            original_submission_id, original_time = image_hashes[hash_value]
                            original_submission = reddit.submission(id=original_submission_id)
                            original_post_author = original_submission.author
                            original_post_link = f"https://www.reddit.com{original_submission.permalink}"

                            # Allow if the resubmission is by the same author
                            if submission.author == original_post_author:
                                print("Resubmission by the same author allowed: ", submission.url)
                                image_hashes[hash_value] = (submission.id, submission.created_utc)
                            else:
                                # Handle duplicate by a different author
                                if submission.created_utc > original_time:
                                    if not submission.approved:
                                        submission.mod.remove()
                                        print("Duplicate removed: ", submission.url)
                                        post_comment(submission, original_post_author.name, original_submission.title, original_post_link, datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'))
                                    else:
                                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                        else:
                            image_hashes[hash_value] = (submission.id, submission.created_utc)
                    except Exception as e:
                        handle_exception(e)
    except prawcore.exceptions.ServerError:
        handle_exception()

    while True:
        try:
            make_api_request()
            try:
                for submission in subreddit.new():
                    if submission.created_utc > current_time:
                        if isinstance(submission, praw.models.Submission):
                            print("Scanning new image/post: ", submission.url)
                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                try:
                                    image_data = requests.get(submission.url).content
                                    img = np.asarray(bytearray(image_data), dtype=np.uint8)
                                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                    hash_value = str(imagehash.phash(Image.fromarray(gray)))
    
                                    # Block reposts of moderator-removed images
                                    if hash_value in moderator_removed_hashes:
                                        if not submission.approved:
                                            submission.mod.remove()
                                            print("Repost of a moderator-removed image removed: ", submission.url)
                                            continue
    
                                    # Check for duplicates
                                    if hash_value in image_hashes:
                                        original_submission_id, original_time = image_hashes[hash_value]
                                        original_submission = reddit.submission(id=original_submission_id)
                                        original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                                        if submission.created_utc > original_time:
                                            if not submission.approved:
                                                submission.mod.remove()
                                                print("Duplicate removed: ", submission.url)
                                                post_comment(submission, original_submission.author.name, original_submission.title, original_post_link, datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'))
                                            else:
                                                image_hashes[hash_value] = (submission.id, submission.created_utc)
                                    else:
                                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                                except Exception as e:
                                    handle_exception(e)
                                    continue
                            current_time = int(time.time())
    
                modqueue_submissions = subreddit.mod.modqueue(only='submission', limit=None)
                modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
    
                for submission in modqueue_submissions:
                    if isinstance(submission, praw.models.Submission):
                        print("Scanning Mod Queue: ", submission.url)
                        if submission.num_reports > 0:
                            print("Skipping reported image: ", submission.url)
                            image_hashes = {k: v for k, v in image_hashes.items() if v[0] != submission.id}
                            continue
    
                        if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                            image_data = requests.get(submission.url).content
                            img = np.asarray(bytearray(image_data), dtype=np.uint8)
                            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            hash_value = str(imagehash.phash(Image.fromarray(gray)))
    
                            if hash_value in image_hashes:
                                original_submission_id, original_time = image_hashes[hash_value]
                                original_submission = reddit.submission(id=original_submission_id)
                                original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                                original_post_title = original_submission.title
                                original_post_author = original_submission.author
    
                                if submission.id != original_submission_id:
                                    if submission.created_utc > original_time:
                                        if not submission.approved:
                                            submission.mod.remove()
                                            if post_comment(submission, original_post_author.name, original_post_title, original_post_link, original_post_date):
                                                processed_modqueue_submissions.add(submission.id)
                                        else:
                                            image_hashes[hash_value] = (submission.id, submission.created_utc)
                                    else:
                                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                                else:
                                    if not submission.approved:
                                        submission.mod.approve()
                                        print("Original submission approved: ", submission.url)
                            else:
                                image_hashes[hash_value] = (submission.id, submission.created_utc)
                        else:
                            image_hashes = {k: v for k, v in image_hashes.items() if v[0] != submission.id}
            except Exception as e:
                handle_exception(e)
    
        except Exception as e:
            handle_exception(e)
def run_moderation_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    openai_client = OpenAI(api_key="")
    subreddit_name = "PokeLeaks"
    human_moderators = ["u/Gismo69", "u/vagrantwade", "u/Aether13", "u/cmholde2"]
    bot_decisions = []

    def fetch_automod_rules():
        try:
            wiki_page = reddit.subreddit(subreddit_name).wiki["config/automoderator"]
            return wiki_page.content_md
        except Exception as e:
            print("❌ Error fetching AutoMod rules:", e)
            return ""

    def parse_automod_rules(content):
        rules = {}
        rule_pattern = r"Rule #(\d+):\s*(.*?)(?=\nRule #|\Z)"
        matches = re.findall(rule_pattern, content, re.DOTALL)
        for number, text in matches:
            rules[int(number)] = text.strip()
        return rules

    def gpt_decision(comment_text, rules_summary):
        prompt =f"""
You are a Reddit moderator bot that mimics a human moderator. Analyze the following comment using common Reddit rules, community standards, and AutoMod rules below. Pay special attention to the tone and context, especially when profanity or strong language is used. If the language is used in excitement, humor, or lightheartedness, please indicate that in your reasoning.

    Comment:
    \"\"\"{comment_text}\"\"\"

    AutoMod Rules:
    {rules_summary}

    Please determine if the language is inappropriate or if it can be approved based on context. If the profanity is used in excitement, humor, or lightheartedness, indicate this in your reasoning. Please ignore the comments that involve Khu. Those should stay removed. Comments containing sexual and political themes should be removed.

    Reply only with either:
    APPROVE: <short reason>
    REMOVE: <short reason>
""" 
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=100
            )
            content = response.choices[0].message.content.strip()
            print("🧠 GPT Response:", content)
            if content.upper().startswith("REMOVE"):
                return "REMOVE", content
            elif content.upper().startswith("APPROVE"):
                return "APPROVE", content
            else:
                return "NEUTRAL", content
        except Exception as e:
            print("❌ GPT Error:", e)
            return "NEUTRAL", str(e)

    def analyze_similarity(comment, approved, removed, result_holder):
        try:
            all_comments = approved + removed
            if not all_comments:
                result_holder["similar"] = False
                return
            texts = [comment.body] + [c.body for c in all_comments]
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(texts)
            sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).max()
            result_holder["similar"] = sim_score > 0.5
            result_holder["similarity_score"] = sim_score
        except Exception as e:
            print("❌ Similarity analysis failed:", e)
            result_holder["similar"] = False

    def learn_from_mods(subreddit):
        approved, removed = [], []
        print("📘 Learning from mod logs...")
        for log in subreddit.mod.log(limit=100000, action="approvecomment"):
            try:
                approved.append(reddit.comment(log.target_fullname.split("_")[1]))
            except:
                continue
        for log in subreddit.mod.log(limit=100000, action="removecomment"):
            try:
                removed.append(reddit.comment(log.target_fullname.split("_")[1]))
            except:
                continue
        print(f"✅ Learned {len(approved)} approved and {len(removed)} removed comments.")
        return approved, removed

    def was_removed_by_automod(comment):
        try:
            logs = list(comment.subreddit.mod.log(limit=100000, action="removecomment"))
            for log in logs:
                if log.target_fullname == comment.fullname and \
                   isinstance(log.mod, praw.models.Redditor) and log.mod.name.lower() == "automoderator":
                    print("📛 Removed by AutoMod confirmed.")
                    return True
        except Exception as e:
            print("⚠️ Could not check mod log:", e)
        return False

    def store_bot_decision(comment, decision, explanation, human_action):
        bot_decisions.append({
            "comment_id": comment.id,
            "bot_decision": decision,
            "bot_reason": explanation,
            "human_action": human_action
        })

    # --- MAIN MONITOR FUNCTION ---
    subreddit = reddit.subreddit(subreddit_name)
    approved, removed = learn_from_mods(subreddit)
    rules_raw = fetch_automod_rules()
    parsed_rules = parse_automod_rules(rules_raw)
    rules_summary = "\n".join([f"Rule #{k}: {v}" for k, v in parsed_rules.items()])

    print("🚀 Monitoring AutoMod-removed comments in real-time...")

    for comment in subreddit.stream.comments(skip_existing=True):
        time.sleep(2)
        if not was_removed_by_automod(comment):
            continue

        print(f"\n🗨️ New AutoMod-Removed Comment: {comment.body}")
        result_holder = {}

        sim_thread = Thread(target=analyze_similarity, args=(comment, approved, removed, result_holder))
        sim_thread.start()

        decision, explanation = gpt_decision(comment.body, rules_summary)
        result_holder["gpt_decision"] = decision
        result_holder["gpt_reason"] = explanation

        sim_thread.join()

        is_similar = result_holder.get("similar", False)
        print(f"🔎 Similar to human-approved: {'✅ Yes' if is_similar else '❌ No'}")
        print(f"🤖 GPT Decision: {decision} — {explanation}")

        human_action = None
        for log in comment.subreddit.mod.log(limit=1000):
            if log.target_fullname == comment.fullname:
                human_action = log.action
                break

        store_bot_decision(comment, decision, explanation, human_action)

        if decision == "REMOVE":
            comment.mod.remove()
            print("❌ Action: Removed (confirmed by GPT)")
        elif decision == "APPROVE" or is_similar:
            comment.mod.approve()
            print("✅ Action: Approved")
        else:
            print("⏳ Action: No action taken")

        time.sleep(1)


def monitor_threads(threads):
    while True:
        try:
            for thread_name, thread in threads.items():
                if not thread.is_alive():
                    print(f"Thread {thread_name} has stopped. Restarting...")
                    new_thread = threading.Thread(target=thread._target)
                    threads[thread_name] = new_thread
                    new_thread.start()
            time.sleep(10)
    
    
        except Exception as e:
            handle_exception(e)
if __name__ == "__main__":
    # Create threads for each function
    threads = {
        'modqueue_thread': threading.Thread(target=handle_modqueue_items),
        'reported_posts_thread': threading.Thread(target=monitor_reported_posts),
        'spoiler_status_thread': threading.Thread(target=handle_spoiler_status),
        'user_reports_removal_thread': threading.Thread(target=handle_user_reports_and_removal),
        'submissions_based_on_user_reports_thread': threading.Thread(target=handle_submissions_based_on_user_reports),
        'posts_based_on_removal_thread': threading.Thread(target=handle_posts_based_on_removal),
        'comments_based_on_approval_thread': threading.Thread(target=handle_comments_based_on_approval),
        'run_pokemon_duplicate_bot_thread': threading.Thread(target=run_pokemon_duplicate_bot),
        'run_moderation_bot_thread' : threading.Thread(target=run_moderation_bot),
    }
    
    
    # Start all threads
    for thread in threads.values():
        thread.start()

    # Monitor threads and restart them if they stop
    monitor_threads(threads)

