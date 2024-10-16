import praw
import prawcore
import time
import threading
import requests

# Function to initialize the Reddit API connection
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
            for post in subreddit.mod.reports():
                if post.approved:
                    post.mod.approve()
                    print(f"Post {post.id} has been approved again")
            
            time.sleep(60)
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to handle reported items in the mod queue
def handle_modqueue_items():
    reddit = initialize_reddit()
    timers = {}

    try:
        while True:
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

            time.sleep(60)

    except Exception as e:
        handle_exception(e)

# Function to handle marking/unmarking posts as spoilers
def handle_spoiler_status():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    previous_spoiler_status = {}

    while True:
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

# Function to handle user reports and removal
def handle_user_reports_and_removal():
    try:
        reddit = initialize_reddit()
        subreddit = reddit.subreddit("PokeLeaks")
        # Set up a dictionary of removal reasons to their report threshold
        thresholds = {'Comments complaining about Riddler Khu will be removed. Ignore the post if you are going to complain': 1, 'No insults or harassment of other subreddit members in the comments': 1, 'No asking about or sharing of XCI or NSP files': 1}

        while True:
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

            time.sleep(60)

    except Exception as e:
        handle_exception(e)

# Function to handle removal of submissions based on user reports
def handle_submissions_based_on_user_reports():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        for post in reddit.subreddit('PokeLeaks').mod.modqueue(limit=100):
            if isinstance(post, praw.models.Submission) and post.user_reports:
                if post.user_reports[0][0] in thresholds:
                    if post.user_reports[0][1] >= thresholds[post.user_reports[0][0]]:
                        try:
                            post.mod.approve()
                            print(f'post "{post.title}" approved due to {post.user_reports[0][1]} reports for reason: {post.user_reports[0][0]}')
                        except prawcore.exceptions.ServerError as e:
                            handle_exception(e)
        time.sleep(60)

def handle_posts_based_on_removal():
    reddit = initialize_reddit()
    thresholds = {'Posts must be about leaks, riddles, news, and rumours about Pokémon content': 2, 'Post must not contain any profanities, vulgarity, sexual content, slurs, be appropriate in nature': 2, 'No reposting of posts already up on the subreddit': 1, 'No memes allowed': 2, 'No joke submissions or shitposts': 2, 'No Self Advertisements/Promotion': 2, 'No Fan art': 2, 'Theories, Questions, Speculations must be commented in the Theory/Speculation/Question Megathread': 2, 'Posts with spoilers must have required spoiler flair, indicate spoiler alert in title, and be vague': 3, 'Retarded': 1}

    while True:
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

            time.sleep(60)

def handle_comments_based_on_approval():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
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
        time.sleep(60)

def process_reddit_posts():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit("PokeLeaks")

    # Set up the Discord webhook
    webhook_url = 'https://discord.com/api/webhooks/1079468430593376266/oJuM4zm6ocbET2XsF2uWxAhoC9cve3cYiGu_3iWRjU2rNQFggIMbB_3RPL6J_w1kk-1C'

    # Set up the subreddit and message payload
    subreddit = reddit.subreddit('PokeLeaks')
    payload = {
        'username': '',
        'avatar_url': None,
        'content': '',
        'thread_name': '',
        'thread_tags': []
    }

    # Define the mappings between flairs and subcategory tags
    flair_to_subcategory_tag = {
        'News': 'News',
        'Insider Information': 'Insider Information',
        'Pre-Release Game Leak': 'Pre-Release Game Leak',
        'Unverified': 'Unverified',
        '4Chan': '4Chan',
        'Datamine': 'Datamine',
        'Riddle': 'Riddle',
        'TCG Leak': 'TCG Leak',
        'Anime Leak': 'Anime Leak',
        'Merchanise Leak': 'Merchanise Leak',
        'Video': 'Video',
        # add more mappings as needed
    }

    # Get the current webhook data to extract the avatar URL
    webhook_data = requests.get(webhook_url).json()
    if 'avatar_url' in webhook_data:
        payload['avatar_url'] = webhook_data['avatar_url']

    # Main script logic
    processed_post_ids = set()  # Maintain a set to track processed post IDs
    while True:
        try:
            # Listen for new posts on the subreddit
            for submission in subreddit.stream.submissions(skip_existing=True):
                submission_time = submission.created_utc
                current_time = time.time()

                # Compare the post's creation time with the current time
                if current_time - submission_time < 604800:
                    if submission.id not in processed_post_ids:
                        # Add the post ID to the set of processed post IDs
                        processed_post_ids.add(submission.id)

                        # Process the new post
                        payload['content'] = f'{submission.title}\n{submission.url}'
                        payload['username'] = f'u/{submission.author.name}'
                        payload['thread_name'] = submission.title
                        flair = submission.link_flair_text
                        if flair in flair_to_subcategory_tag:
                            payload['thread_tags'] = [flair_to_subcategory_tag[flair]]

                        response = requests.post(webhook_url, json=payload)

                        if response.status_code == 204:
                            print('Successfully sent message to Discord')
                        else:
                            print(f'Failed to send message to Discord: {response.text}')

        except prawcore.exceptions.TooManyRequests:
            handle_exception()
            
            time.sleep(60)

if __name__ == "__main__":
    # Create threads for each function
    modqueue_thread = threading.Thread(target=handle_modqueue_items)
    reported_posts_thread = threading.Thread(target=monitor_reported_posts)
    spoiler_status_thread = threading.Thread(target=handle_spoiler_status)
    user_reports_removal_thread = threading.Thread(target=handle_user_reports_and_removal)
    submissions_based_on_user_reports_thread = threading.Thread(target=handle_submissions_based_on_user_reports)
    posts_based_on_removal_thread = threading.Thread(target=handle_posts_based_on_removal)
    comments_based_on_approval_thread = threading.Thread(target=handle_comments_based_on_approval)
    process_reddit_posts_thread = threading.Thread(target=process_reddit_posts)

    # Start all threads
    modqueue_thread.start()
    reported_posts_thread.start()
    spoiler_status_thread.start()
    user_reports_removal_thread.start()
    submissions_based_on_user_reports_thread.start()
    posts_based_on_removal_thread.start()
    comments_based_on_approval_thread.start()
    process_reddit_posts_thread.start()

    # Wait for all threads to finish (if needed)
    modqueue_thread.join()
    reported_posts_thread.join()
    spoiler_status_thread.join()
    user_reports_removal_thread.join()
    submissions_based_on_user_reports_thread.join()
    posts_based_on_removal_thread.join()
    comments_based_on_approval_thread.join()
    process_reddit_posts_thread.join()