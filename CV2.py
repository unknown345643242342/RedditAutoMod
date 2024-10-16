import praw
import prawcore.exceptions
import requests
import time
from datetime import datetime
import numpy as np
from PIL import Image
import imagehash
import cv2

# Connect to Reddit
reddit = praw.Reddit(client_id='jl-I3OHYH2_VZMC1feoJMQ',
                     client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
                     username='PokeLeakBot3',
                     password='testbot1',
                     user_agent='testbot')

# Set the subreddit you want the bot to run on
subreddit = reddit.subreddit('PokeLeaks')

# Create a dictionary to store the image hashes, their corresponding submission IDs, and the time of the post
image_hashes = {}

# Create a set to store processed modqueue submission IDs
processed_modqueue_submissions = set()

# Get the current timestamp
current_time = int(time.time())

# Function to handle exceptions without waiting
def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and e.response.status_code == 429:
        print("Rate limited by Reddit API. Ignoring error.")
    else:
        print(f"Exception encountered: {str(e)}")

# Function to post a comment and handle exceptions with retries
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

# Function to continuously check for and remove hashes associated with deleted original posts
def check_removed_original_posts():
    while True:
        try:
            # Check for removed original posts and remove their hashes
            for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                original_submission = reddit.submission(id=submission_id)
                if original_submission.author is None:  # Check if the author of the original post is None (deleted)
                    del image_hashes[hash_value]  # Remove the hash from the dictionary

        except Exception as e:
            handle_exception(e)

# Start a separate thread to continuously check for removed original posts
import threading
original_post_checker_thread = threading.Thread(target=check_removed_original_posts)
original_post_checker_thread.start()

# Scan previous submissions
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
                    if hash_value in image_hashes:
                        original_submission_id, original_time = image_hashes[hash_value]
                        original_submission = reddit.submission(id=original_submission_id)
                        original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                        if submission.created_utc > original_time:
                            submission.mod.remove()
                            print("Duplicate removed: ", submission.url)
                            post_comment(submission, original_submission.author.name, original_submission.title, original_post_link, datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'))
                        else:
                            image_hashes[hash_value] = (submission.id, submission.created_utc)
                    else:
                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                except Exception as e:
                    handle_exception(e)
except prawcore.exceptions.ServerError:
    handle_exception()

# Main loop to continuously check for new submissions
while True:
    try:
        for submission in subreddit.new():
            # Check if the submission is a new one
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
                            if hash_value in image_hashes:
                                original_submission_id, original_time = image_hashes[hash_value]
                                original_submission = reddit.submission(id=original_submission_id)
                                original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                                if submission.created_utc > original_time:
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

        # Get the modqueue submissions sorted by creation time (oldest first)
        modqueue_submissions = subreddit.mod.modqueue(only='submission', limit=None)
        modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)

        # Process modqueue submissions
        for submission in modqueue_submissions:
            if isinstance(submission, praw.models.Submission):
                print("Scanning Mod Queue: ", submission.url)
                # Check if the submission has any reports
                if submission.num_reports > 0:
                    print("Skipping reported image: ", submission.url)
                    # Remove the image URL from the dictionary to avoid processing it again
                    image_hashes = {k: v for k, v in image_hashes.items() if v[0] != submission.id}
                    continue

                if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    # Get the image data
                    image_data = requests.get(submission.url).content

                    # Open the image and calculate the pHash
                    img = np.asarray(bytearray(image_data), dtype=np.uint8)
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    hash_value = str(imagehash.phash(Image.fromarray(gray)))

                    # Check if the image hash is already in the dictionary
                    if hash_value in image_hashes:
                        # If the image hash is already in the dictionary, it's a duplicate
                        # Get the original submission ID and time of the post
                        original_submission_id, original_time = image_hashes[hash_value]
                        original_submission = reddit.submission(id=original_submission_id)
                        original_post_link = f"https://www.reddit.com{original_submission.permalink}"
                        original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                        original_post_title = original_submission.title
                        original_post_author = original_submission.author
                        # Check if the current submission is not the original submission
                        if submission.id != original_submission_id:
                            # Compare the time of the original post with the time of the new post
                            if submission.created_utc > original_time:
                                # Remove the duplicate submission if it hasn't been processed from modqueue yet
                                if submission.id not in processed_modqueue_submissions:
                                    submission.mod.remove()
                                    if post_comment(submission, original_post_author.name, original_post_title, original_post_link, original_post_date):
                                        processed_modqueue_submissions.add(submission.id)
                            else:
                                # Update the dictionary with the new submission ID and time
                                image_hashes[hash_value] = (submission.id, submission.created_utc)
                        else:
                            # If the current submission is the original submission, approve it
                            submission.mod.approve()
                            print("Original submission approved: ", submission.url)
                    else:
                        # If the image hash is not in the dictionary, add it with the submission ID and time
                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                else:
                    # Remove the image URL from the dictionary to avoid processing it again
                    image_hashes = {k: v for k, v in image_hashes.items() if v[0] != submission.id}
    except Exception as e:
        handle_exception(e)
