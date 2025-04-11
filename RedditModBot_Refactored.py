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
from functools import wraps

# Shared Reddit instance
reddit = praw.Reddit(
    client_id='jl-I3OHYH2_VZMC1feoJMQ',
    client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
    username='PokeLeakBot3',
    password='testbot1',
    user_agent='testbot'
)

# Throttling variables
MAX_REQUESTS_PER_MINUTE = 100
api_requests_count = 0
minute_start_time = time.time()

# API throttling mechanism
def make_api_request():
    global api_requests_count, minute_start_time
    current_time = time.time()
    if current_time - minute_start_time >= 60:
        minute_start_time = current_time
        api_requests_count = 0
    if api_requests_count >= MAX_REQUESTS_PER_MINUTE:
        sleep_duration = 60 - (current_time - minute_start_time) + 5
        print(f"[THROTTLE] Sleeping for {sleep_duration:.2f}s to respect rate limits.")
        time.sleep(sleep_duration)
    api_requests_count += 1

# Decorator to apply API throttling
def throttle_api(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        make_api_request()
        return func(*args, **kwargs)
    return wrapper

# Exception handler
def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and e.response.status_code == 429:
        reset_time = int(e.response.headers.get("x-ratelimit-reset", 60))
        print(f"[ERROR] Rate limited. Sleeping for {reset_time + 5} seconds.")
        time.sleep(reset_time + 5)
    elif isinstance(e, prawcore.exceptions.ServerError):
        print("[ERROR] Reddit server error. Retrying in 10s...")
        time.sleep(10)
    else:
        print(f"[ERROR] {str(e)}")

@throttle_api
def fetch_reports(subreddit):
    return list(subreddit.mod.reports())

@throttle_api
def fetch_modqueue(subreddit):
    return list(subreddit.mod.modqueue(limit=100))

def monitor_reported_posts():
    subreddit = reddit.subreddit("PokeLeaks")
    while True:
        try:
            for post in fetch_reports(subreddit):
                if post.approved:
                    post.mod.approve()
                    print(f"[INFO] Reapproved post {post.id}")
        except Exception as e:
            handle_exception(e)

def handle_modqueue_items():
    subreddit = reddit.subreddit('PokeLeaks')
    timers = {}
    while True:
        try:
            for item in fetch_modqueue(subreddit):
                if item.num_reports == 1 and item.id not in timers:
                    timers[item.id] = time.time()
                    print(f"[INFO] Starting timer for post {item.id}")
                if item.id in timers:
                    time_diff = time.time() - timers[item.id]
                    if time_diff >= 3600:
                        try:
                            item.mod.approve()
                            print(f"[INFO] Approved post {item.id} after 1 hour")
                            del timers[item.id]
                        except Exception as e:
                            handle_exception(e)
                    else:
                        remaining = int(3600 - time_diff)
                        print(f"[INFO] Post {item.id} has {remaining}s remaining before action")
        except Exception as e:
            handle_exception(e)

def monitor_threads(threads):
    while True:
        for name, thread in threads.items():
            if not thread.is_alive():
                print(f"[WARNING] Thread '{name}' stopped. Restarting...")
                new_thread = threading.Thread(target=thread._target)
                threads[name] = new_thread
                new_thread.start()
        time.sleep(10)

if __name__ == "__main__":
    threads = {
        'reported_posts_thread': threading.Thread(target=monitor_reported_posts),
        'modqueue_thread': threading.Thread(target=handle_modqueue_items),
    }
    for t in threads.values():
        t.start()
    monitor_threads(threads)
