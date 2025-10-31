import asyncpraw
import prawcore.exceptions
import aiohttp
import time
from datetime import datetime
import numpy as np
from PIL import Image
import imagehash
import cv2
import asyncio
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

# =========================
# Crash-proof runner
# =========================
async def safe_run(target, *args, **kwargs):
    """
    Keeps a target async function running forever.
    If the function raises, log the error, sleep briefly, and run it again.
    """
    while True:
        try:
            await target(*args, **kwargs)
        except Exception as e:
            print(f"[FATAL] {target.__name__} crashed: {e}")
            traceback.print_exc()
            await asyncio.sleep(10)  # brief cooldown before retrying

# =========================
# Reddit init + error handler
# =========================
async def initialize_reddit():
    return asyncpraw.Reddit(
        client_id='jl-I3OHYH2_VZMC1feoJMQ',
        client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
        username='PokeLeakBot3',
        password='testbot1',
        user_agent='testbot'
    )

def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and getattr(e, "response", None) and e.response.status_code == 429:
        print("Rate limited by Reddit API. Ignoring error.")

# =========================
# Workers
# =========================

async def monitor_reported_posts():
    reddit = await initialize_reddit()
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

async def handle_modqueue_items():
    reddit = await initialize_reddit()
    subreddit = await reddit.subreddit('PokeLeaks')
    timers = {}

    while True:
        try:
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
            await asyncio.sleep(60)

async def handle_spoiler_status():
    reddit = await initialize_reddit()
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
                        moderators = []
                        async for mod in subreddit.moderator():
                            moderators.append(mod)
                        is_moderator = submission.author in moderators
                    except Exception:
                        is_moderator = False  # be safe if something weird happens

                    if not submission.spoiler:
                        if not is_moderator:
                            try:
                                print(f'Post {submission.id} unmarked as spoiler by non-mod. Re-marking.')
                                await submission.mod.spoiler()
                            except prawcore.exceptions.ServerError as se:
                                handle_exception(se)
                        else:
                            print(f'Post {submission.id} unmarked as spoiler by a moderator. Leaving as-is.')
                    previous_spoiler_status[submission.id] = submission.spoiler
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(30)

async def handle_user_reports_and_removal():
    reddit = await initialize_reddit()
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
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)

async def handle_submissions_based_on_user_reports():
    reddit = await initialize_reddit()
    subreddit = await reddit.subreddit('PokeLeaks')
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            async for post in subreddit.mod.modqueue(limit=100):
                if isinstance(post, asyncpraw.models.Submission) and getattr(post, "user_reports", None):
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            await post.mod.approve()
                            print(f'post "{post.title}" approved due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)

async def handle_posts_based_on_removal():
    reddit = await initialize_reddit()
    subreddit = await reddit.subreddit('PokeLeaks')
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
            async for post in subreddit.mod.modqueue(limit=100):
                if isinstance(post, asyncpraw.models.Submission) and getattr(post, "user_reports", None):
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            await post.mod.remove()
                            print(f'Submission "{post.title}" removed due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)

async def handle_comments_based_on_approval():
    reddit = await initialize_reddit()
    subreddit = await reddit.subreddit('PokeLeaks')
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            async for comment in subreddit.mod.modqueue(limit=100):
                if getattr(comment, "user_reports", None):
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            await comment.mod.approve()
                            print(f'Comment "{comment.body}" approved due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            await asyncio.sleep(60)
            
# =========================
# Main: start tasks via safe_run
# =========================
if __name__ == "__main__":
    async def main():
        tasks = []

        def add_task(name, func, *args, **kwargs):
            task = asyncio.create_task(safe_run(func, *args, **kwargs))
            tasks.append(task)
            print(f"[STARTED] {name}")

        add_task('modqueue_task', handle_modqueue_items)
        add_task('reported_posts_task', monitor_reported_posts)
        add_task('spoiler_status_task', handle_spoiler_status)
        add_task('user_reports_removal_task', handle_user_reports_and_removal)
        add_task('submissions_based_on_user_reports_task', handle_submissions_based_on_user_reports)
        add_task('posts_based_on_removal_task', handle_posts_based_on_removal)
        add_task('comments_based_on_approval_task', handle_comments_based_on_approval)

        # Keep the main coroutine alive indefinitely so tasks keep running.
        while True:
            await asyncio.sleep(30)
    
    asyncio.run(main())
