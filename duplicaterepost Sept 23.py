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
