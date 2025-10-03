def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('pythonbottest333')
    image_hashes = {}
    orb_descriptors = {}  # Store ORB descriptors for images
    video_hashes = {}  # Store hashes for video frames
    moderator_removed_hashes = set()  # Track images/videos removed by moderators
    processed_modqueue_submissions = set()
    approved_by_moderator = set()  # Track submissions approved by moderators
    ai_features = {}  # Cache AI feature vectors for IMAGES ONLY
    current_time = int(time.time())

    # --- Tiny AI similarity model (for images only) ---
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
            print("AI feature extraction error:", e)
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
                print("Duplicate removed and comment posted: ", submission.url)
                return True
            except Exception as e:
                handle_exception(e)
                retries += 1
                time.sleep(1)
        return False

    # Helper: download a short clip via yt-dlp (returns local path or None)
    def download_sample_video(url, duration_seconds=120):
        try:
            import yt_dlp
            import tempfile, os, glob, shutil
            tmpdir = tempfile.mkdtemp(prefix="ytclip_")
            outtmpl = os.path.join(tmpdir, "sample.%(ext)s")
            dl_opts = {
                "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
                "noplaylist": True,
                "quiet": True,
                "outtmpl": outtmpl,
                "download_sections": [f"*00:00:00-00:00:{duration_seconds:02d}"],
            }
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([url])
            files = glob.glob(os.path.join(tmpdir, "sample.*"))
            if not files:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return None
            return files[0]
        except Exception as e:
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
            print("yt-dlp download failed or not available for URL:", url, "error:", e)
            return None

    def check_removed_original_posts():
        while True:
            try:
                for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                    original_submission = reddit.submission(id=submission_id)
                    original_author = original_submission.author
                    banned_by_moderator = original_submission.banned_by
                    if banned_by_moderator is not None:
                        if hash_value not in moderator_removed_hashes:
                            moderator_removed_hashes.add(hash_value)
                            print(f"[MOD REMOVE] Original submission {submission_id} removed by a moderator. Hash kept.")
                    elif original_author is None:
                        try:
                            del image_hashes[hash_value]
                        except Exception:
                            pass
                        if submission_id in orb_descriptors:
                            del orb_descriptors[submission_id]
                        if submission_id in ai_features:
                            del ai_features[submission_id]
                        print(f"[USER REMOVE] Original submission {submission_id} removed by user. Hash deleted.")

                for vid_hash, (submission_id, creation_time) in list(video_hashes.items()):
                    original_submission = reddit.submission(id=submission_id)
                    original_author = original_submission.author
                    banned_by_moderator = original_submission.banned_by
                    if banned_by_moderator is not None:
                        if vid_hash not in moderator_removed_hashes:
                            moderator_removed_hashes.add(vid_hash)
                            print(f"[MOD REMOVE] Original video {submission_id} removed by a moderator. Hash kept.")
                    elif original_author is None:
                        try:
                            del video_hashes[vid_hash]
                        except Exception:
                            pass
                        print(f"[USER REMOVE] Original video {submission_id} removed by user. Hash deleted.")
            except Exception as e:
                handle_exception(e)
            time.sleep(5)

    threading.Thread(target=check_removed_original_posts, daemon=True).start()

    # --- Initial scan ---
    def initial_scan():
        try:
            for submission in subreddit.new(limit=10):
                if isinstance(submission, praw.models.Submission):
                    try:
                        if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                            print("Indexing submission (initial scan): ", submission.url)
                            image_data = requests.get(submission.url).content
                            img = np.asarray(bytearray(image_data), dtype=np.uint8)
                            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                            hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
                            if hash_value not in image_hashes:
                                descriptors = get_orb_descriptors_conditional(img)
                                image_hashes[hash_value] = (submission.id, submission.created_utc)
                                orb_descriptors[submission.id] = descriptors
                                ai_features[submission.id] = get_ai_features(img)
                        elif submission.is_video and 'v.redd.it' in submission.url:
                            print("Indexing video submission (initial scan): ", submission.url)
                            try:
                                video_url = submission.media['reddit_video']['fallback_url']
                                cap = cv2.VideoCapture(video_url)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                                sample_frames = min(45, frame_count)
                                for i in range(sample_frames):
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                    ret, frame = cap.read()
                                    if not ret:
                                        continue
                                    hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
                                    if hash_value not in video_hashes:
                                        video_hashes[hash_value] = (submission.id, submission.created_utc)
                                cap.release()
                            except Exception as e:
                                handle_exception(e)
                        else:
                            try:
                                print("Attempting to index external video (initial scan): ", submission.url)
                                local_clip = download_sample_video(submission.url, duration_seconds=5)
                                if local_clip is None:
                                    continue
                                cap = cv2.VideoCapture(local_clip)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                                sample_frames = min(45, frame_count)
                                for i in range(sample_frames):
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                    ret, frame = cap.read()
                                    if not ret:
                                        continue
                                    hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
                                    if hash_value not in video_hashes:
                                        video_hashes[hash_value] = (submission.id, submission.created_utc)
                                cap.release()
                                try:
                                    import os, shutil
                                    shutil.rmtree(os.path.dirname(local_clip), ignore_errors=True)
                                except Exception:
                                    pass
                            except Exception as e:
                                handle_exception(e)
                    except Exception as e:
                        handle_exception(e)
        except Exception as e:
            handle_exception(e)

    initial_scan()  # run only once

    # --- Video worker (uses phash only, no AI) ---
    def video_worker():
        nonlocal video_hashes, moderator_removed_hashes, processed_modqueue_submissions

        def video_modqueue():
            nonlocal video_hashes, moderator_removed_hashes, processed_modqueue_submissions
            while True:
                try:
                    # Process video modqueue submissions
                    modqueue_submissions = subreddit.mod.modqueue(only='submission', limit=None)
                    modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                    for submission in modqueue_submissions:
                        if not isinstance(submission, praw.models.Submission):
                            continue
                        # skip images
                        if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                            continue

                        print("Scanning Mod Queue (video worker - modqueue): ", submission.url)

                        if submission.num_reports > 0:
                            print("Skipping reported submission: ", submission.url)
                            video_hashes = {k: v for k, v in video_hashes.items() if v[0] != submission.id}
                            continue

                        # --- Reddit v.redd.it handling ---
                        if submission.is_video and 'v.redd.it' in submission.url:
                            try:
                                video_url = submission.media['reddit_video']['fallback_url']
                                cap = cv2.VideoCapture(video_url)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                                sample_frames = min(45, frame_count)
                                duplicate_detected = False

                                for i in range(sample_frames):
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                    ret, frame = cap.read()
                                    if not ret:
                                        continue
                                    hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))

                                    # Skip duplicates removed by mods
                                    if hash_value in moderator_removed_hashes and not submission.approved:
                                        try:
                                            original_submission_id = video_hashes[hash_value][0]
                                            original_submission = reddit.submission(id=original_submission_id)
                                            submission.mod.remove()
                                            post_comment(submission, original_submission.author.name, original_submission.title,
                                                       datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                                       original_submission.created_utc, "Removed by Moderator", original_submission.permalink)
                                            print("Repost of a moderator-removed video frame removed: ", submission.url)
                                            duplicate_detected = True
                                            break
                                        except Exception:
                                            pass

                                    # Check against original video frames (phash only)
                                    if hash_value in video_hashes:
                                        original_id, _ = video_hashes[hash_value]
                                        if not submission.approved:
                                            submission.mod.remove()
                                            original_submission = reddit.submission(id=original_id)
                                            post_comment(submission, original_submission.author.name, original_submission.title,
                                                       datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                                       original_submission.created_utc, "Active", original_submission.permalink)
                                            print("Duplicate video frame removed: ", submission.url)
                                            duplicate_detected = True
                                            break

                                cap.release()

                                # Approve only if no duplicates were detected
                                if not duplicate_detected and not submission.approved:
                                    submission.mod.approve()
                                    print("Original video approved (v.redd.it): ", submission.url)

                                # index sampled frames into video_hashes ONLY if this was NOT a duplicate
                                if not duplicate_detected:
                                    cap = cv2.VideoCapture(video_url)
                                    for i in range(sample_frames):
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                        ret, frame = cap.read()
                                        if not ret:
                                            continue
                                        hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
                                        if hash_value not in video_hashes:
                                            video_hashes[hash_value] = (submission.id, submission.created_utc)
                                    cap.release()
                            except Exception as e:
                                handle_exception(e)

                        # --- External video handling ---
                        else:
                            try:
                                local_clip = download_sample_video(submission.url, duration_seconds=180)
                                if local_clip is None:
                                    continue
                                cap = cv2.VideoCapture(local_clip)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                                sample_frames = min(45, frame_count)
                                duplicate_detected = False

                                for i in range(sample_frames):
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                    ret, frame = cap.read()
                                    if not ret:
                                        continue
                                    hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))

                                    # moderator-removed check
                                    if hash_value in moderator_removed_hashes and not submission.approved:
                                        try:
                                            original_submission_id = video_hashes[hash_value][0]
                                            original_submission = reddit.submission(id=original_submission_id)
                                            submission.mod.remove()
                                            post_comment(submission, original_submission.author.name, original_submission.title,
                                                       datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                                       original_submission.created_utc, "Removed by Moderator", original_submission.permalink)
                                            print("Duplicate external video frame removed (moderator-removed): ", submission.url)
                                            duplicate_detected = True
                                            break
                                        except Exception:
                                            pass

                                    if hash_value in video_hashes:
                                        if not submission.approved:
                                            submission.mod.remove()
                                            original_id, _ = video_hashes[hash_value]
                                            original_submission = reddit.submission(id=original_id)
                                            post_comment(submission, original_submission.author.name, original_submission.title,
                                                       datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                                       original_submission.created_utc, "Active", original_submission.permalink)
                                            print("Duplicate external video frame removed: ", submission.url)
                                            duplicate_detected = True
                                            break

                                cap.release()
                                try:
                                    import os, shutil
                                    shutil.rmtree(os.path.dirname(local_clip), ignore_errors=True)
                                except Exception:
                                    pass

                                if not duplicate_detected and not submission.approved:
                                    submission.mod.approve()
                                    print("Original external video approved: ", submission.url)

                                # index frames (only if not duplicate)
                                if not duplicate_detected:
                                    local_clip = download_sample_video(submission.url, duration_seconds=5)
                                    if local_clip is not None:
                                        cap = cv2.VideoCapture(local_clip)
                                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                                        sample_frames = min(45, frame_count)
                                        for i in range(sample_frames):
                                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                            ret, frame = cap.read()
                                            if not ret:
                                                continue
                                            hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
                                            if hash_value not in video_hashes:
                                                video_hashes[hash_value] = (submission.id, submission.created_utc)
                                        cap.release()
                                        try:
                                            import shutil, os
                                            shutil.rmtree(os.path.dirname(local_clip), ignore_errors=True)
                                        except Exception:
                                            pass
                            except Exception as e:
                                handle_exception(e)
                    time.sleep(5)
                except Exception as e:
                    handle_exception(e)
                    time.sleep(5)

        def video_stream():
            nonlocal video_hashes, moderator_removed_hashes, processed_modqueue_submissions
            for submission in subreddit.stream.submissions(skip_existing=True):
                try:
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        continue
                    if submission.id in processed_modqueue_submissions:
                        continue

                    print("Scanning new submission (video worker - stream): ", submission.url)

                    # --- Video stream processing (Reddit v.redd.it) ---
                    if submission.is_video and 'v.redd.it' in submission.url:
                        try:
                            video_url = submission.media['reddit_video']['fallback_url']
                            cap = cv2.VideoCapture(video_url)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                            sample_frames = min(45, frame_count)
                            duplicate_detected = False

                            for i in range(sample_frames):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                ret, frame = cap.read()
                                if not ret:
                                    continue
                                hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))

                                if hash_value in moderator_removed_hashes and not submission.approved:
                                    try:
                                        original_submission_id = video_hashes[hash_value][0]
                                        original_submission = reddit.submission(id=original_submission_id)
                                        submission.mod.remove()
                                        post_comment(submission, original_submission.author.name, original_submission.title,
                                                   datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                                   original_submission.created_utc, "Removed by Moderator", original_submission.permalink)
                                        print("Duplicate video frame removed (moderator-removed): ", submission.url)
                                        duplicate_detected = True
                                        break
                                    except Exception:
                                        pass

                                if hash_value in video_hashes:
                                    if not submission.approved:
                                        submission.mod.remove()
                                        original_id, _ = video_hashes[hash_value]
                                        original_submission = reddit.submission(id=original_id)
                                        post_comment(submission, original_submission.author.name, original_submission.title,
                                                   datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                                   original_submission.created_utc, "Active", original_submission.permalink)
                                        print("Duplicate video frame removed: ", submission.url)
                                        duplicate_detected = True
                                        break

                            cap.release()

                            if not duplicate_detected and not submission.approved:
                                submission.mod.approve()
                                print("Original video approved (v.redd.it stream): ", submission.url)

                            # index frames only if not duplicate
                            if not duplicate_detected:
                                cap = cv2.VideoCapture(video_url)
                                for i in range(sample_frames):
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                    ret, frame = cap.read()
                                    if not ret:
                                        continue
                                    hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
                                    if hash_value not in video_hashes:
                                        video_hashes[hash_value] = (submission.id, submission.created_utc)
                                cap.release()
                        except Exception as e:
                            handle_exception(e)

                    # --- External video stream processing (non-reddit hosts) ---
                    else:
                        try:
                            local_clip = download_sample_video(submission.url, duration_seconds=180)
                            if local_clip is None:
                                continue
                            cap = cv2.VideoCapture(local_clip)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                            sample_frames = min(45, frame_count)
                            duplicate_detected = False

                            for i in range(sample_frames):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                ret, frame = cap.read()
                                if not ret:
                                    continue
                                hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))

                                if hash_value in moderator_removed_hashes and not submission.approved:
                                    try:
                                        original_submission_id = video_hashes[hash_value][0]
                                        original_submission = reddit.submission(id=original_submission_id)
                                        submission.mod.remove()
                                        post_comment(submission, original_submission.author.name, original_submission.title,
                                                   datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                                   original_submission.created_utc, "Removed by Moderator", original_submission.permalink)
                                        print("Duplicate external video frame removed (moderator-removed): ", submission.url)
                                        duplicate_detected = True
                                        break
                                    except Exception:
                                        pass

                                if hash_value in video_hashes:
                                    if not submission.approved:
                                        submission.mod.remove()
                                        original_id, _ = video_hashes[hash_value]
                                        original_submission = reddit.submission(id=original_id)
                                        post_comment(submission, original_submission.author.name, original_submission.title,
                                                   datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                                                   original_submission.created_utc, "Active", original_submission.permalink)
                                        print("Duplicate external video frame removed: ", submission.url)
                                        duplicate_detected = True
                                        break

                            cap.release()
                            try:
                                import shutil, os
                                shutil.rmtree(os.path.dirname(local_clip), ignore_errors=True)
                            except Exception:
                                pass

                            if not duplicate_detected and not submission.approved:
                                submission.mod.approve()
                                print("Original external video approved (stream): ", submission.url)

                            # index frames only if not duplicate
                            if not duplicate_detected:
                                local_clip = download_sample_video(submission.url, duration_seconds=5)
                                if local_clip is not None:
                                    cap = cv2.VideoCapture(local_clip)
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                                    sample_frames = min(45, frame_count)
                                    for i in range(sample_frames):
                                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * frame_count / sample_frames))
                                        ret, frame = cap.read()
                                        if not ret:
                                            continue
                                        hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
                                        if hash_value not in video_hashes:
                                            video_hashes[hash_value] = (submission.id, submission.created_utc)
                                    cap.release()
                                    try:
                                        import shutil, os
                                        shutil.rmtree(os.path.dirname(local_clip), ignore_errors=True)
                                    except Exception:
                                        pass
                        except Exception as e:
                            handle_exception(e)
                except Exception as e:
                    handle_exception(e)

        # start both video modqueue and stream as separate threads so they don't block each other
        threading.Thread(target=safe_run, args=(video_modqueue,), daemon=True).start()
        threading.Thread(target=safe_run, args=(video_stream,), daemon=True).start()

        # keep this worker alive
        while True:
            time.sleep(5)

    # Start the two dedicated worker threads (image-only and video-only)
    threading.Thread(target=safe_run, args=(image_worker,), daemon=True).start()
    threading.Thread(target=safe_run, args=(video_worker,), daemon=True).start()

    # Keep the function alive so workers keep running
    while True:
        time.sleep(30)
