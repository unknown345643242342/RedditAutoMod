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
                            sample_frames = min(30, frame_count)
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
                                        processed_modqueue_submissions.add(submission.id)
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
                                        processed_modqueue_submissions.add(submission.id)
                                        break

                            cap.release()

                            # Approve only if no duplicates were detected
                            if not duplicate_detected and not submission.approved:
                                submission.mod.approve()
                                print("Original video approved (v.redd.it): ", submission.url)
                                processed_modqueue_submissions.add(submission.id)

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
                            sample_frames = min(30, frame_count)
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
                                        processed_modqueue_submissions.add(submission.id)
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
                                        processed_modqueue_submissions.add(submission.id)
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
                                processed_modqueue_submissions.add(submission.id)

                            # index frames (only if not duplicate)
                            if not duplicate_detected:
                                local_clip = download_sample_video(submission.url, duration_seconds=5)
                                if local_clip is not None:
                                    cap = cv2.VideoCapture(local_clip)
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                                    sample_frames = min(30, frame_count)
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
                    print(f"Skipping already processed submission from modqueue: {submission.url}")
                    continue

                print("Scanning new submission (video worker - stream): ", submission.url)

                # --- Video stream processing (Reddit v.redd.it) ---
                if submission.is_video and 'v.redd.it' in submission.url:
                    try:
                        video_url = submission.media['reddit_video']['fallback_url']
                        cap = cv2.VideoCapture(video_url)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                        sample_frames = min(30, frame_count)
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
                                    processed_modqueue_submissions.add(submission.id)
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
                                    processed_modqueue_submissions.add(submission.id)
                                    break

                        cap.release()

                        if not duplicate_detected and not submission.approved:
                            submission.mod.approve()
                            print("Original video approved (v.redd.it stream): ", submission.url)
                            processed_modqueue_submissions.add(submission.id)

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
                        sample_frames = min(30, frame_count)
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
                                    processed_modqueue_submissions.add(submission.id)
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
                                    processed_modqueue_submissions.add(submission.id)
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
                            processed_modqueue_submissions.add(submission.id)

                        # index frames only if not duplicate
                        if not duplicate_detected:
                            local_clip = download_sample_video(submission.url, duration_seconds=5)
                            if local_clip is not None:
                                cap = cv2.VideoCapture(local_clip)
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                                sample_frames = min(30, frame_count)
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
