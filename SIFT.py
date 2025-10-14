def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    image_hashes = {}
    sift_descriptors = {}
    moderator_removed_hashes = set()
    processed_modqueue_submissions = set()
    approved_by_moderator = set()
    ai_features = {}
    current_time = int(time.time())

    # --- AI similarity model ---
    device = "cpu"
    efficientnet_model = models.efficientnet_b0(pretrained=True)
    efficientnet_model.eval()
    efficientnet_model.to(device)
    efficientnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # OPTIMIZATION 1: Reuse SIFT detector instead of creating new one each time
    sift_detector = cv2.SIFT_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # --- Helper functions ---
    def get_ai_features(img):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_tensor = efficientnet_transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = efficientnet_model(img_tensor)
                feat = feat / feat.norm(dim=1, keepdim=True)
            return feat
        except Exception as e:
            print("AI feature extraction error:", e)
            return None

    def is_problematic_image(img, white_threshold=0.7, text_threshold=0.05):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_ratio = np.mean(gray > 240)
        
        # OPTIMIZATION 9: Early return if white ratio is high enough
        if white_ratio > white_threshold:
            return True
        
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.mean(edges > 0)
        return edge_ratio > text_threshold

    def preprocess_image_for_sift(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
        edges = cv2.Canny(gray, 100, 200)
        return edges

    def get_sift_descriptors_conditional(img):
        if is_problematic_image(img):
            processed_img = preprocess_image_for_sift(img)
        else:
            processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # OPTIMIZATION 10: Use pre-created SIFT detector
        kp, des = sift_detector.detectAndCompute(processed_img, None)
        return des

    def sift_similarity(desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0
        # OPTIMIZATION 11: Use pre-created BFMatcher
        matches = bf_matcher.match(desc1, desc2)
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

    def load_and_process_image(url):
        """Load image from URL and compute hash, descriptors, and AI features"""
        # OPTIMIZATION 12: Set timeout for requests to prevent hanging
        image_data = requests.get(url, timeout=10).content
        img = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # OPTIMIZATION 13: Compute grayscale once and reuse
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_pil = Image.fromarray(gray)
        phash_val = str(imagehash.phash(gray_pil))
        whash_val = str(imagehash.whash(gray_pil))
        dhash_val = str(imagehash.dhash(gray_pil))
        hash_value = f"{phash_val}_{whash_val}_{dhash_val}"
        
        descriptors = get_sift_descriptors_conditional(img)
        features = get_ai_features(img)
        
        return img, hash_value, descriptors, features

    def get_cached_ai_features(submission_id):
        """Get AI features from cache or compute them"""
        if submission_id in ai_features:
            return ai_features[submission_id]
        
        old_submission = reddit.submission(id=submission_id)
        old_image_data = requests.get(old_submission.url, timeout=10).content
        old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        old_features = get_ai_features(old_img)
        ai_features[submission_id] = old_features
        return old_features

    def calculate_ai_similarity(features1, features2):
        """Calculate AI similarity score between two feature vectors"""
        if features1 is not None and features2 is not None:
            return (features1 @ features2.T).item()
        return 0

    def check_hash_duplicate(submission, hash_value, new_features):
        """Check if submission is a hash-based duplicate"""
        if hash_value not in image_hashes:
            return False, None, None, None, None, None, None
        
        original_id, original_time = image_hashes[hash_value]
        
        if submission.id == original_id or submission.created_utc <= original_time:
            return False, None, None, None, None, None, None
        
        original_submission = reddit.submission(id=original_id)
        original_features = get_cached_ai_features(original_submission.id)
        
        ai_score = calculate_ai_similarity(new_features, original_features)
        
        print(f"Hash match detected. AI similarity: {ai_score:.2f}")
        
        if ai_score > 0.70:
            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            original_status = "Removed by Moderator" if hash_value in moderator_removed_hashes else "Active"
            return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def check_sift_duplicate(submission, descriptors, new_features):
        """Check if submission is a SIFT-based duplicate"""
        for old_id, old_desc in sift_descriptors.items():
            sim = sift_similarity(descriptors, old_desc)
            
            if sim > 0.30:
                old_features = get_cached_ai_features(old_id)
                
                ai_score = calculate_ai_similarity(new_features, old_features)
                
                print(f"SIFT match detected. AI similarity: {ai_score:.2f}")
                
                if ai_score > 0.70:
                    original_submission = reddit.submission(id=old_id)
                    original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    old_hash = next((h for h, v in image_hashes.items() if v[0] == old_id), None)
                    original_status = "Removed by Moderator" if old_hash and old_hash in moderator_removed_hashes else "Active"
                    
                    return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def handle_duplicate(submission, is_hash_dup, detection_method, author, title, date, utc, status, permalink):
        """Remove duplicate and post comment if not approved"""
        if not submission.approved:
            submission.mod.remove()
            post_comment(submission, author, title, date, utc, status, permalink)
            print(f"Duplicate removed by {detection_method}: {submission.url}")
        return True

    def handle_moderator_removed_repost(submission, hash_value):
        """Handle reposts of moderator-removed images"""
        if hash_value in moderator_removed_hashes and not submission.approved:
            submission.mod.remove()
            original_submission = reddit.submission(id=image_hashes[hash_value][0])
            post_comment(
                submission,
                original_submission.author.name,
                original_submission.title,
                datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                original_submission.created_utc,
                "Removed by Moderator",
                original_submission.permalink
            )
            print("Repost of a moderator-removed image removed: ", submission.url)
            return True
        return False

    def process_submission_for_duplicates(submission, context="stream"):
        """Main duplicate detection logic - works for both mod queue and stream"""
        try:
            img, hash_value, descriptors, new_features = load_and_process_image(submission.url)
            ai_features[submission.id] = new_features
            
            if handle_moderator_removed_repost(submission, hash_value):
                return True
            
            is_duplicate, author, title, date, utc, status, permalink = check_hash_duplicate(
                submission, hash_value, new_features
            )
            if is_duplicate:
                return handle_duplicate(submission, True, "hash + AI", author, title, date, utc, status, permalink)
            
            is_duplicate, author, title, date, utc, status, permalink = check_sift_duplicate(
                submission, descriptors, new_features
            )
            if is_duplicate:
                return handle_duplicate(submission, False, "SIFT + AI", author, title, date, utc, status, permalink)
            
            if context == "modqueue" and not submission.approved:
                submission.mod.approve()
                print("Original submission approved: ", submission.url)
            
            if hash_value not in image_hashes:
                image_hashes[hash_value] = (submission.id, submission.created_utc)
                sift_descriptors[submission.id] = descriptors
                ai_features[submission.id] = new_features
            
            return False
            
        except Exception as e:
            handle_exception(e)
            return False

    def check_removed_original_posts():
        """Monitor for immediate removal detection using dual approach"""
        processed_log_items = set()
        last_checked = {}
        
        def monitor_mod_log():
            while True:
                try:
                    for log_entry in subreddit.mod.stream.log(action='removelink', skip_existing=True):
                        if log_entry.id in processed_log_items:
                            continue
                        
                        processed_log_items.add(log_entry.id)
                        removed_submission_id = log_entry.target_fullname.replace('t3_', '')
                        
                        hash_to_process = None
                        for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                            if submission_id == removed_submission_id:
                                hash_to_process = hash_value
                                break
                        
                        if hash_to_process and hash_to_process not in moderator_removed_hashes:
                            moderator_removed_hashes.add(hash_to_process)
                            print(f"[MOD REMOVE] Submission {removed_submission_id} removed by moderator. Hash kept for future duplicate detection.")
                        
                        if len(processed_log_items) > 1000:
                            processed_log_items.clear()
                    
                except Exception as e:
                    print(f"Error in mod log monitor: {e}")
                    time.sleep(5)
        
        threading.Thread(target=monitor_mod_log, daemon=True).start()
        
        while True:
            try:
                current_check_time = time.time()
                checked_this_cycle = 0
                
                recent_submissions = []
                medium_submissions = []
                old_submissions = []
                
                for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                    if hash_value in moderator_removed_hashes:
                        continue
                    
                    age = current_check_time - creation_time
                    last_check = last_checked.get(submission_id, 0)
                    
                    if age < 3600:
                        check_interval = 30
                        if current_check_time - last_check >= check_interval:
                            recent_submissions.append((hash_value, submission_id))
                    elif age < 86400:
                        check_interval = 300
                        if current_check_time - last_check >= check_interval:
                            medium_submissions.append((hash_value, submission_id))
                    else:
                        check_interval = 1800
                        if current_check_time - last_check >= check_interval:
                            old_submissions.append((hash_value, submission_id))
                
                all_to_check = recent_submissions + medium_submissions[:20] + old_submissions[:10]
                
                for hash_value, submission_id in all_to_check:
                    try:
                        original_submission = reddit.submission(id=submission_id)
                        original_author = original_submission.author
                        
                        if original_author is None:
                            if hash_value in image_hashes:
                                del image_hashes[hash_value]
                            if submission_id in sift_descriptors:
                                del sift_descriptors[submission_id]
                            if submission_id in ai_features:
                                del ai_features[submission_id]
                            if submission_id in last_checked:
                                del last_checked[submission_id]
                            print(f"[USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                        else:
                            last_checked[submission_id] = current_check_time
                        
                        checked_this_cycle += 1
                        
                        if checked_this_cycle >= 10:
                            time.sleep(60)
                            checked_this_cycle = 0
                        
                    except Exception as e:
                        print(f"Error checking submission {submission_id}: {e}")
                        last_checked[submission_id] = current_check_time
                
            except Exception as e:
                handle_exception(e)
            
            time.sleep(60)
    threading.Thread(target=check_removed_original_posts, daemon=True).start()

    # --- Initial scan ---
    try:
        for submission in subreddit.new(limit=100):
            if isinstance(submission, praw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                print("Indexing submission (initial scan): ", submission.url)
                try:
                    img, hash_value, descriptors, features = load_and_process_image(submission.url)
                    if hash_value not in image_hashes:
                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                        sift_descriptors[submission.id] = descriptors
                        ai_features[submission.id] = features
                except Exception as e:
                    handle_exception(e)
    except Exception as e:
        handle_exception(e)

    # --- Mod Queue worker ---
    def modqueue_worker():
        nonlocal image_hashes, sift_descriptors, moderator_removed_hashes, processed_modqueue_submissions, ai_features
        while True:
            try:
                modqueue_submissions = subreddit.mod.modqueue(only='submission', limit=None)
                modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                for submission in modqueue_submissions:
                    if not isinstance(submission, praw.models.Submission):
                        continue
                    
                    print("Scanning Mod Queue: ", submission.url)
                    
                    if submission.num_reports > 0:
                        print("Skipping reported image: ", submission.url)
                        image_hashes = {k: v for k, v in image_hashes.items() if v[0] != submission.id}
                        sift_descriptors.pop(submission.id, None)
                        ai_features.pop(submission.id, None)
                        continue
                    
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        is_duplicate = process_submission_for_duplicates(submission, context="modqueue")
                        processed_modqueue_submissions.add(submission.id)

            except Exception as e:
                handle_exception(e)
            time.sleep(15)

    threading.Thread(target=modqueue_worker, daemon=True).start()

    # --- Stream new submissions ---
    while True:
        try:
            for submission in subreddit.stream.submissions(skip_existing=True):
                if submission.created_utc > current_time and isinstance(submission, praw.models.Submission):
                    if submission.id in processed_modqueue_submissions:
                        continue

                    print("Scanning new image/post: ", submission.url)
                    
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        process_submission_for_duplicates(submission, context="stream")

            current_time = int(time.time())
        except Exception as e:
            handle_exception(e)
        time.sleep(20)
