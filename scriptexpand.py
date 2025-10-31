def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    image_hashes = {}
    orb_descriptors = {}
    moderator_removed_hashes = set()
    processed_modqueue_submissions = set()
    approved_by_moderator = set()
    ai_features = {}
    image_text_raw = {}  # Store raw normalized text for fuzzy matching
    current_time = int(time.time())

    # --- Tiny AI similarity model ---
    device = "cpu"
    efficientnet_model = models.efficientnet_b0(pretrained=True)
    efficientnet_model.eval()
    efficientnet_model.to(device)
    efficientnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # OPTIMIZATION 1: Reuse ORB detector instead of creating new one each time
    orb_detector = cv2.ORB_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # OPTIMIZATION 2: Pre-compile regex for text cleaning (if needed later)
    import re
    import hashlib
    whitespace_pattern = re.compile(r'\s+')

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

    def has_significant_text(img):
        """Enhanced detection for social media screenshots"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check if it's a dark mode screenshot
            mean_brightness = np.mean(gray)
            is_dark_mode = mean_brightness < 100
            
            # For dark mode, check for bright text with multiple thresholds
            if is_dark_mode:
                # Check multiple brightness thresholds
                bright_pixels_180 = np.sum(gray > 180) / gray.size
                bright_pixels_150 = np.sum(gray > 150) / gray.size
                bright_pixels_120 = np.sum(gray > 120) / gray.size
                
                if bright_pixels_180 > 0.005 or bright_pixels_150 > 0.015 or bright_pixels_120 > 0.03:
                    return True
            
            # Check for light mode with dark text
            is_light_mode = mean_brightness > 200
            if is_light_mode:
                dark_pixels = np.sum(gray < 100) / gray.size
                if dark_pixels > 0.01:
                    return True
            
            # Original checks
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # OPTIMIZATION 3: Early return if edge density is high enough
            if edge_density > 0.08:
                return True
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detected_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            horizontal_ratio = np.sum(detected_lines > 0) / detected_lines.size
            
            has_text = (
                (edge_density > 0.02) or
                (horizontal_ratio > 0.0008) or
                is_dark_mode
            )
            
            return has_text
        except Exception as e:
            print(f"Text detection error: {e}")
            return True

    def extract_text_from_image(img):
        """Single strategy OCR optimized for social media screenshots"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Invert for dark mode (white text → black text)
            inverted = cv2.bitwise_not(gray)
            
            # PSM 6: Assume uniform block of text (best for social media posts)
            text = pytesseract.image_to_string(Image.fromarray(inverted), config='--psm 6').strip()
            
            return text if len(text) > 10 else ""
            
        except Exception as e:
            print(f"Text extraction error: {e}")
            return ""

    # NEW: Fuzzy text similarity for when hashes don't match due to OCR variations
    def fuzzy_text_similarity(text1, text2):
        """Calculate similarity between two text strings for OCR variations"""
        if not text1 or not text2:
            return 0
        
        # Normalize both texts the same way we do for hashing
        norm1 = whitespace_pattern.sub(' ', text1.lower()).strip()
        norm1 = norm1.replace(':', '').replace('-', '').replace(',', '').replace('.', '')
        norm1 = whitespace_pattern.sub(' ', norm1).strip()
        
        norm2 = whitespace_pattern.sub(' ', text2.lower()).strip()
        norm2 = norm2.replace(':', '').replace('-', '').replace(',', '').replace('.', '')
        norm2 = whitespace_pattern.sub(' ', norm2).strip()
        
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0

    def is_problematic_image(img, white_threshold=0.7, text_threshold=0.05):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_ratio = np.mean(gray > 240)
        
        # OPTIMIZATION 9: Early return if white ratio is high enough
        if white_ratio > white_threshold:
            return True
        
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.mean(edges > 0)
        return edge_ratio > text_threshold

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
        # OPTIMIZATION 10: Use pre-created ORB detector
        kp, des = orb_detector.detectAndCompute(processed_img, None)
        return des

    def orb_similarity(desc1, desc2):
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
        """Load image from URL and compute hash, descriptors, AI features, and text"""
        # OPTIMIZATION 12: Set timeout for requests to prevent hanging
        image_data = requests.get(url, timeout=10).content
        img = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        # OPTIMIZATION 13: Compute grayscale once and reuse
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hash_value = str(imagehash.phash(Image.fromarray(gray)))
        print(f"Generated hash: {hash_value}")
        
        descriptors = get_orb_descriptors_conditional(img)
        features = get_ai_features(img)
        
        # Extract text for fuzzy matching
        text = extract_text_from_image(img)
        
        if text:
            print(f"Text extracted: {len(text)} chars")
            print(f"  First 100 chars: {text[:100]}...")
        else:
            print(f"No significant text extracted")
            
        return img, hash_value, descriptors, features, text

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

    def get_cached_text(submission_id):
        """Get text from cache or extract it"""
        if submission_id in image_text_raw:
            return image_text_raw[submission_id]
        
        old_submission = reddit.submission(id=submission_id)
        old_image_data = requests.get(old_submission.url, timeout=10).content
        old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # Extract text
        old_text = extract_text_from_image(old_img)
        
        image_text_raw[submission_id] = old_text if old_text else ""
        return old_text if old_text else ""

    def calculate_ai_similarity(features1, features2):
        """Calculate AI similarity score between two feature vectors"""
        if features1 is not None and features2 is not None:
            return (features1 @ features2.T).item()
        return 0

    def check_hash_duplicate(submission, hash_value, new_features, new_text):
        """Check if submission is a hash-based duplicate"""
        # Check for exact match or Hamming distance <= 3
        matched_hash = None
        for stored_hash in image_hashes.keys():
            if hash_value == stored_hash or (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= 3:
                matched_hash = stored_hash
                break
        
        if matched_hash is None:
            return False, None, None, None, None, None, None
        
        original_id, original_time = image_hashes[matched_hash]
        
        if submission.id == original_id or submission.created_utc <= original_time:
            return False, None, None, None, None, None, None
        
        original_submission = reddit.submission(id=original_id)
        original_features = get_cached_ai_features(original_submission.id)
        original_text = get_cached_text(original_submission.id)
        
        ai_score = calculate_ai_similarity(new_features, original_features)
        
        # Fuzzy text comparison
        fuzzy_score = 0
        if new_text and original_text:
            fuzzy_score = fuzzy_text_similarity(new_text, original_text)
        
        # DEBUG: Show details
        print(f"Hash match detected. AI similarity: {ai_score:.2f}, Fuzzy: {fuzzy_score:.2f}")
        
        # AGGRESSIVE: Hash match is strong evidence
        # Accept if: AI > 0.50 OR fuzzy text > 0.85
        if ai_score > 0.50 or fuzzy_score > 0.80:
            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            original_status = "Removed by Moderator" if matched_hash in moderator_removed_hashes else "Active"
            return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def check_orb_duplicate(submission, descriptors, new_features, new_text):
        """Check if submission is an ORB-based duplicate"""
        for old_id, old_desc in orb_descriptors.items():
            sim = orb_similarity(descriptors, old_desc)
            
            if sim > 0.50:  # Lowered from 0.30 for screenshots with different crops
                old_features = get_cached_ai_features(old_id)
                old_text = get_cached_text(old_id)
                
                ai_score = calculate_ai_similarity(new_features, old_features)
                
                # Fuzzy text comparison
                fuzzy_score = 0
                if new_text and old_text:
                    fuzzy_score = fuzzy_text_similarity(new_text, old_text)
                
                # For ORB matches, be more aggressive with screenshots
                if ai_score > 0.75 or fuzzy_score > 0.80:
                    # ONLY PRINT WHEN DUPLICATE IS CONFIRMED
                    print(f"ORB duplicate found! AI similarity: {ai_score:.2f}, Fuzzy: {fuzzy_score:.2f}")
                    
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

    def check_text_only_duplicate(submission, new_text, new_features):
        """Check for text-based duplicates even without hash/ORB match"""
        if not new_text or len(new_text) < 50:
            return False, None, None, None, None, None, None
        
        # Check against all stored texts
        for old_id, old_text in image_text_raw.items():
            if not old_text or old_id == submission.id:
                continue
            
            # Fuzzy match
            fuzzy_score = fuzzy_text_similarity(new_text, old_text)
            
            # If text is very similar (0.85+), check AI to confirm
            if fuzzy_score > 0.80:
                old_features = get_cached_ai_features(old_id)
                ai_score = calculate_ai_similarity(new_features, old_features)
                
                # Require reasonable AI similarity to avoid false positives
                if ai_score > 0.70:
                    print(f"Text-only duplicate found! Fuzzy: {fuzzy_score:.2f}, AI: {ai_score:.2f}")
                    
                    original_submission = reddit.submission(id=old_id)
                    original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    old_hash = next((h for h, v in image_hashes.items() if v[0] == old_id), None)
                    original_status = "Removed by Moderator" if old_hash and old_hash in moderator_removed_hashes else "Active"
                    
                    return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def process_submission_for_duplicates(submission, context="stream"):
        """Main duplicate detection logic - works for both mod queue and stream"""
        try:
            img, hash_value, descriptors, new_features, new_text = load_and_process_image(submission.url)
            ai_features[submission.id] = new_features
            image_text_raw[submission.id] = new_text if new_text else ""
            
            if handle_moderator_removed_repost(submission, hash_value):
                return True
            
            is_duplicate, author, title, date, utc, status, permalink = check_hash_duplicate(
                submission, hash_value, new_features, new_text
            )
            if is_duplicate:
                return handle_duplicate(submission, True, "hash + AI/Text", author, title, date, utc, status, permalink)
            
            is_duplicate, author, title, date, utc, status, permalink = check_orb_duplicate(
                submission, descriptors, new_features, new_text
            )
            if is_duplicate:
                return handle_duplicate(submission, False, "ORB + AI/Text", author, title, date, utc, status, permalink)
            
            # NEW: Check text similarity even if hash/ORB didn't match
            is_duplicate, author, title, date, utc, status, permalink = check_text_only_duplicate(
                submission, new_text, new_features
            )
            if is_duplicate:
                return handle_duplicate(submission, False, "Text + AI", author, title, date, utc, status, permalink)
            
            # Store as new original BEFORE approving (ensures database is updated first)
            if hash_value not in image_hashes:
                image_hashes[hash_value] = (submission.id, submission.created_utc)
                orb_descriptors[submission.id] = descriptors
                ai_features[submission.id] = new_features
                image_text_raw[submission.id] = new_text if new_text else ""
                print(f"Stored new original: {submission.url}")
            
            # Only approve if no duplicate was found AND we stored it as new original
            if context == "modqueue" and not submission.approved:
                submission.mod.approve()
                print("Original submission approved: ", submission.url)
            
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
                            if submission_id in orb_descriptors:
                                del orb_descriptors[submission_id]
                            if submission_id in ai_features:
                                del ai_features[submission_id]
                            if submission_id in image_text_raw:
                                del image_text_raw[submission_id]
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
        for submission in subreddit.new(limit=300):
            if isinstance(submission, praw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                print("Indexing submission (initial scan): ", submission.url)
                try:
                    img, hash_value, descriptors, features, text = load_and_process_image(submission.url)
                    if hash_value not in image_hashes:
                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                        orb_descriptors[submission.id] = descriptors
                        ai_features[submission.id] = features
                        image_text_raw[submission.id] = text if text else ""
                except Exception as e:
                    handle_exception(e)
                except Exception as e:
                    handle_exception(e)
    except Exception as e:
        handle_exception(e)

    # --- Mod Queue worker ---
    def modqueue_worker():
        nonlocal image_hashes, orb_descriptors, moderator_removed_hashes, processed_modqueue_submissions, ai_features, image_text_raw
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
                        orb_descriptors.pop(submission.id, None)
                        ai_features.pop(submission.id, None)
                        image_text_raw.pop(submission.id, None)
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
