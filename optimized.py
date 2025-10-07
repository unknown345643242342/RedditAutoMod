def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    image_hashes = {}
    orb_descriptors = {}
    moderator_removed_hashes = set()
    processed_modqueue_submissions = set()
    approved_by_moderator = set()
    ai_features = {}
    image_text = {}
    current_time = int(time.time())

    # --- Tiny AI similarity model ---
    device = "cpu"
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.eval()
    resnet_model.to(device)
    resnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # OPTIMIZATION 1: Reuse ORB detector instead of creating new one each time
    orb_detector = cv2.ORB_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # OPTIMIZATION 2: Pre-compile regex for text cleaning (if needed later)
    import re
    whitespace_pattern = re.compile(r'\s+')

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

    def has_significant_text(img):
        """Enhanced detection for social media screenshots"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check if it's a dark mode screenshot
            mean_brightness = np.mean(gray)
            is_dark_mode = mean_brightness < 100
            
            # For dark mode, check for bright text
            if is_dark_mode:
                bright_pixels = np.sum(gray > 180) / gray.size
                if bright_pixels > 0.01:
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
                (edge_density > 0.03) or
                (horizontal_ratio > 0.001) or
                is_dark_mode
            )
            
            return has_text
        except Exception as e:
            print(f"Text detection error: {e}")
            return True

    def extract_text_from_image(img):
        """Enhanced text extraction optimized for social media screenshots"""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            all_texts = []
            
            # OPTIMIZATION 4: Try fastest strategy first, skip others if good result
            # Strategy 1: Direct extraction with PSM 11 (fastest and often best for screenshots)
            text = pytesseract.image_to_string(img_pil, config='--psm 11').strip()
            if text and len(text) > 50:  # If we got substantial text, add it
                all_texts.append(text)
            
            # Strategy 2: Try PSM 6 and 3 only if PSM 11 didn't get much
            if len(text) < 50:
                for psm in [6, 3]:
                    text = pytesseract.image_to_string(img_pil, config=f'--psm {psm}').strip()
                    if text:
                        all_texts.append(text)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            # Strategy 3: Dark mode optimization (only if dark)
            if mean_brightness < 100:
                inverted = cv2.bitwise_not(gray)
                text = pytesseract.image_to_string(Image.fromarray(inverted), config='--psm 11').strip()
                if text:
                    all_texts.append(text)
                
                # OPTIMIZATION 5: Skip binary threshold if inversion already got good text
                if len(text) < 30:
                    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                    text = pytesseract.image_to_string(Image.fromarray(thresh), config='--psm 11').strip()
                    if text:
                        all_texts.append(text)
            
            # Strategy 4: Upscale only if needed (small images)
            h, w = img.shape[:2]
            if h < 800 or w < 800:
                scale = 2
                upscaled = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)
                upscaled_pil = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))
                text = pytesseract.image_to_string(upscaled_pil, config='--psm 11').strip()
                if text:
                    all_texts.append(text)
            
            # OPTIMIZATION 6: Skip contrast enhancement if we already have good text
            if sum(len(t) for t in all_texts) < 100:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB))
                text = pytesseract.image_to_string(enhanced_pil, config='--psm 11').strip()
                if text:
                    all_texts.append(text)
            
            # OPTIMIZATION 7: Skip adaptive threshold if we have enough text
            if sum(len(t) for t in all_texts) < 100:
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 11, 2)
                text = pytesseract.image_to_string(Image.fromarray(adaptive), config='--psm 11').strip()
                if text:
                    all_texts.append(text)
            
            # OPTIMIZATION 8: Use regex for faster whitespace normalization
            unique_texts = []
            seen_texts = set()
            for text in all_texts:
                text_normalized = whitespace_pattern.sub(' ', text.lower()).strip()
                if text_normalized and text_normalized not in seen_texts:
                    unique_texts.append(text_normalized)
                    seen_texts.add(text_normalized)
            
            cleaned_text = ' '.join(unique_texts)
            return cleaned_text if len(cleaned_text) > 10 else ""
            
        except Exception as e:
            print(f"Text extraction error: {e}")
            return ""

    def text_similarity(text1, text2):
        """Calculate similarity between two text strings"""
        if not text1 or not text2:
            return 0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
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
        
        descriptors = get_orb_descriptors_conditional(img)
        features = get_ai_features(img)
        
        if has_significant_text(img):
            text = extract_text_from_image(img)
            print("Text detected and extracted")
        else:
            text = ""
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
        if submission_id in image_text:
            return image_text[submission_id]
        
        old_submission = reddit.submission(id=submission_id)
        old_image_data = requests.get(old_submission.url, timeout=10).content
        old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if has_significant_text(old_img):
            old_text = extract_text_from_image(old_img)
        else:
            old_text = ""
        image_text[submission_id] = old_text
        return old_text

    def calculate_ai_similarity(features1, features2):
        """Calculate AI similarity score between two feature vectors"""
        if features1 is not None and features2 is not None:
            return (features1 @ features2.T).item()
        return 0

    def check_hash_duplicate(submission, hash_value, new_features, new_text):
        """Check if submission is a hash-based duplicate"""
        if hash_value not in image_hashes:
            return False, None, None, None, None, None, None
        
        original_id, original_time = image_hashes[hash_value]
        
        if submission.id == original_id or submission.created_utc <= original_time:
            return False, None, None, None, None, None, None
        
        original_submission = reddit.submission(id=original_id)
        original_features = get_cached_ai_features(original_submission.id)
        original_text = get_cached_text(original_submission.id)
        
        ai_score = calculate_ai_similarity(new_features, original_features)
        text_score = text_similarity(new_text, original_text)
        
        print(f"Hash match detected. AI similarity: {ai_score:.2f}, Text similarity: {text_score:.2f}")
        
        if ai_score > 0.70 or text_score > 0.75:
            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            original_status = "Removed by Moderator" if hash_value in moderator_removed_hashes else "Active"
            return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def check_orb_duplicate(submission, descriptors, new_features, new_text):
        """Check if submission is an ORB-based duplicate"""
        for old_id, old_desc in orb_descriptors.items():
            sim = orb_similarity(descriptors, old_desc)
            
            if sim > 0.30:
                old_features = get_cached_ai_features(old_id)
                old_text = get_cached_text(old_id)
                
                ai_score = calculate_ai_similarity(new_features, old_features)
                text_score = text_similarity(new_text, old_text)
                
                print(f"ORB match detected. AI similarity: {ai_score:.2f}, Text similarity: {text_score:.2f}")
                
                if ai_score > 0.70 or text_score > 0.75:
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
            img, hash_value, descriptors, new_features, new_text = load_and_process_image(submission.url)
            ai_features[submission.id] = new_features
            image_text[submission.id] = new_text
            
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
            
            if context == "modqueue" and not submission.approved:
                submission.mod.approve()
                print("Original submission approved: ", submission.url)
            
            if hash_value not in image_hashes:
                image_hashes[hash_value] = (submission.id, submission.created_utc)
                orb_descriptors[submission.id] = descriptors
                ai_features[submission.id] = new_features
                image_text[submission.id] = new_text
            
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
                            if submission_id in image_text:
                                del image_text[submission_id]
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
                    img, hash_value, descriptors, features, text = load_and_process_image(submission.url)
                    if hash_value not in image_hashes:
                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                        orb_descriptors[submission.id] = descriptors
                        ai_features[submission.id] = features
                        image_text[submission.id] = text
                except Exception as e:
                    handle_exception(e)
    except Exception as e:
        handle_exception(e)

    # --- Mod Queue worker ---
    def modqueue_worker():
        nonlocal image_hashes, orb_descriptors, moderator_removed_hashes, processed_modqueue_submissions, ai_features, image_text
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
                        image_text.pop(submission.id, None)
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
