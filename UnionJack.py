def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    image_hashes = {}
    orb_descriptors = {}
    moderator_removed_hashes = set()
    processed_modqueue_submissions = set()
    approved_by_moderator = set()
    ai_features = {}
    image_text = {}  # NEW: Store extracted text
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
        """Enhanced text detection with better dark background handling"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Check 1: Edge density (text has lots of edges)
            edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for better detection
            edge_density = np.sum(edges > 0) / edges.size
            
            # Check 2: Variance in local regions (text creates high local variance)
            kernel_size = 5
            mean = cv2.blur(gray, (kernel_size, kernel_size))
            sqr_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
            variance = sqr_mean - mean**2
            high_var_ratio = np.sum(variance > 100) / variance.size
            
            # Check 3: Stroke Width Transform approximation
            # Text has consistent stroke widths
            dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 3)
            stroke_mask = (dist_transform > 0) & (dist_transform < 20)
            stroke_ratio = np.sum(stroke_mask) / stroke_mask.size
            
            # Check 4: Horizontal and vertical line detection
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            line_ratio = (np.sum(h_lines > 0) + np.sum(v_lines > 0)) / (h * w)
            
            # Check 5: Multiple threshold analysis (catches both light and dark text)
            thresholds = []
            for thresh_val in [0, 127, 200]:
                if thresh_val == 0:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                text_like = sum(1 for c in contours if 50 < cv2.contourArea(c) < 10000)
                thresholds.append(text_like)
            
            max_text_contours = max(thresholds)
            
            # Decision logic: Multiple pathways to detect text
            has_text = (
                (edge_density > 0.02 and high_var_ratio > 0.05) or
                (stroke_ratio > 0.01 and line_ratio > 0.0005) or
                (max_text_contours > 5) or
                (edge_density > 0.04) or
                (high_var_ratio > 0.15)
            )
            
            print(f"[TEXT DETECTION] edge={edge_density:.4f}, var={high_var_ratio:.4f}, "
                  f"stroke={stroke_ratio:.4f}, line={line_ratio:.6f}, contours={max_text_contours} -> {has_text}")
            
            return has_text
        except Exception as e:
            print(f"Text detection error: {e}")
            return True  # Changed: If detection fails, TRY OCR anyway

    def normalize_text(text):
        """Normalize text to handle OCR errors and improve matching"""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Common OCR error corrections
        corrections = {
            r'[|!l1]': 'i',  # Common misreads for 'i'
            r'[0o]': 'o',    # Zero vs O
            r'\s+': ' ',     # Multiple spaces to single space
            r'[^\w\s]': '',  # Remove punctuation
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Only return if meaningful length
        return text if len(text) > 8 else ""

    def extract_text_from_image(img):
        """Enhanced text extraction with better preprocessing for dark backgrounds and small text"""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            extracted_texts = []
            
            # Strategy 1: Direct extraction with optimized config
            text1 = pytesseract.image_to_string(img_pil, config='--psm 6 --oem 3').strip()
            extracted_texts.append(text1)
            
            # Strategy 2: Inverted (for white text on dark background)
            inverted = cv2.bitwise_not(gray)
            text2 = pytesseract.image_to_string(Image.fromarray(inverted), config='--psm 6 --oem 3').strip()
            extracted_texts.append(text2)
            
            # Strategy 3: Adaptive threshold (handles varying lighting)
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
            text3 = pytesseract.image_to_string(Image.fromarray(adaptive), config='--psm 6 --oem 3').strip()
            extracted_texts.append(text3)
            
            # Strategy 4: Inverted adaptive threshold
            adaptive_inv = cv2.bitwise_not(adaptive)
            text4 = pytesseract.image_to_string(Image.fromarray(adaptive_inv), config='--psm 6 --oem 3').strip()
            extracted_texts.append(text4)
            
            # Strategy 5: Enhanced contrast with CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            text5 = pytesseract.image_to_string(Image.fromarray(enhanced), config='--psm 6 --oem 3').strip()
            extracted_texts.append(text5)
            
            # Strategy 6: Upscale for small text (2x)
            scale_factor = 2
            upscaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            upscaled_enhanced = clahe.apply(upscaled)
            text6 = pytesseract.image_to_string(Image.fromarray(upscaled_enhanced), config='--psm 6 --oem 3').strip()
            extracted_texts.append(text6)
            
            # Strategy 7: Binary threshold with Otsu (automatic threshold)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text7 = pytesseract.image_to_string(Image.fromarray(otsu), config='--psm 6 --oem 3').strip()
            extracted_texts.append(text7)
            
            # Strategy 8: Inverted Otsu
            otsu_inv = cv2.bitwise_not(otsu)
            text8 = pytesseract.image_to_string(Image.fromarray(otsu_inv), config='--psm 6 --oem 3').strip()
            extracted_texts.append(text8)
            
            # Combine and normalize all extracted text
            all_text = ' '.join(extracted_texts)
            final_text = normalize_text(all_text)
            
            if final_text:
                print(f"[OCR EXTRACTED] {len(final_text)} chars: {final_text[:150]}...")
            else:
                print("[OCR EXTRACTED] No significant text found")
            
            return final_text
        except Exception as e:
            print(f"Text extraction error: {e}")
            return ""

    def text_similarity(text1, text2):
        """Enhanced text similarity using multiple methods"""
        if not text1 or not text2:
            return 0
        
        # Normalize both texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if not text1 or not text2:
            return 0
        
        # Method 1: Jaccard similarity (word overlap)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0
        
        # Method 2: Character n-gram similarity (handles minor OCR errors)
        def get_ngrams(text, n=3):
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        
        if ngrams1 and ngrams2:
            ngram_intersection = len(ngrams1.intersection(ngrams2))
            ngram_union = len(ngrams1.union(ngrams2))
            ngram_score = ngram_intersection / ngram_union if ngram_union > 0 else 0
        else:
            ngram_score = 0
        
        # Method 3: Sequence matching (fuzzy string matching)
        from difflib import SequenceMatcher
        sequence_score = SequenceMatcher(None, text1, text2).ratio()
        
        # Method 4: Longest common substring ratio
        def longest_common_substring(s1, s2):
            m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
            longest = 0
            for x in range(1, 1 + len(s1)):
                for y in range(1, 1 + len(s2)):
                    if s1[x - 1] == s2[y - 1]:
                        m[x][y] = m[x - 1][y - 1] + 1
                        longest = max(longest, m[x][y])
            return longest
        
        lcs_length = longest_common_substring(text1, text2)
        lcs_score = lcs_length / max(len(text1), len(text2)) if max(len(text1), len(text2)) > 0 else 0
        
        # Weighted combination of all methods
        final_score = (
            jaccard * 0.30 +
            ngram_score * 0.25 +
            sequence_score * 0.25 +
            lcs_score * 0.20
        )
        
        print(f"[TEXT SIMILARITY] jaccard={jaccard:.3f}, ngram={ngram_score:.3f}, "
              f"seq={sequence_score:.3f}, lcs={lcs_score:.3f} -> final={final_score:.3f}")
        
        return final_score

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

    # --- NEW: Consolidated helper functions ---
    def load_and_process_image(url):
        """Load image from URL and compute hash, descriptors, AI features, and text"""
        image_data = requests.get(url).content
        img = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
        descriptors = get_orb_descriptors_conditional(img)
        features = get_ai_features(img)
        # NEW: Only extract text if image likely contains text
        if has_significant_text(img):
            text = extract_text_from_image(img)
        else:
            text = ""
        return img, hash_value, descriptors, features, text

    def get_cached_ai_features(submission_id):
        """Get AI features from cache or compute them"""
        if submission_id in ai_features:
            return ai_features[submission_id]
        
        old_submission = reddit.submission(id=submission_id)
        old_image_data = requests.get(old_submission.url).content
        old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        old_features = get_ai_features(old_img)
        ai_features[submission_id] = old_features
        return old_features

    def get_cached_text(submission_id):
        """NEW: Get text from cache or extract it"""
        if submission_id in image_text:
            return image_text[submission_id]
        
        old_submission = reddit.submission(id=submission_id)
        old_image_data = requests.get(old_submission.url).content
        old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        # NEW: Only extract text if image likely contains text
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
            return False, None, None, None, None, None, None, None, None
        
        original_id, original_time = image_hashes[hash_value]
        
        # Skip if same submission or older
        if submission.id == original_id or submission.created_utc <= original_time:
            return False, None, None, None, None, None, None, None, None
        
        original_submission = reddit.submission(id=original_id)
        original_features = get_cached_ai_features(original_submission.id)
        original_text = get_cached_text(original_submission.id)  # NEW
        
        ai_score = calculate_ai_similarity(new_features, original_features)
        text_score = text_similarity(new_text, original_text)  # NEW
        
        print(f"Hash match detected. AI similarity: {ai_score:.2f}, Text similarity: {text_score:.2f}")
        
        # NEW: Accept if either AI or text similarity is high
        if ai_score > 0.70 or text_score > 0.60:
            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            original_status = "Removed by Moderator" if hash_value in moderator_removed_hashes else "Active"
            return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink, ai_score, text_score
        
        return False, None, None, None, None, None, None, None, None

    def check_orb_duplicate(submission, descriptors, new_features, new_text):
        """Check if submission is an ORB-based duplicate"""
        for old_id, old_desc in orb_descriptors.items():
            sim = orb_similarity(descriptors, old_desc)
            
            if sim > 0.30:
                old_features = get_cached_ai_features(old_id)
                old_text = get_cached_text(old_id)  # NEW
                
                ai_score = calculate_ai_similarity(new_features, old_features)
                text_score = text_similarity(new_text, old_text)  # NEW
                
                print(f"ORB match detected. AI similarity: {ai_score:.2f}, Text similarity: {text_score:.2f}")
                
                # NEW: Accept if either AI or text similarity is high
                if ai_score > 0.70 or text_score > 0.60:
                    original_submission = reddit.submission(id=old_id)
                    original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    old_hash = next((h for h, v in image_hashes.items() if v[0] == old_id), None)
                    original_status = "Removed by Moderator" if old_hash and old_hash in moderator_removed_hashes else "Active"
                    
                    return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink, ai_score, text_score
        
        return False, None, None, None, None, None, None, None, None

    def handle_duplicate(submission, is_hash_dup, detection_method, author, title, date, utc, status, permalink, ai_score=None, text_score=None):
        """Remove duplicate and post comment if not approved"""
        if not submission.approved:
            submission.mod.remove()
            post_comment(submission, author, title, date, utc, status, permalink)
            if ai_score is not None and text_score is not None:
                print(f"Duplicate removed by {detection_method}: {submission.url} (AI: {ai_score:.2f}, Text: {text_score:.2f})")
            else:
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
            image_text[submission.id] = new_text  # NEW: Store text
            
            # Check for moderator-removed reposts first
            if handle_moderator_removed_repost(submission, hash_value):
                return True
            
            # Check hash-based duplicates
            is_duplicate, author, title, date, utc, status, permalink, ai_score, text_score = check_hash_duplicate(
                submission, hash_value, new_features, new_text
            )
            if is_duplicate:
                return handle_duplicate(submission, True, "hash + AI/Text", author, title, date, utc, status, permalink, ai_score, text_score)
            
            # Check ORB-based duplicates
            is_duplicate, author, title, date, utc, status, permalink, ai_score, text_score = check_orb_duplicate(
                submission, descriptors, new_features, new_text
            )
            if is_duplicate:
                return handle_duplicate(submission, False, "ORB + AI/Text", author, title, date, utc, status, permalink, ai_score, text_score)
            
            # Not a duplicate - approve if in mod queue and store data
            if context == "modqueue" and not submission.approved:
                submission.mod.approve()
                print("Original submission approved: ", submission.url)
            
            if hash_value not in image_hashes:
                image_hashes[hash_value] = (submission.id, submission.created_utc)
                orb_descriptors[submission.id] = descriptors
                ai_features[submission.id] = new_features
                image_text[submission.id] = new_text  # NEW: Store text
            
            return False
            
        except Exception as e:
            handle_exception(e)
            return False

    def check_removed_original_posts():
        """Monitor for immediate removal detection using dual approach"""
        processed_log_items = set()
        last_checked = {}
        
        # Thread for monitoring mod log (immediate mod removals)
        def monitor_mod_log():
            while True:
                try:
                    for log_entry in subreddit.mod.stream.log(action='removelink', skip_existing=True):
                        if log_entry.id in processed_log_items:
                            continue
                        
                        processed_log_items.add(log_entry.id)
                        removed_submission_id = log_entry.target_fullname.replace('t3_', '')
                        
                        # Find the hash for this submission
                        hash_to_process = None
                        for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                            if submission_id == removed_submission_id:
                                hash_to_process = hash_value
                                break
                        
                        if hash_to_process and hash_to_process not in moderator_removed_hashes:
                            moderator_removed_hashes.add(hash_to_process)
                            print(f"[MOD REMOVE] Submission {removed_submission_id} removed by moderator. Hash kept for future duplicate detection.")
                        
                        # Limit processed log items
                        if len(processed_log_items) > 1000:
                            processed_log_items.clear()
                    
                except Exception as e:
                    print(f"Error in mod log monitor: {e}")
                    time.sleep(5)
        
        # Start mod log monitor in separate thread
        threading.Thread(target=monitor_mod_log, daemon=True).start()
        
        # Main thread: prioritized check for user deletions
        while True:
            try:
                current_check_time = time.time()
                checked_this_cycle = 0
                
                # Separate submissions into priority tiers based on age
                recent_submissions = []  # < 1 hour old
                medium_submissions = []  # 1-24 hours old
                old_submissions = []     # > 24 hours old
                
                for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                    # Skip if already marked as mod-removed
                    if hash_value in moderator_removed_hashes:
                        continue
                    
                    age = current_check_time - creation_time
                    last_check = last_checked.get(submission_id, 0)
                    
                    # Determine check interval based on age
                    if age < 3600:  # Less than 1 hour old
                        check_interval = 30  # Check every 30 seconds
                        if current_check_time - last_check >= check_interval:
                            recent_submissions.append((hash_value, submission_id))
                    elif age < 86400:  # 1-24 hours old
                        check_interval = 300  # Check every 5 minutes
                        if current_check_time - last_check >= check_interval:
                            medium_submissions.append((hash_value, submission_id))
                    else:  # Older than 24 hours
                        check_interval = 1800  # Check every 30 minutes
                        if current_check_time - last_check >= check_interval:
                            old_submissions.append((hash_value, submission_id))
                
                # Process in priority order: recent first, then medium, then old
                all_to_check = recent_submissions + medium_submissions[:20] + old_submissions[:10]
                
                for hash_value, submission_id in all_to_check:
                    try:
                        original_submission = reddit.submission(id=submission_id)
                        original_author = original_submission.author
                        
                        # Check if user deleted their post
                        if original_author is None:
                            if hash_value in image_hashes:
                                del image_hashes[hash_value]
                            if submission_id in orb_descriptors:
                                del orb_descriptors[submission_id]
                            if submission_id in ai_features:
                                del ai_features[submission_id]
                            if submission_id in image_text:  # NEW
                                del image_text[submission_id]
                            if submission_id in last_checked:
                                del last_checked[submission_id]
                            print(f"[USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                        else:
                            last_checked[submission_id] = current_check_time
                        
                        checked_this_cycle += 1
                        
                        # Rate limiting: check 10 at a time, then pause
                        if checked_this_cycle >= 10:
                            time.sleep(2)
                            checked_this_cycle = 0
                        
                    except Exception as e:
                        print(f"Error checking submission {submission_id}: {e}")
                        last_checked[submission_id] = current_check_time
                
            except Exception as e:
                handle_exception(e)
            
            time.sleep(5)  # Short 5-second pause between cycles
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
                        image_text[submission.id] = text  # NEW
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
                        image_text.pop(submission.id, None)  # NEW
                        continue
                    
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        is_duplicate = process_submission_for_duplicates(submission, context="modqueue")
                        # Always add to processed set so it doesn't get reprocessed in stream
                        processed_modqueue_submissions.add(submission.id)

            except Exception as e:
                handle_exception(e)
            time.sleep(2)

    threading.Thread(target=modqueue_worker, daemon=True).start()

    # --- Stream new submissions ---
    while True:
        try:
            for submission in subreddit.stream.submissions(skip_existing=True):
                if submission.created_utc > current_time and isinstance(submission, praw.models.Submission):
                    # Skip originals already approved in mod queue
                    if submission.id in processed_modqueue_submissions:
                        continue

                    print("Scanning new image/post: ", submission.url)
                    
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        process_submission_for_duplicates(submission, context="stream")

            current_time = int(time.time())
        except Exception as e:
            handle_exception(e)
