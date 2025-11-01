def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    
    # Global dictionary to store per-subreddit data
    subreddit_data = {}
    subreddit_data_lock = threading.Lock()
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        'hash_distance': 3,
        'hash_ai_similarity': 0.50,
        'orb_similarity': 0.50,
        'orb_ai_similarity': 0.75
    }
    
    # Shared AI models (initialized once)
    device = "cpu"
    efficientnet_model = models.efficientnet_b0(pretrained=True)
    efficientnet_model.eval()
    efficientnet_model.to(device)
    efficientnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    orb_detector = cv2.ORB_create()
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def load_thresholds_from_wiki(subreddit):
        """Load thresholds from subreddit wiki page"""
        try:
            wiki_page = subreddit.wiki['duplicatebot_config']
            config_text = wiki_page.content_md
            
            thresholds = DEFAULT_THRESHOLDS.copy()
            
            for line in config_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key in thresholds:
                        try:
                            thresholds[key] = float(value)
                        except ValueError:
                            print(f"Invalid value for {key}: {value}")
            
            return thresholds
        except Exception as e:
            print(f"Could not load config from wiki, using defaults: {e}")
            return DEFAULT_THRESHOLDS.copy()
    
    def save_thresholds_to_wiki(subreddit, thresholds):
        """Save thresholds to subreddit wiki page"""
        try:
            config_text = f"""# Duplicate Detection Bot Configuration

## Threshold Settings

hash_distance: {thresholds['hash_distance']}
hash_ai_similarity: {thresholds['hash_ai_similarity']}
orb_similarity: {thresholds['orb_similarity']}
orb_ai_similarity: {thresholds['orb_ai_similarity']}

## Threshold Descriptions

- **hash_distance** (0-64): Maximum difference between perceptual hashes to consider images similar. Lower = stricter. Default: 3
- **hash_ai_similarity** (0.0-1.0): Minimum AI similarity score for hash matches. Higher = stricter. Default: 0.50
- **orb_similarity** (0.0-1.0): Minimum ORB feature matching score. Higher = stricter. Default: 0.50
- **orb_ai_similarity** (0.0-1.0): Minimum AI similarity score for ORB matches. Higher = stricter. Default: 0.75

## How to Adjust Thresholds

Moderators can adjust these values by:
1. Editing this wiki page directly
2. Sending a modmail with: `!setthreshold <parameter> <value>`
3. Replying to any bot comment with: `!setthreshold <parameter> <value>`

Examples:
- `!setthreshold hash_ai_similarity 0.60` - Require 60% AI similarity for hash matches
- `!setthreshold orb_similarity 0.65` - Require 65% ORB feature matching
- `!showthresholds` - Display current threshold values
- `!resetthresholds` - Reset all thresholds to defaults

Changes take effect immediately.
"""
            
            subreddit.wiki.create('duplicatebot_config', config_text, reason='Bot configuration update')
            return True
        except Exception as e:
            # Try to edit if page already exists
            try:
                wiki_page = subreddit.wiki['duplicatebot_config']
                wiki_page.edit(config_text, reason='Bot configuration update')
                return True
            except Exception as e2:
                print(f"Could not save config to wiki: {e2}")
                return False
    
    def setup_subreddit(subreddit_name):
        """Initialize data structures for a specific subreddit (no new threads)"""
        print(f"\n=== Setting up bot for r/{subreddit_name} ===")
        
        subreddit = reddit.subreddit(subreddit_name)
        
        # Load thresholds from wiki or use defaults
        thresholds = load_thresholds_from_wiki(subreddit)
        
        # Initialize wiki page if it doesn't exist
        save_thresholds_to_wiki(subreddit, thresholds)
        
        # Create dedicated dictionaries for this subreddit
        data = {
            'subreddit': subreddit,
            'thresholds': thresholds,
            'image_hashes': {},
            'orb_descriptors': {},
            'moderator_removed_hashes': set(),
            'processed_modqueue_submissions': set(),
            'approved_by_moderator': set(),
            'ai_features': {},
            'current_time': int(time.time()),
            'processed_log_items': set(),
            'last_checked': {},
            'last_threshold_reload': time.time()
        }
        
        with subreddit_data_lock:
            subreddit_data[subreddit_name] = data
        
        print(f"[r/{subreddit_name}] Loaded thresholds: {thresholds}")
        
        # --- Helper functions for this subreddit ---
        def get_ai_features(img):
            try:
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_tensor = efficientnet_transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = efficientnet_model(img_tensor)
                    feat = feat / feat.norm(dim=1, keepdim=True)
                return feat
            except Exception as e:
                print(f"[r/{subreddit_name}] AI feature extraction error:", e)
                return None

        def is_problematic_image(img, white_threshold=0.7, text_threshold=0.05):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            white_ratio = np.mean(gray > 240)
            
            if white_threshold > white_threshold:
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
            kp, des = orb_detector.detectAndCompute(processed_img, None)
            return des

        def orb_similarity(desc1, desc2):
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                return 0
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
                    print(f"[r/{subreddit_name}] Duplicate removed and comment posted: ", submission.url)
                    return True
                except Exception as e:
                    handle_exception(e)
                    retries += 1
                    time.sleep(1)
            return False

        def load_and_process_image(url):
            """Load image from URL and compute hash, descriptors, AI features"""
            image_data = requests.get(url, timeout=10).content
            img = np.asarray(bytearray(image_data), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hash_value = str(imagehash.phash(Image.fromarray(gray)))
            print(f"[r/{subreddit_name}] Generated hash: {hash_value}")
            
            descriptors = get_orb_descriptors_conditional(img)
            features = get_ai_features(img)
                
            return img, hash_value, descriptors, features

        def get_cached_ai_features(submission_id):
            """Get AI features from cache or compute them"""
            if submission_id in data['ai_features']:
                return data['ai_features'][submission_id]
            
            old_submission = reddit.submission(id=submission_id)
            old_image_data = requests.get(old_submission.url, timeout=10).content
            old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
            old_features = get_ai_features(old_img)
            data['ai_features'][submission_id] = old_features
            return old_features

        def calculate_ai_similarity(features1, features2):
            """Calculate AI similarity score between two feature vectors"""
            if features1 is not None and features2 is not None:
                return (features1 @ features2.T).item()
            return 0

        def check_hash_duplicate(submission, hash_value, new_features):
            """Check if submission is a hash-based duplicate"""
            matched_hash = None
            hash_dist_threshold = int(data['thresholds']['hash_distance'])
            
            for stored_hash in data['image_hashes'].keys():
                if hash_value == stored_hash or (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= hash_dist_threshold:
                    matched_hash = stored_hash
                    break
            
            if matched_hash is None:
                return False, None, None, None, None, None, None
            
            original_id, original_time = data['image_hashes'][matched_hash]
            
            if submission.id == original_id or submission.created_utc <= original_time:
                return False, None, None, None, None, None, None
            
            original_submission = reddit.submission(id=original_id)
            original_features = get_cached_ai_features(original_submission.id)
            
            ai_score = calculate_ai_similarity(new_features, original_features)
            
            print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f}")
            
            if ai_score > data['thresholds']['hash_ai_similarity']:
                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                original_status = "Removed by Moderator" if matched_hash in data['moderator_removed_hashes'] else "Active"
                return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
            
            return False, None, None, None, None, None, None

        def check_orb_duplicate(submission, descriptors, new_features):
            """Check if submission is an ORB-based duplicate"""
            for old_id, old_desc in data['orb_descriptors'].items():
                sim = orb_similarity(descriptors, old_desc)
                
                if sim > data['thresholds']['orb_similarity']:
                    old_features = get_cached_ai_features(old_id)
                    
                    ai_score = calculate_ai_similarity(new_features, old_features)
                    
                    if ai_score > data['thresholds']['orb_ai_similarity']:
                        print(f"[r/{subreddit_name}] ORB duplicate found! AI similarity: {ai_score:.2f}")
                        
                        original_submission = reddit.submission(id=old_id)
                        original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                        old_hash = next((h for h, v in data['image_hashes'].items() if v[0] == old_id), None)
                        original_status = "Removed by Moderator" if old_hash and old_hash in data['moderator_removed_hashes'] else "Active"
                        
                        return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
            
            return False, None, None, None, None, None, None

        def handle_duplicate(submission, is_hash_dup, detection_method, author, title, date, utc, status, permalink):
            """Remove duplicate and post comment if not approved"""
            if not submission.approved:
                submission.mod.remove()
                post_comment(submission, author, title, date, utc, status, permalink)
                print(f"[r/{subreddit_name}] Duplicate removed by {detection_method}: {submission.url}")
            return True

        def handle_moderator_removed_repost(submission, hash_value):
            """Handle reposts of moderator-removed images"""
            if hash_value in data['moderator_removed_hashes'] and not submission.approved:
                submission.mod.remove()
                original_submission = reddit.submission(id=data['image_hashes'][hash_value][0])
                post_comment(
                    submission,
                    original_submission.author.name,
                    original_submission.title,
                    datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    original_submission.created_utc,
                    "Removed by Moderator",
                    original_submission.permalink
                )
                print(f"[r/{subreddit_name}] Repost of a moderator-removed image removed: ", submission.url)
                return True
            return False

        def process_submission_for_duplicates(submission, context="stream"):
            """Main duplicate detection logic - works for both mod queue and stream"""
            try:
                img, hash_value, descriptors, new_features = load_and_process_image(submission.url)
                data['ai_features'][submission.id] = new_features
                
                if handle_moderator_removed_repost(submission, hash_value):
                    return True
                
                is_duplicate, author, title, date, utc, status, permalink = check_hash_duplicate(
                    submission, hash_value, new_features
                )
                if is_duplicate:
                    return handle_duplicate(submission, True, "hash + AI", author, title, date, utc, status, permalink)
                
                is_duplicate, author, title, date, utc, status, permalink = check_orb_duplicate(
                    submission, descriptors, new_features
                )
                if is_duplicate:
                    return handle_duplicate(submission, False, "ORB + AI", author, title, date, utc, status, permalink)
                
                if hash_value not in data['image_hashes']:
                    data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                    data['orb_descriptors'][submission.id] = descriptors
                    data['ai_features'][submission.id] = new_features
                    print(f"[r/{subreddit_name}] Stored new original: {submission.url}")
                
                if context == "modqueue" and not submission.approved:
                    submission.mod.approve()
                    print(f"[r/{subreddit_name}] Original submission approved: ", submission.url)
                
                return False
                
            except Exception as e:
                handle_exception(e)
                return False

        # Store the process_submission function for this subreddit
        data['process_submission'] = process_submission_for_duplicates
        
        # --- Initial scan ---
        print(f"[r/{subreddit_name}] Starting initial scan...")
        try:
            for submission in subreddit.new(limit=300):
                if isinstance(submission, praw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    print(f"[r/{subreddit_name}] Indexing submission (initial scan): ", submission.url)
                    try:
                        img, hash_value, descriptors, features = load_and_process_image(submission.url)
                        if hash_value not in data['image_hashes']:
                            data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                            data['orb_descriptors'][submission.id] = descriptors
                            data['ai_features'][submission.id] = features
                    except Exception as e:
                        handle_exception(e)
        except Exception as e:
            handle_exception(e)
        
        print(f"[r/{subreddit_name}] Initial scan complete. Indexed {len(data['image_hashes'])} images.")
        print(f"[r/{subreddit_name}] Bot setup complete!\n")

    # --- SHARED WORKER THREADS (one of each for ALL subreddits) ---
    
    def shared_command_listener():
        """Listen for threshold adjustment commands from moderators"""
        processed_items = set()
        
        while True:
            try:
                with subreddit_data_lock:
                    subreddits = list(subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        subreddit = data['subreddit']
                        
                        # Check mentions and replies to bot comments
                        for mention in reddit.inbox.mentions(limit=25):
                            if mention.id in processed_items:
                                continue
                            
                            processed_items.add(mention.id)
                            
                            # Check if commenter is a moderator
                            try:
                                if not subreddit.moderator(mention.author)[0]:
                                    continue
                            except:
                                continue
                            
                            body = mention.body.strip().lower()
                            
                            # Show current thresholds
                            if '!showthresholds' in body:
                                response = f"""**Current Threshold Settings for r/{subreddit_name}:**

- hash_distance: {data['thresholds']['hash_distance']}
- hash_ai_similarity: {data['thresholds']['hash_ai_similarity']}
- orb_similarity: {data['thresholds']['orb_similarity']}
- orb_ai_similarity: {data['thresholds']['orb_ai_similarity']}

To adjust: `!setthreshold <parameter> <value>`
To reset: `!resetthresholds`
To edit directly: [View wiki config](https://reddit.com/r/{subreddit_name}/wiki/duplicatebot_config)"""
                                mention.reply(response)
                                print(f"[r/{subreddit_name}] Showed thresholds to {mention.author}")
                            
                            # Reset thresholds
                            elif '!resetthresholds' in body:
                                data['thresholds'] = DEFAULT_THRESHOLDS.copy()
                                save_thresholds_to_wiki(subreddit, data['thresholds'])
                                mention.reply(f"✅ Thresholds reset to defaults for r/{subreddit_name}")
                                print(f"[r/{subreddit_name}] Thresholds reset by {mention.author}")
                            
                            # Set threshold
                            elif '!setthreshold' in body:
                                parts = body.split()
                                if len(parts) >= 3:
                                    param = parts[1].lower()
                                    try:
                                        value = float(parts[2])
                                        
                                        if param in data['thresholds']:
                                            old_value = data['thresholds'][param]
                                            data['thresholds'][param] = value
                                            save_thresholds_to_wiki(subreddit, data['thresholds'])
                                            mention.reply(f"✅ Updated `{param}` from `{old_value}` to `{value}` for r/{subreddit_name}")
                                            print(f"[r/{subreddit_name}] Threshold {param} updated to {value} by {mention.author}")
                                        else:
                                            mention.reply(f"❌ Unknown parameter: `{param}`\n\nValid parameters: hash_distance, hash_ai_similarity, orb_similarity, orb_ai_similarity")
                                    except ValueError:
                                        mention.reply(f"❌ Invalid value. Please provide a number.")
                                else:
                                    mention.reply(f"❌ Usage: `!setthreshold <parameter> <value>`")
                        
                        # Check modmail
                        for message in subreddit.modmail.conversations(state='new', limit=10):
                            if message.id in processed_items:
                                continue
                            
                            processed_items.add(message.id)
                            
                            # Similar command parsing for modmail
                            # (implementation similar to above)
                        
                        # Periodically reload thresholds from wiki (in case edited directly)
                        if time.time() - data['last_threshold_reload'] > 300:  # Every 5 minutes
                            new_thresholds = load_thresholds_from_wiki(subreddit)
                            if new_thresholds != data['thresholds']:
                                data['thresholds'] = new_thresholds
                                print(f"[r/{subreddit_name}] Reloaded thresholds from wiki: {new_thresholds}")
                            data['last_threshold_reload'] = time.time()
                        
                        # Clean up old processed items
                        if len(processed_items) > 1000:
                            processed_items.clear()
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in command listener: {e}")
            
            except Exception as e:
                print(f"Error in shared command listener: {e}")
            
            time.sleep(10)
    
    def shared_mod_log_monitor():
        """Single thread monitoring mod logs for ALL subreddits"""
        while True:
            try:
                with subreddit_data_lock:
                    subreddits = list(subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        subreddit = data['subreddit']
                        
                        for log_entry in subreddit.mod.log(action='removelink', limit=50):
                            if log_entry.id in data['processed_log_items']:
                                continue
                            
                            data['processed_log_items'].add(log_entry.id)
                            removed_submission_id = log_entry.target_fullname.replace('t3_', '')
                            
                            hash_to_process = None
                            for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                                if submission_id == removed_submission_id:
                                    hash_to_process = hash_value
                                    break
                            
                            if hash_to_process and hash_to_process not in data['moderator_removed_hashes']:
                                data['moderator_removed_hashes'].add(hash_to_process)
                                print(f"[r/{subreddit_name}] [MOD REMOVE] Submission {removed_submission_id} removed by moderator. Hash kept for future duplicate detection.")
                            
                            if len(data['processed_log_items']) > 1000:
                                data['processed_log_items'].clear()
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in mod log check: {e}")
                
            except Exception as e:
                print(f"Error in shared mod log monitor: {e}")
            
            time.sleep(30)
    
    def shared_removal_checker():
        """Single thread checking for removed posts across ALL subreddits"""
        while True:
            try:
                with subreddit_data_lock:
                    subreddits = list(subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        current_check_time = time.time()
                        checked_this_cycle = 0
                        
                        recent_submissions = []
                        medium_submissions = []
                        old_submissions = []
                        
                        for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                            if hash_value in data['moderator_removed_hashes']:
                                continue
                            
                            age = current_check_time - creation_time
                            last_check = data['last_checked'].get(submission_id, 0)
                            
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
                                    if hash_value in data['image_hashes']:
                                        del data['image_hashes'][hash_value]
                                    if submission_id in data['orb_descriptors']:
                                        del data['orb_descriptors'][submission_id]
                                    if submission_id in data['ai_features']:
                                        del data['ai_features'][submission_id]
                                    if submission_id in data['last_checked']:
                                        del data['last_checked'][submission_id]
                                    print(f"[r/{subreddit_name}] [USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                                else:
                                    data['last_checked'][submission_id] = current_check_time
                                
                                checked_this_cycle += 1
                                
                                if checked_this_cycle >= 10:
                                    time.sleep(60)
                                    checked_this_cycle = 0
                                
                            except Exception as e:
                                print(f"[r/{subreddit_name}] Error checking submission {submission_id}: {e}")
                                data['last_checked'][submission_id] = current_check_time
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in removal check: {e}")
                
            except Exception as e:
                print(f"Error in shared removal checker: {e}")
            
            time.sleep(60)
    
    def shared_modqueue_worker():
        """Single thread processing mod queue for ALL subreddits"""
        while True:
            try:
                with subreddit_data_lock:
                    subreddits = list(subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        subreddit = data['subreddit']
                        
                        modqueue_submissions = subreddit.mod.modqueue(only='submission', limit=None)
                        modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                        
                        for submission in modqueue_submissions:
                            if not isinstance(submission, praw.models.Submission):
                                continue
                            
                            print(f"[r/{subreddit_name}] Scanning Mod Queue: ", submission.url)
                            
                            if submission.num_reports > 0:
                                print(f"[r/{subreddit_name}] Skipping reported image: ", submission.url)
                                data['image_hashes'] = {k: v for k, v in data['image_hashes'].items() if v[0] != submission.id}
                                data['orb_descriptors'].pop(submission.id, None)
                                data['ai_features'].pop(submission.id, None)
                                continue
                            
                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                data['process_submission'](submission, context="modqueue")
                                data['processed_modqueue_submissions'].add(submission.id)
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in modqueue worker: {e}")
            
            except Exception as e:
                print(f"Error in shared modqueue worker: {e}")
            
            time.sleep(15)
    
    def shared_stream_worker():
        """Single thread streaming new submissions for ALL subreddits"""
        while True:
            try:
                with subreddit_data_lock:
                    subreddits = list(subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = subreddit_data[subreddit_name]
                        subreddit = data['subreddit']
                        
                        for submission in subreddit.new(limit=10):
                            if submission.created_utc > data['current_time'] and isinstance(submission, praw.models.Submission):
                                if submission.id in data['processed_modqueue_submissions']:
                                    continue

                                print(f"[r/{subreddit_name}] Scanning new image/post: ", submission.url)
                                
                                if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                    data['process_submission'](submission, context="stream")
                        
                        data['current_time'] = int(time.time())
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in stream worker: {e}")
            
            except Exception as e:
                print(f"Error in shared stream worker: {e}")
            
            time.sleep(20)
    
    # --- Accept invites and setup subreddits ---
    def check_for_invites():
        """Check for moderator invites and automatically accept them"""
        while True:
            try:
                # Check unread messages for mod invites
                for message in reddit.inbox.unread(limit=None):
                    if "invitation to moderate" in message.subject.lower():
                        subreddit_name = message.subreddit.display_name
                        print(f"\n*** Found mod invite for r/{subreddit_name} ***")
                        try:
                            message.subreddit.mod.accept_invite()
                            print(f"✅ Accepted mod invite for r/{subreddit_name}")
                            setup_subreddit(subreddit_name)
                        except Exception as e:
                            print(f"Error accepting invite for r/{subreddit_name}: {e}")
                        message.mark_read()
            
                # Also check for already accepted subreddits
                for subreddit in reddit.user.moderator_subreddits(limit=None):
                    subreddit_name = subreddit.display_name
                    if subreddit_name not in subreddit_data:
                        print(f"\n*** Already moderating r/{subreddit_name}, setting up bot ***")
                        setup_subreddit(subreddit_name)
            
            except Exception as e:
                print(f"Error checking for invites: {e}")
        
            time.sleep(60)
    
    # Start ONLY 6 shared worker threads (total, regardless of number of subreddits)
    threading.Thread(target=check_for_invites, daemon=True).start()
    threading.Thread(target=shared_command_listener, daemon=True).start()
    threading.Thread(target=shared_mod_log_monitor, daemon=True).start()
    threading.Thread(target=shared_removal_checker, daemon=True).start()
    threading.Thread(target=shared_modqueue_worker, daemon=True).start()
    threading.Thread(target=shared_stream_worker, daemon=True).start()
    
    # Keep main thread alive
    print("=== Multi-subreddit duplicate bot started ===")
    print("Running with 6 shared worker threads for all subreddits")
    print("Monitoring for mod invites and threshold commands...")
    while True:
        time.sleep(3600)  # Keep alive
