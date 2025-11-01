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
    
    def setup_subreddit(subreddit_name):
        """Initialize data structures for a specific subreddit (no new threads)"""
        print(f"\n=== Setting up bot for r/{subreddit_name} ===")
        
        subreddit = reddit.subreddit(subreddit_name)
        
        # Create dedicated dictionaries for this subreddit
        data = {
            'subreddit': subreddit,
            'thresholds': DEFAULT_THRESHOLDS.copy(),
            'image_hashes': {},
            'orb_descriptors': {},
            'moderator_removed_hashes': set(),
            'processed_modqueue_submissions': set(),
            'approved_by_moderator': set(),
            'ai_features': {},
            'current_time': int(time.time()),
            'processed_log_items': set(),
            'last_checked': {}
        }
        
        with subreddit_data_lock:
            subreddit_data[subreddit_name] = data
        
        print(f"[r/{subreddit_name}] Loaded thresholds: {data['thresholds']}")
        
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
    
    # --- Accept invites and check for threshold adjustment messages ---
    def check_for_invites_and_messages():
        """Check for moderator invites and threshold adjustment messages"""
        while True:
            try:
                # Check unread messages for mod invites AND threshold commands
                for message in reddit.inbox.unread(limit=None):
                    # Handle mod invites
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
                    
                    # Handle threshold adjustment messages
                    else:
                        try:
                            body = message.body.strip()
                            body_lower = body.lower()
                            author = message.author.name
                            
                            # Show current thresholds for a specific subreddit
                            if '!showthresholds' in body_lower:
                                parts = body_lower.split()
                                
                                # Check if subreddit was specified: !showthresholds r/pokemon
                                if len(parts) >= 2 and parts[1].startswith('r/'):
                                    target_subreddit = parts[1].replace('r/', '')
                                    
                                    # Verify subreddit exists in bot and user is a mod
                                    if target_subreddit in subreddit_data:
                                        target_data = subreddit_data[target_subreddit]
                                        try:
                                            if target_data['subreddit'].moderator(author):
                                                response = f"""**Current Threshold Settings for r/{target_subreddit}:**

- hash_distance: {target_data['thresholds']['hash_distance']}
- hash_ai_similarity: {target_data['thresholds']['hash_ai_similarity']}
- orb_similarity: {target_data['thresholds']['orb_similarity']}
- orb_ai_similarity: {target_data['thresholds']['orb_ai_similarity']}

To adjust: `!setthreshold r/{target_subreddit} <parameter> <value>`
To reset: `!resetthresholds r/{target_subreddit}`"""
                                                message.reply(response)
                                                print(f"[r/{target_subreddit}] Showed thresholds to {author}")
                                            else:
                                                message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                                        except:
                                            message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                                    else:
                                        message.reply(f"❌ Bot is not running on r/{target_subreddit}")
                                else:
                                    # No subreddit specified - show usage
                                    moderated_subs = []
                                    for sub_name, sub_data in subreddit_data.items():
                                        try:
                                            if sub_data['subreddit'].moderator(author):
                                                moderated_subs.append(sub_name)
                                        except:
                                            continue
                                    
                                    if moderated_subs:
                                        subs_list = ', '.join([f"r/{s}" for s in moderated_subs])
                                        message.reply(f"❌ Please specify a subreddit.\n\nUsage: `!showthresholds r/subredditname`\n\nYou moderate: {subs_list}")
                                    else:
                                        message.reply(f"❌ You do not moderate any subreddits where this bot is running.")
                                
                                message.mark_read()
                            
                            # Reset thresholds for a specific subreddit
                            elif '!resetthresholds' in body_lower:
                                parts = body_lower.split()
                                
                                # Check if subreddit was specified: !resetthresholds r/pokemon
                                if len(parts) >= 2 and parts[1].startswith('r/'):
                                    target_subreddit = parts[1].replace('r/', '')
                                    
                                    # Verify subreddit exists in bot and user is a mod
                                    if target_subreddit in subreddit_data:
                                        target_data = subreddit_data[target_subreddit]
                                        try:
                                            if target_data['subreddit'].moderator(author):
                                                target_data['thresholds'] = DEFAULT_THRESHOLDS.copy()
                                                message.reply(f"✅ Thresholds reset to defaults for r/{target_subreddit}")
                                                print(f"[r/{target_subreddit}] Thresholds reset to defaults by {author}")
                                                print(f"[r/{target_subreddit}] Current thresholds: {target_data['thresholds']}")
                                            else:
                                                message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                                        except:
                                            message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                                    else:
                                        message.reply(f"❌ Bot is not running on r/{target_subreddit}")
                                else:
                                    # No subreddit specified - show usage
                                    moderated_subs = []
                                    for sub_name, sub_data in subreddit_data.items():
                                        try:
                                            if sub_data['subreddit'].moderator(author):
                                                moderated_subs.append(sub_name)
                                        except:
                                            continue
                                    
                                    if moderated_subs:
                                        subs_list = ', '.join([f"r/{s}" for s in moderated_subs])
                                        message.reply(f"❌ Please specify a subreddit.\n\nUsage: `!resetthresholds r/subredditname`\n\nYou moderate: {subs_list}")
                                    else:
                                        message.reply(f"❌ You do not moderate any subreddits where this bot is running.")
                                
                                message.mark_read()
                            
                            # Set threshold for a specific subreddit
                            elif '!setthreshold' in body_lower:
                                parts = body.split()  # Use original body for case-sensitive parameter names
                                
                                # Expected format: !setthreshold r/pokemon hash_distance 5
                                if len(parts) >= 4 and parts[1].lower().startswith('r/'):
                                    target_subreddit = parts[1].lower().replace('r/', '')
                                    param = parts[2].lower()
                                    
                                    try:
                                        value = float(parts[3])
                                        
                                        # Verify subreddit exists in bot and user is a mod
                                        if target_subreddit in subreddit_data:
                                            target_data = subreddit_data[target_subreddit]
                                            try:
                                                if target_data['subreddit'].moderator(author):
                                                    if param in target_data['thresholds']:
                                                        old_value = target_data['thresholds'][param]
                                                        target_data['thresholds'][param] = value
                                                        message.reply(f"✅ Updated `{param}` from `{old_value}` to `{value}` for r/{target_subreddit}")
                                                        print(f"[r/{target_subreddit}] Threshold {param} updated: {old_value} → {value} by {author}")
                                                        print(f"[r/{target_subreddit}] Current thresholds: {target_data['thresholds']}")
                                                    else:
                                                        message.reply(f"❌ Unknown parameter: `{param}`\n\nValid parameters: hash_distance, hash_ai_similarity, orb_similarity, orb_ai_similarity")
                                                else:
                                                    message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                                            except:
                                                message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                                        else:
                                            message.reply(f"❌ Bot is not running on r/{target_subreddit}")
                                    
                                    except ValueError:
                                        message.reply(f"❌ Invalid value. Please provide a number.")
                                else:
                                    # Invalid format - show usage
                                    moderated_subs = []
                                    for sub_name, sub_data in subreddit_data.items():
                                        try:
                                            if sub_data['subreddit'].moderator(author):
                                                moderated_subs.append(sub_name)
                                        except:
                                            continue
                                    
                                    if moderated_subs:
                                        subs_list = ', '.join([f"r/{s}" for s in moderated_subs])
                                        message.reply(f"❌ Usage: `!setthreshold r/subredditname <parameter> <value>`\n\nValid parameters: hash_distance, hash_ai_similarity, orb_similarity, orb_ai_similarity\n\nYou moderate: {subs_list}")
                                    else:
                                        message.reply(f"❌ You do not moderate any subreddits where this bot is running.")
                                
                                message.mark_read()
                        
                        except Exception as e:
                            print(f"Error processing message: {e}")
            
                # Also check for already accepted subreddits
                for subreddit in reddit.user.moderator_subreddits(limit=None):
                    subreddit_name = subreddit.display_name
                    if subreddit_name not in subreddit_data:
                        print(f"\n*** Already moderating r/{subreddit_name}, setting up bot ***")
                        setup_subreddit(subreddit_name)
            
            except Exception as e:
                print(f"Error checking for invites and messages: {e}")
        
            time.sleep(60)
    
    # Start ONLY 5 shared worker threads (total, regardless of number of subreddits)
    threading.Thread(target=check_for_invites_and_messages, daemon=True).start()
    threading.Thread(target=shared_mod_log_monitor, daemon=True).start()
    threading.Thread(target=shared_removal_checker, daemon=True).start()
    threading.Thread(target=shared_modqueue_worker, daemon=True).start()
    threading.Thread(target=shared_stream_worker, daemon=True).start()
    
    # Keep main thread alive
    print("=== Multi-subreddit duplicate bot started ===")
    print("Running with 5 shared worker threads for all subreddits")
    print("Monitoring for mod invites and threshold adjustment messages...")
    while True:
        time.sleep(3600)  # Keep alive
