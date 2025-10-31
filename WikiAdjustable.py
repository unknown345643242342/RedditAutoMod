def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    
    # Global dictionary to store per-subreddit data
    subreddit_data = {}
    
    def create_default_wiki_config(subreddit):
        """Create default wiki configuration page"""
        default_config = """# Duplicate Detection Bot Configuration

## Similarity Thresholds

Edit the values below to adjust duplicate detection sensitivity. Values should be between 0.0 and 1.0.

### Hash Match + AI Verification
**hash_ai_threshold**: 0.50
- When a perceptual hash match is found, this AI similarity score must also be met to confirm duplicate
- Lower values = more sensitive (more duplicates detected)
- Higher values = less sensitive (fewer duplicates detected)

### ORB Match Threshold
**orb_match_threshold**: 0.50
- Feature matching threshold for detecting visually similar images
- Lower values = more sensitive
- Higher values = less sensitive

### ORB + AI Verification
**orb_ai_threshold**: 0.75
- When an ORB feature match is found, this AI similarity score must also be met to confirm duplicate
- Lower values = more sensitive
- Higher values = less sensitive

---
*Do not modify the parameter names (text before the colon). Only change the numerical values.*
"""
        try:
            subreddit.wiki.create(
                name="duplicatebot_config",
                content=default_config,
                reason="Initial duplicate bot configuration"
            )
            print(f"[r/{subreddit.display_name}] ✅ Created default wiki config page")
            return True
        except Exception as e:
            if "WIKI_CREATE_ERROR" in str(e) or "already exists" in str(e).lower():
                print(f"[r/{subreddit.display_name}] Wiki page already exists")
                return True
            print(f"[r/{subreddit.display_name}] ❌ Error creating wiki page: {e}")
            return False

    def parse_wiki_config(wiki_content):
        """Parse configuration values from wiki page"""
        config = {
            'hash_ai_threshold': 0.50,
            'orb_match_threshold': 0.50,
            'orb_ai_threshold': 0.75
        }
        
        lines = wiki_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('**') and '**:' in line:
                # Extract parameter name and value
                parts = line.split('**:')
                if len(parts) == 2:
                    param_name = parts[0].replace('**', '').strip()
                    try:
                        value = float(parts[1].strip())
                        if 0.0 <= value <= 1.0:
                            config[param_name] = value
                    except ValueError:
                        pass
        
        return config

    def load_wiki_config(subreddit):
        """Load configuration from wiki page"""
        try:
            wiki_page = subreddit.wiki["duplicatebot_config"]
            config = parse_wiki_config(wiki_page.content_md)
            print(f"[r/{subreddit.display_name}] 📖 Loaded wiki config:")
            print(f"    - hash_ai_threshold: {config['hash_ai_threshold']}")
            print(f"    - orb_match_threshold: {config['orb_match_threshold']}")
            print(f"    - orb_ai_threshold: {config['orb_ai_threshold']}")
            return config
        except Exception as e:
            print(f"[r/{subreddit.display_name}] ⚠️ Could not load wiki config: {e}")
            print(f"[r/{subreddit.display_name}] Using default values")
            return {
                'hash_ai_threshold': 0.50,
                'orb_match_threshold': 0.50,
                'orb_ai_threshold': 0.75
            }

    def monitor_wiki_changes(subreddit_name):
        """Monitor wiki page for changes and update config in real-time"""
        processed_revisions = set()
        
        while True:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                wiki_page = subreddit.wiki["duplicatebot_config"]
                
                # Get latest revision
                revisions = list(wiki_page.revisions(limit=1))
                if revisions:
                    latest_revision = revisions[0]
                    revision_id = latest_revision['id']
                    
                    if revision_id not in processed_revisions:
                        processed_revisions.add(revision_id)
                        
                        # Load new config
                        new_config = parse_wiki_config(wiki_page.content_md)
                        
                        # Update stored config
                        if subreddit_name in subreddit_data:
                            old_config = subreddit_data[subreddit_name]['config']
                            subreddit_data[subreddit_name]['config'] = new_config
                            
                            print(f"\n[r/{subreddit_name}] 🔄 Wiki configuration updated!")
                            print(f"[r/{subreddit_name}] Editor: {latest_revision['author']}")
                            
                            # Show what changed
                            for key in new_config:
                                if old_config[key] != new_config[key]:
                                    print(f"[r/{subreddit_name}]   {key}: {old_config[key]} → {new_config[key]}")
                            
                            if old_config == new_config:
                                print(f"[r/{subreddit_name}]   (No changes to threshold values)")
                            
                            print(f"[r/{subreddit_name}] ✅ Configuration update successful!\n")
                        
                        # Keep only recent revisions in memory
                        if len(processed_revisions) > 50:
                            processed_revisions.clear()
                
            except Exception as e:
                print(f"[r/{subreddit_name}] Error monitoring wiki: {e}")
            
            time.sleep(30)  # Check every 30 seconds
    
    def setup_subreddit(subreddit_name):
        """Initialize data structures and monitoring for a specific subreddit"""
        print(f"\n=== Setting up bot for r/{subreddit_name} ===")
        
        subreddit = reddit.subreddit(subreddit_name)
        
        # Create wiki config page if it doesn't exist
        create_default_wiki_config(subreddit)
        
        # Load initial configuration
        config = load_wiki_config(subreddit)
        
        # Create dedicated dictionaries for this subreddit
        data = {
            'subreddit': subreddit,
            'config': config,  # Store config here
            'image_hashes': {},
            'orb_descriptors': {},
            'moderator_removed_hashes': set(),
            'processed_modqueue_submissions': set(),
            'approved_by_moderator': set(),
            'ai_features': {},
            'current_time': int(time.time())
        }
        
        subreddit_data[subreddit_name] = data
        
        # Start wiki monitor thread
        threading.Thread(target=monitor_wiki_changes, args=(subreddit_name,), daemon=True).start()
        
        # --- AI model (shared across all subreddits) ---
        device = "cpu"
        efficientnet_model = models.efficientnet_b0(pretrained=True)
        efficientnet_model.eval()
        efficientnet_model.to(device)
        efficientnet_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Reuse ORB detector
        orb_detector = cv2.ORB_create()
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
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
            for stored_hash in data['image_hashes'].keys():
                if hash_value == stored_hash or (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= 3:
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
            
            # Get threshold from config
            threshold = data['config']['hash_ai_threshold']
            print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f} (threshold: {threshold:.2f})")
            
            if ai_score > threshold:
                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                original_status = "Removed by Moderator" if matched_hash in data['moderator_removed_hashes'] else "Active"
                return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
            
            return False, None, None, None, None, None, None

        def check_orb_duplicate(submission, descriptors, new_features):
            """Check if submission is an ORB-based duplicate"""
            orb_threshold = data['config']['orb_match_threshold']
            ai_threshold = data['config']['orb_ai_threshold']
            
            for old_id, old_desc in data['orb_descriptors'].items():
                sim = orb_similarity(descriptors, old_desc)
                
                if sim > orb_threshold:
                    old_features = get_cached_ai_features(old_id)
                    
                    ai_score = calculate_ai_similarity(new_features, old_features)
                    
                    print(f"[r/{subreddit_name}] ORB match: {sim:.2f} (threshold: {orb_threshold:.2f}), AI similarity: {ai_score:.2f} (threshold: {ai_threshold:.2f})")
                    
                    if ai_score > ai_threshold:
                        print(f"[r/{subreddit_name}] ORB duplicate confirmed!")
                        
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
                            for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                                if submission_id == removed_submission_id:
                                    hash_to_process = hash_value
                                    break
                            
                            if hash_to_process and hash_to_process not in data['moderator_removed_hashes']:
                                data['moderator_removed_hashes'].add(hash_to_process)
                                print(f"[r/{subreddit_name}] [MOD REMOVE] Submission {removed_submission_id} removed by moderator. Hash kept for future duplicate detection.")
                            
                            if len(processed_log_items) > 1000:
                                processed_log_items.clear()
                        
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in mod log monitor: {e}")
                        time.sleep(5)
            
            threading.Thread(target=monitor_mod_log, daemon=True).start()
            
            while True:
                try:
                    current_check_time = time.time()
                    checked_this_cycle = 0
                    
                    recent_submissions = []
                    medium_submissions = []
                    old_submissions = []
                    
                    for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                        if hash_value in data['moderator_removed_hashes']:
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
                                if hash_value in data['image_hashes']:
                                    del data['image_hashes'][hash_value]
                                if submission_id in data['orb_descriptors']:
                                    del data['orb_descriptors'][submission_id]
                                if submission_id in data['ai_features']:
                                    del data['ai_features'][submission_id]
                                if submission_id in last_checked:
                                    del last_checked[submission_id]
                                print(f"[r/{subreddit_name}] [USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                            else:
                                last_checked[submission_id] = current_check_time
                            
                            checked_this_cycle += 1
                            
                            if checked_this_cycle >= 10:
                                time.sleep(60)
                                checked_this_cycle = 0
                            
                        except Exception as e:
                            print(f"[r/{subreddit_name}] Error checking submission {submission_id}: {e}")
                            last_checked[submission_id] = current_check_time
                    
                except Exception as e:
                    handle_exception(e)
                
                time.sleep(60)
        
        threading.Thread(target=check_removed_original_posts, daemon=True).start()

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

        # --- Mod Queue worker ---
        def modqueue_worker():
            while True:
                try:
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
                            is_duplicate = process_submission_for_duplicates(submission, context="modqueue")
                            data['processed_modqueue_submissions'].add(submission.id)

                except Exception as e:
                    handle_exception(e)
                time.sleep(15)

        threading.Thread(target=modqueue_worker, daemon=True).start()

        # --- Stream new submissions ---
        def stream_worker():
            while True:
                try:
                    for submission in subreddit.stream.submissions(skip_existing=True):
                        if submission.created_utc > data['current_time'] and isinstance(submission, praw.models.Submission):
                            if submission.id in data['processed_modqueue_submissions']:
                                continue

                            print(f"[r/{subreddit_name}] Scanning new image/post: ", submission.url)
                            
                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                process_submission_for_duplicates(submission, context="stream")

                    data['current_time'] = int(time.time())
                except Exception as e:
                    handle_exception(e)
                time.sleep(20)
        
        threading.Thread(target=stream_worker, daemon=True).start()
        
        print(f"[r/{subreddit_name}] Bot fully operational!")
        print(f"[r/{subreddit_name}] Wiki config page: reddit.com/r/{subreddit_name}/wiki/duplicatebot_config\n")

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
                            threading.Thread(target=setup_subreddit, args=(subreddit_name,), daemon=True).start()
                        except Exception as e:
                            print(f"Error accepting invite for r/{subreddit_name}: {e}")
                        message.mark_read()
            
                # Also check for already accepted subreddits
                for subreddit in reddit.user.moderator_subreddits(limit=None):
                    subreddit_name = subreddit.display_name
                    if subreddit_name not in subreddit_data:
                        print(f"\n*** Already moderating r/{subreddit_name}, setting up bot ***")
                        threading.Thread(target=setup_subreddit, args=(subreddit_name,), daemon=True).start()
            
            except Exception as e:
                print(f"Error checking for invites: {e}")
        
            time.sleep(60)
    
    # Start the invite checker
    threading.Thread(target=check_for_invites, daemon=True).start()
    
    # Keep main thread alive
    print("=== Multi-subreddit duplicate bot started ===")
    print("Monitoring for mod invites...")
    while True:
        time.sleep(3600)  # Keep alive
