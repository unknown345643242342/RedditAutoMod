def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    
    # Global dictionary to store per-subreddit data
    subreddit_data = {}
    
    def setup_subreddit(subreddit_name):
        """Initialize data structures and monitoring for a specific subreddit"""
        print(f"\n=== Setting up bot for r/{subreddit_name} ===")
        
        subreddit = reddit.subreddit(subreddit_name)
        
        # Create dedicated dictionaries for this subreddit
        data = {
            'subreddit': subreddit,
            'image_hashes': {},
            'video_hashes': {},  # NEW: For video perceptual hashes
            'video_features': {},  # NEW: For video AI features
            'orb_descriptors': {},
            'moderator_removed_hashes': set(),
            'moderator_removed_video_hashes': set(),  # NEW: Track removed videos
            'processed_modqueue_submissions': set(),
            'approved_by_moderator': set(),
            'ai_features': {},
            'current_time': int(time.time())
        }
        
        subreddit_data[subreddit_name] = data
        
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

        # --- NEW: Video processing functions ---
        def download_video(url):
            """Download video to temporary file"""
            try:
                response = requests.get(url, timeout=30, stream=True)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                return temp_file.name
            except Exception as e:
                print(f"[r/{subreddit_name}] Video download error:", e)
                return None

        def extract_video_frames(video_path, num_frames=10):
            """Extract evenly spaced frames from video"""
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames == 0:
                    cap.release()
                    return []
                
                # Extract evenly spaced frames
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frames = []
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                
                cap.release()
                return frames
            except Exception as e:
                print(f"[r/{subreddit_name}] Frame extraction error:", e)
                return []

        def get_video_hash(frames):
            """Compute average perceptual hash from video frames"""
            if not frames:
                return None
            
            try:
                hashes = []
                for frame in frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_hash = imagehash.phash(Image.fromarray(gray))
                    hashes.append(frame_hash)
                
                # Average hash
                avg_hash = sum(hashes) / len(hashes)
                return str(avg_hash)
            except Exception as e:
                print(f"[r/{subreddit_name}] Video hash error:", e)
                return None

        def get_video_features(frames):
            """Extract AI features from video frames and average them"""
            if not frames:
                return None
            
            try:
                features_list = []
                for frame in frames:
                    feat = get_ai_features(frame)
                    if feat is not None:
                        features_list.append(feat)
                
                if not features_list:
                    return None
                
                # Average features
                avg_features = torch.mean(torch.stack(features_list), dim=0)
                return avg_features
            except Exception as e:
                print(f"[r/{subreddit_name}] Video features error:", e)
                return None

        def is_video_url(url):
            """Check if URL is a video"""
            video_extensions = ('.mp4', '.webm', '.mov', '.avi', '.gifv')
            video_domains = ('v.redd.it', 'i.imgur.com/.*\.gifv', 'gfycat.com', 'redgifs.com')
            
            if url.endswith(video_extensions):
                return True
            
            for domain in video_domains:
                if domain in url:
                    return True
            
            return False

        def load_and_process_video(url):
            """Download video, extract frames, compute hash and features"""
            video_path = download_video(url)
            if not video_path:
                return None, None, None
            
            try:
                frames = extract_video_frames(video_path, num_frames=10)
                if not frames:
                    return None, None, None
                
                video_hash = get_video_hash(frames)
                video_features = get_video_features(frames)
                
                print(f"[r/{subreddit_name}] Generated video hash: {video_hash}")
                
                return video_hash, video_features, frames
            finally:
                # Clean up temp file
                try:
                    os.unlink(video_path)
                except:
                    pass

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
            
            # Check if it's a video
            if is_video_url(old_submission.url):
                if submission_id in data['video_features']:
                    return data['video_features'][submission_id]
                
                video_hash, video_features, _ = load_and_process_video(old_submission.url)
                if video_features is not None:
                    data['video_features'][submission_id] = video_features
                    return video_features
                return None
            
            # It's an image
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
            
            print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f}")
            
            if ai_score > 0.50:
                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                original_status = "Removed by Moderator" if matched_hash in data['moderator_removed_hashes'] else "Active"
                return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
            
            return False, None, None, None, None, None, None

        def check_video_hash_duplicate(submission, video_hash, new_features):
            """Check if video submission is a hash-based duplicate"""
            matched_hash = None
            for stored_hash in data['video_hashes'].keys():
                if video_hash == stored_hash or (imagehash.hex_to_hash(video_hash) - imagehash.hex_to_hash(stored_hash)) <= 5:
                    matched_hash = stored_hash
                    break
            
            if matched_hash is None:
                return False, None, None, None, None, None, None
            
            original_id, original_time = data['video_hashes'][matched_hash]
            
            if submission.id == original_id or submission.created_utc <= original_time:
                return False, None, None, None, None, None, None
            
            original_submission = reddit.submission(id=original_id)
            original_features = get_cached_ai_features(original_submission.id)
            
            ai_score = calculate_ai_similarity(new_features, original_features)
            
            print(f"[r/{subreddit_name}] Video hash match detected. AI similarity: {ai_score:.2f}")
            
            if ai_score > 0.65:  # Slightly higher threshold for videos
                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                original_status = "Removed by Moderator" if matched_hash in data['moderator_removed_video_hashes'] else "Active"
                return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
            
            return False, None, None, None, None, None, None

        def check_orb_duplicate(submission, descriptors, new_features):
            """Check if submission is an ORB-based duplicate"""
            for old_id, old_desc in data['orb_descriptors'].items():
                sim = orb_similarity(descriptors, old_desc)
                
                if sim > 0.50:
                    old_features = get_cached_ai_features(old_id)
                    
                    ai_score = calculate_ai_similarity(new_features, old_features)
                    
                    if ai_score > 0.75:
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

        def handle_moderator_removed_repost(submission, hash_value, is_video=False):
            """Handle reposts of moderator-removed images/videos"""
            removed_set = data['moderator_removed_video_hashes'] if is_video else data['moderator_removed_hashes']
            hash_dict = data['video_hashes'] if is_video else data['image_hashes']
            
            if hash_value in removed_set and not submission.approved:
                submission.mod.remove()
                original_submission = reddit.submission(id=hash_dict[hash_value][0])
                post_comment(
                    submission,
                    original_submission.author.name,
                    original_submission.title,
                    datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    original_submission.created_utc,
                    "Removed by Moderator",
                    original_submission.permalink
                )
                media_type = "video" if is_video else "image"
                print(f"[r/{subreddit_name}] Repost of a moderator-removed {media_type} removed: ", submission.url)
                return True
            return False

        def process_video_submission(submission, context="stream"):
            """Process video submissions for duplicates"""
            try:
                video_hash, video_features, frames = load_and_process_video(submission.url)
                
                if video_hash is None or video_features is None:
                    print(f"[r/{subreddit_name}] Failed to process video: {submission.url}")
                    return False
                
                data['video_features'][submission.id] = video_features
                
                if handle_moderator_removed_repost(submission, video_hash, is_video=True):
                    return True
                
                is_duplicate, author, title, date, utc, status, permalink = check_video_hash_duplicate(
                    submission, video_hash, video_features
                )
                if is_duplicate:
                    return handle_duplicate(submission, True, "video hash + AI", author, title, date, utc, status, permalink)
                
                if video_hash not in data['video_hashes']:
                    data['video_hashes'][video_hash] = (submission.id, submission.created_utc)
                    data['video_features'][submission.id] = video_features
                    print(f"[r/{subreddit_name}] Stored new original video: {submission.url}")
                
                if context == "modqueue" and not submission.approved:
                    submission.mod.approve()
                    print(f"[r/{subreddit_name}] Original video submission approved: ", submission.url)
                
                return False
                
            except Exception as e:
                handle_exception(e)
                return False

        def process_submission_for_duplicates(submission, context="stream"):
            """Main duplicate detection logic - works for both mod queue and stream"""
            try:
                # Check if it's a video
                if is_video_url(submission.url):
                    return process_video_submission(submission, context)
                
                # Otherwise process as image
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
                            
                            # Check image hashes
                            hash_to_process = None
                            for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                                if submission_id == removed_submission_id:
                                    hash_to_process = hash_value
                                    break
                            
                            if hash_to_process and hash_to_process not in data['moderator_removed_hashes']:
                                data['moderator_removed_hashes'].add(hash_to_process)
                                print(f"[r/{subreddit_name}] [MOD REMOVE] Image {removed_submission_id} removed by moderator.")
                            
                            # Check video hashes
                            video_hash_to_process = None
                            for hash_value, (submission_id, creation_time) in list(data['video_hashes'].items()):
                                if submission_id == removed_submission_id:
                                    video_hash_to_process = hash_value
                                    break
                            
                            if video_hash_to_process and video_hash_to_process not in data['moderator_removed_video_hashes']:
                                data['moderator_removed_video_hashes'].add(video_hash_to_process)
                                print(f"[r/{subreddit_name}] [MOD REMOVE] Video {removed_submission_id} removed by moderator.")
                            
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
                    
                    # Check both images and videos
                    all_items = []
                    
                    for hash_value, (submission_id, creation_time) in list(data['image_hashes'].items()):
                        if hash_value not in data['moderator_removed_hashes']:
                            all_items.append(('image', hash_value, submission_id, creation_time))
                    
                    for hash_value, (submission_id, creation_time) in list(data['video_hashes'].items()):
                        if hash_value not in data['moderator_removed_video_hashes']:
                            all_items.append(('video', hash_value, submission_id, creation_time))
                    
                    recent_items = []
                    medium_items = []
                    old_items = []
                    
                    for item_type, hash_value, submission_id, creation_time in all_items:
                        age = current_check_time - creation_time
                        last_check = last_checked.get(submission_id, 0)
                        
                        if age < 3600:
                            check_interval = 30
                            if current_check_time - last_check >= check_interval:
                                recent_items.append((item_type, hash_value, submission_id))
                        elif age < 86400:
                            check_interval = 300
                            if current_check_time - last_check >= check_interval:
                                medium_items.append((item_type, hash_value, submission_id))
                        else:
                            check_interval = 1800
                            if current_check_time - last_check >= check_interval:
                                old_items.append((item_type, hash_value, submission_id))
                    
                    all_to_check = recent_items + medium_items[:20] + old_items[:10]
                    
                    for item_type, hash_value, submission_id in all_to_check:
                        try:
                            original_submission = reddit.submission(id=submission_id)
                            original_author = original_submission.author
                            
                            if original_author is None:
                                # Clean up deleted submission
                                if item_type == 'image':
                                    if hash_value in data['image_hashes']:
                                        del data['image_hashes'][hash_value]
                                    if submission_id in data['orb_descriptors']:
                                        del data['orb_descriptors'][submission_id]
                                    if submission_id in data['ai_features']:
                                        del data['ai_features'][submission_id]
                                else:  # video
                                    if hash_value in data['video_hashes']:
                                        del data['video_hashes'][hash_value]
                                    if submission_id in data['video_features']:
                                        del data['video_features'][submission_id]
                                
                                if submission_id in last_checked:
                                    del last_checked[submission_id]
                                print(f"[r/{subreddit_name}] [USER DELETE] {item_type.capitalize()} {submission_id} deleted by user.")
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
                if isinstance(submission, praw.models.Submission):
                    # Check for images
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        print(f"[r/{subreddit_name}] Indexing image (initial scan): ", submission.url)
                        try:
                            img, hash_value, descriptors, features = load_and_process_image(submission.url)
                            if hash_value not in data['image_hashes']:
                                data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                                data['orb_descriptors'][submission.id] = descriptors
                                data['ai_features'][submission.id] = features
                        except Exception as e:
                            handle_exception(e)
                    # Check for videos
                    elif is_video_url(submission.url):
                        print(f"[r/{subreddit_name}] Indexing video (initial scan): ", submission.url)
                        try:
                            video_hash, video_features, _ = load_and_process_video(submission.url)
                            if video_hash and video_hash not in data['video_hashes']:
                                data['video_hashes'][video_hash] = (submission.id, submission.created_utc)
                                data['video_features'][submission.id] = video_features
                        except Exception as e:
                            handle_exception(e)
        except Exception as e:
            handle_exception(e)
        
        print(f"[r/{subreddit_name}] Initial scan complete. Indexed {len(data['image_hashes'])} images and {len(data['video_hashes'])} videos.")

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
                            print(f"[r/{subreddit_name}] Skipping reported submission: ", submission.url)
                            # Clean up from tracking
                            data['image_hashes'] = {k: v for k, v in data['image_hashes'].items() if v[0] != submission.id}
                            data['video_hashes'] = {k: v for k, v in data['video_hashes'].items() if v[0] != submission.id}
                            data['orb_descriptors'].pop(submission.id, None)
                            data['ai_features'].pop(submission.id, None)
                            data['video_features'].pop(submission.id, None)
                            continue
                        
                        # Process images
                        if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                            is_duplicate = process_submission_for_duplicates(submission, context="modqueue")
                            data['processed_modqueue_submissions'].add(submission.id)
                        # Process videos
                        elif is_video_url(submission.url):
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

                            print(f"[r/{subreddit_name}] Scanning new submission: ", submission.url)
                            
                            # Process images
                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                process_submission_for_duplicates(submission, context="stream")
                            # Process videos
                            elif is_video_url(submission.url):
                                process_submission_for_duplicates(submission, context="stream")

                    data['current_time'] = int(time.time())
                except Exception as e:
                    handle_exception(e)
                time.sleep(20)
        
        threading.Thread(target=stream_worker, daemon=True).start()
        
        print(f"[r/{subreddit_name}] Bot fully operational!\n")

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
    print("Now supporting both images and videos!")
    while True:
        time.sleep(3600)  # Keep alive
