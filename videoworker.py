async def run_pokemon_duplicate_bot():
    reddit = await initialize_reddit()
    
    # Global dictionary to store per-subreddit data
    subreddit_data = {}
    
    async def setup_subreddit(subreddit_name):
        """Initialize data structures and monitoring for a specific subreddit"""
        print(f"\n=== Setting up bot for r/{subreddit_name} ===")
        
        subreddit = await reddit.subreddit(subreddit_name)
        
        # Create dedicated dictionaries for this subreddit
        data = {
            'subreddit': subreddit,
            'image_hashes': {},
            'orb_descriptors': {},
            'moderator_removed_hashes': set(),
            'processed_modqueue_submissions': set(),
            'approved_by_moderator': set(),
            'ai_features': {},
            'repost_history': {},  # Track repost attempts: hash -> list of (author, title, date, utc, permalink)
            'current_time': int(time.time()),
            'currently_processing': set()  # Track submissions currently being processed
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

        async def post_comment(submission, original_post_author, original_post_title, original_post_date, original_post_utc, original_status, original_post_permalink, matched_hash):
            max_retries = 3
            retries = 0
            age_text = format_age(original_post_utc)
            
            # Build repost history section (excluding current submission)
            repost_section = ""
            if matched_hash in data['repost_history'] and len(data['repost_history'][matched_hash]) > 0:
                # Filter out current submission from history
                previous_reposts = [r for r in data['repost_history'][matched_hash] if r[4] != submission.permalink]
                
                if len(previous_reposts) > 0:
                    repost_section = "\n\n---\n\n**Previous repost attempts:**\n\n"
                    repost_section += "| Author | Title | Date | Age | Status |\n"
                    repost_section += "|:------:|:-----:|:----:|:---:|:------:|\n"
                    
                    for repost in previous_reposts:
                        repost_author, repost_title, repost_date, repost_utc, repost_permalink, repost_status = repost
                        repost_age = format_age(repost_utc)
                        repost_section += f"| {repost_author} | [{repost_title}]({repost_permalink}) | {repost_date} | {repost_age} | {repost_status} |\n"
            
            while retries < max_retries:
                try:
                    comment_text = (
                        "> **Duplicate detected**\n\n"
                        "| Original Author | Title | Date | Age | Status |\n"
                        "|:---------------:|:-----:|:----:|:---:|:------:|\n"
                        f"| {original_post_author} | [{original_post_title}]({original_post_permalink}) | {original_post_date} | {age_text} | {original_status} |"
                        f"{repost_section}"
                    )
                    comment = await submission.reply(comment_text)
                    await comment.mod.distinguish(sticky=True)
                    await comment.mod.lock()
                    print(f"[r/{subreddit_name}] Duplicate removed and comment posted (locked): ", submission.url)
                    return True
                except Exception as e:
                    handle_exception(e)
                    retries += 1
                    await asyncio.sleep(1)
            return False

        def add_to_repost_history(matched_hash, submission):
            """Add current submission to repost history"""
            if matched_hash not in data['repost_history']:
                data['repost_history'][matched_hash] = []
            
            repost_data = (
                submission.author.name,
                submission.title,
                datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                submission.created_utc,
                submission.permalink,
                "Removed (Repost)"  # Status for reposts
            )
            data['repost_history'][matched_hash].append(repost_data)

        async def load_and_process_image(url):
            """Load image from URL and compute hash, descriptors, AI features"""
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    image_data = await response.read()
            
            img = np.asarray(bytearray(image_data), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hash_value = str(imagehash.phash(Image.fromarray(gray)))
            print(f"[r/{subreddit_name}] Generated hash: {hash_value}")
            
            descriptors = get_orb_descriptors_conditional(img)
            features = get_ai_features(img)
                
            return img, hash_value, descriptors, features

        async def get_cached_ai_features(submission_id):
            """Get AI features from cache or compute them"""
            if submission_id in data['ai_features']:
                return data['ai_features'][submission_id]
            
            old_submission = await reddit.submission(id=submission_id)
            async with aiohttp.ClientSession() as session:
                async with session.get(old_submission.url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    old_image_data = await response.read()
            old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
            old_features = get_ai_features(old_img)
            data['ai_features'][submission_id] = old_features
            return old_features

        def calculate_ai_similarity(features1, features2):
            """Calculate AI similarity score between two feature vectors"""
            if features1 is not None and features2 is not None:
                return (features1 @ features2.T).item()
            return 0

        async def check_hash_duplicate(submission, hash_value, new_features):
            """Check if submission is a hash-based duplicate"""
            matched_hash = None
            for stored_hash in data['image_hashes'].keys():
                if hash_value == stored_hash or (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= 4:
                    matched_hash = stored_hash
                    break
            
            if matched_hash is None:
                return False, None, None, None, None, None, None, None
            
            original_id, original_time = data['image_hashes'][matched_hash]
            
            if submission.id == original_id or submission.created_utc <= original_time:
                return False, None, None, None, None, None, None, None
            
            original_submission = await reddit.submission(id=original_id)
            original_features = await get_cached_ai_features(original_submission.id)
            
            ai_score = calculate_ai_similarity(new_features, original_features)
            
            print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f}")
            
            if ai_score > 0.50:
                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                original_status = "Removed by Moderator" if matched_hash in data['moderator_removed_hashes'] else "Active"
                return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink, matched_hash
            
            return False, None, None, None, None, None, None, None

        async def check_orb_duplicate(submission, descriptors, new_features):
            """Check if submission is an ORB-based duplicate"""
            for old_id, old_desc in data['orb_descriptors'].items():
                sim = orb_similarity(descriptors, old_desc)
                
                if sim > 0.50:
                    old_features = await get_cached_ai_features(old_id)
                    
                    ai_score = calculate_ai_similarity(new_features, old_features)
                    
                    if ai_score > 0.75:
                        print(f"[r/{subreddit_name}] ORB duplicate found! AI similarity: {ai_score:.2f}")
                        
                        original_submission = await reddit.submission(id=old_id)
                        original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                        old_hash = next((h for h, v in data['image_hashes'].items() if v[0] == old_id), None)
                        original_status = "Removed by Moderator" if old_hash and old_hash in data['moderator_removed_hashes'] else "Active"
                        
                        return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink, old_hash
            
            return False, None, None, None, None, None, None, None

        async def handle_duplicate(submission, is_hash_dup, detection_method, author, title, date, utc, status, permalink, matched_hash):
            """Remove duplicate and post comment if not approved"""
            if not submission.approved:
                # Post comment first (before adding to history so current post isn't included)
                await post_comment(submission, author, title, date, utc, status, permalink, matched_hash)
                # Then add to history for future reposts
                add_to_repost_history(matched_hash, submission)
                await submission.mod.remove()
                print(f"[r/{subreddit_name}] Duplicate removed by {detection_method}: {submission.url}")
            return True

        async def handle_moderator_removed_repost(submission, hash_value):
            """Handle reposts of moderator-removed images"""
            if hash_value in data['moderator_removed_hashes'] and not submission.approved:
                original_submission = await reddit.submission(id=data['image_hashes'][hash_value][0])
                # Post comment first (before adding to history)
                await post_comment(
                    submission,
                    original_submission.author.name,
                    original_submission.title,
                    datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    original_submission.created_utc,
                    "Removed by Moderator",
                    original_submission.permalink,
                    hash_value
                )
                # Then add to history
                add_to_repost_history(hash_value, submission)
                await submission.mod.remove()
                print(f"[r/{subreddit_name}] Repost of a moderator-removed image removed: ", submission.url)
                return True
            return False

        async def process_submission_for_duplicates(submission, context="stream"):
            """Main duplicate detection logic - works for both mod queue and stream"""
            # Check if already being processed
            if submission.id in data['currently_processing']:
                print(f"[r/{subreddit_name}] Submission {submission.id} already being processed, skipping")
                return False
            
            # Mark as currently processing
            data['currently_processing'].add(submission.id)
            
            try:
                img, hash_value, descriptors, new_features = await load_and_process_image(submission.url)
                data['ai_features'][submission.id] = new_features
                
                if await handle_moderator_removed_repost(submission, hash_value):
                    return True
                
                is_duplicate, author, title, date, utc, status, permalink, matched_hash = await check_hash_duplicate(
                    submission, hash_value, new_features
                )
                if is_duplicate:
                    return await handle_duplicate(submission, True, "hash + AI", author, title, date, utc, status, permalink, matched_hash)
                
                is_duplicate, author, title, date, utc, status, permalink, matched_hash = await check_orb_duplicate(
                    submission, descriptors, new_features
                )
                if is_duplicate:
                    return await handle_duplicate(submission, False, "ORB + AI", author, title, date, utc, status, permalink, matched_hash)
                
                if hash_value not in data['image_hashes']:
                    data['image_hashes'][hash_value] = (submission.id, submission.created_utc)
                    data['orb_descriptors'][submission.id] = descriptors
                    data['ai_features'][submission.id] = new_features
                    print(f"[r/{subreddit_name}] Stored new original: {submission.url}")
                
                if context == "modqueue" and not submission.approved:
                    await submission.mod.approve()
                    print(f"[r/{subreddit_name}] Original submission approved: ", submission.url)
                
                return False
                
            except Exception as e:
                handle_exception(e)
                return False
            finally:
                # Always remove from currently processing when done
                data['currently_processing'].discard(submission.id)

        async def check_removed_original_posts():
            """Monitor for immediate removal detection using dual approach"""
            processed_log_items = set()
            last_checked = {}
            
            async def monitor_mod_log():
                while True:
                    try:
                        async for log_entry in subreddit.mod.stream.log(action='removelink', skip_existing=True):
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
                        await asyncio.sleep(5)
            
            # Start monitor_mod_log as a task
            asyncio.create_task(monitor_mod_log())
            
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
                            original_submission = await reddit.submission(id=submission_id)
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
                                if hash_value in data['repost_history']:
                                    del data['repost_history'][hash_value]
                                print(f"[r/{subreddit_name}] [USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                            else:
                                last_checked[submission_id] = current_check_time
                            
                            checked_this_cycle += 1
                            
                            if checked_this_cycle >= 10:
                                await asyncio.sleep(60)
                                checked_this_cycle = 0
                            
                        except Exception as e:
                            print(f"[r/{subreddit_name}] Error checking submission {submission_id}: {e}")
                            last_checked[submission_id] = current_check_time
                    
                except Exception as e:
                    handle_exception(e)
                
                await asyncio.sleep(60)

        # --- Initial scan ---
        print(f"[r/{subreddit_name}] Starting initial scan...")
        try:
            async for submission in subreddit.new(limit=15):
                if isinstance(submission, asyncpraw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    print(f"[r/{subreddit_name}] Indexing submission (initial scan): ", submission.url)
                    try:
                        img, hash_value, descriptors, features = await load_and_process_image(submission.url)
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
        async def modqueue_worker():
            while True:
                try:
                    modqueue_submissions = []
                    async for submission in subreddit.mod.modqueue(only='submission', limit=None):
                        modqueue_submissions.append(submission)
                    
                    modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                    for submission in modqueue_submissions:
                        if not isinstance(submission, asyncpraw.models.Submission):
                            continue
                        
                        # Skip if already processed by stream worker
                        if submission.id in data['processed_modqueue_submissions']:
                            continue
                        
                        print(f"[r/{subreddit_name}] Scanning Mod Queue: ", submission.url)
                        
                        if submission.num_reports > 0:
                            print(f"[r/{subreddit_name}] Skipping reported image: ", submission.url)
                            data['image_hashes'] = {k: v for k, v in data['image_hashes'].items() if v[0] != submission.id}
                            data['orb_descriptors'].pop(submission.id, None)
                            data['ai_features'].pop(submission.id, None)
                            continue
                        
                        if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                            is_duplicate = await process_submission_for_duplicates(submission, context="modqueue")
                            data['processed_modqueue_submissions'].add(submission.id)

                except Exception as e:
                    handle_exception(e)
                await asyncio.sleep(15)

        # --- Stream new submissions ---
        async def stream_worker():
            processed_in_stream = set()
            while True:
                try:
                    async for submission in subreddit.stream.submissions(skip_existing=True):
                        if submission.created_utc > data['current_time'] and isinstance(submission, asyncpraw.models.Submission):
                            # Check if already processed in ANY context
                            if submission.id in data['processed_modqueue_submissions'] or submission.id in processed_in_stream:
                                continue

                            print(f"[r/{subreddit_name}] Scanning new image/post: ", submission.url)
                            
                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                # Mark as processed BEFORE processing to prevent race conditions
                                data['processed_modqueue_submissions'].add(submission.id)
                                processed_in_stream.add(submission.id)
                                await process_submission_for_duplicates(submission, context="stream")

                    data['current_time'] = int(time.time())
                except Exception as e:
                    handle_exception(e)
                await asyncio.sleep(20)
        
        # Start async workers
        asyncio.create_task(check_removed_original_posts())
        asyncio.create_task(modqueue_worker())
        asyncio.create_task(stream_worker())
        
        print(f"[r/{subreddit_name}] Bot fully operational!\n")

    # --- Accept invites and setup subreddits ---
    async def check_for_invites():
        """Check for moderator invites and automatically accept them"""
        while True:
            try:
                # Check unread messages for mod invites
                async for message in reddit.inbox.unread(limit=None):
                    if "invitation to moderate" in message.subject.lower():
                        subreddit_name = message.subreddit.display_name
                        print(f"\n*** Found mod invite for r/{subreddit_name} ***")
                        try:
                            await message.subreddit.mod.accept_invite()
                            print(f"✅ Accepted mod invite for r/{subreddit_name}")
                            asyncio.create_task(setup_subreddit(subreddit_name))
                        except Exception as e:
                            print(f"Error accepting invite for r/{subreddit_name}: {e}")
                        await message.mark_read()
            
                # Also check for already accepted subreddits
                async for subreddit in reddit.user.moderator_subreddits(limit=None):
                    subreddit_name = subreddit.display_name
                    if subreddit_name not in subreddit_data:
                        print(f"\n*** Already moderating r/{subreddit_name}, setting up bot ***")
                        asyncio.create_task(setup_subreddit(subreddit_name))
            
            except Exception as e:
                print(f"Error checking for invites: {e}")
        
            await asyncio.sleep(60)
    
    # Start the invite checker
    asyncio.create_task(check_for_invites())
    
    # Keep main thread alive
    print("=== Multi-subreddit duplicate bot started ===")
    print("Monitoring for mod invites...")
    while True:
        await asyncio.sleep(10)  # Keep alive
