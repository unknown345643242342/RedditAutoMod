class RateLimiter:
    """Centralized rate limiter to prevent 429 errors"""
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_times = deque(maxlen=requests_per_minute)
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Block if we're about to exceed rate limit"""
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            
            # If at limit, wait until oldest request is > 1 minute old
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0]) + 1
                if sleep_time > 0:
                    print(f"[RATE LIMIT] Sleeping {sleep_time:.1f}s to avoid 429")
                    time.sleep(sleep_time)
                    self.request_times.popleft()
            
            self.request_times.append(time.time())

# Global rate limiter - 60 requests per minute (conservative)
rate_limiter = RateLimiter(requests_per_minute=50)

# =========================
# Crash-proof runner
# =========================
def safe_run(target, *args, **kwargs):
    """Keeps a target function running forever."""
    while True:
        try:
            target(*args, **kwargs)
        except prawcore.exceptions.ResponseException as e:
            if e.response.status_code == 429:
                print(f"[FATAL 429] {target.__name__} hit rate limit. Sleeping 60s...")
                time.sleep(60)
            else:
                print(f"[FATAL] {target.__name__} crashed: {e}")
                traceback.print_exc()
                time.sleep(10)
        except Exception as e:
            print(f"[FATAL] {target.__name__} crashed: {e}")
            traceback.print_exc()
            time.sleep(10)

# =========================
# Reddit init
# =========================
def initialize_reddit():
    return praw.Reddit(
        client_id='jl-I3OHYH2_VZMC1feoJMQ',
        client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
        username='PokeLeakBot3',
        password='testbot1',
        user_agent='testbot'
    )

# =========================
# Workers (with rate limiting)
# =========================
def monitor_reported_posts():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit("PokeLeaks")
    processed_ids = set()
    
    while True:
        try:
            rate_limiter.wait_if_needed()
            for post in subreddit.mod.reports(limit=25):
                if post.id in processed_ids:
                    continue
                    
                if getattr(post, "approved", False):
                    rate_limiter.wait_if_needed()
                    post.mod.approve()
                    print(f"Post {post.id} re-approved")
                
                processed_ids.add(post.id)
                
                # Limit memory
                if len(processed_ids) > 500:
                    processed_ids.clear()
            
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            print(f"Exception in monitor_reported_posts: {e}")
            time.sleep(10)

def handle_modqueue_items():
    reddit = initialize_reddit()
    timers = {}

    while True:
        try:
            rate_limiter.wait_if_needed()
            for item in reddit.subreddit('PokeLeaks').mod.modqueue(limit=50):
                if getattr(item, "num_reports", 0) == 1 and item.id not in timers:
                    timers[item.id] = time.time()
                    print(f"Starting timer for post {item.id}")

                if item.id in timers:
                    time_diff = time.time() - timers[item.id]
                    if time_diff >= 3600:
                        rate_limiter.wait_if_needed()
                        item.mod.approve()
                        print(f"Approved post {item.id} after 1 hour")
                        del timers[item.id]
            
            time.sleep(60)  # Check every minute instead of constantly
        except Exception as e:
            print(f"Exception in handle_modqueue_items: {e}")
            time.sleep(10)

def handle_spoiler_status():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    previous_spoiler_status = {}

    while True:
        try:
            rate_limiter.wait_if_needed()
            for submission in subreddit.new(limit=50):
                if submission.id not in previous_spoiler_status:
                    previous_spoiler_status[submission.id] = submission.spoiler
                    continue

                if previous_spoiler_status[submission.id] != submission.spoiler:
                    try:
                        is_moderator = submission.author in subreddit.moderator()
                    except:
                        is_moderator = False

                    if not submission.spoiler and not is_moderator:
                        rate_limiter.wait_if_needed()
                        submission.mod.spoiler()
                        print(f'Post {submission.id} re-marked as spoiler')
                    
                    previous_spoiler_status[submission.id] = submission.spoiler
                
                # Limit memory
                if len(previous_spoiler_status) > 200:
                    # Remove oldest entries
                    oldest_keys = list(previous_spoiler_status.keys())[:50]
                    for key in oldest_keys:
                        del previous_spoiler_status[key]
            
            time.sleep(45)  # Check every 45 seconds
        except Exception as e:
            print(f"Exception in handle_spoiler_status: {e}")
            time.sleep(10)

def handle_user_reports_and_removal():
    reddit = initialize_reddit()
    thresholds = {
        'No Linking to Downloadable Content in Posts or Comments': 1,
        'No ROMs, ISOs, or Game Files Sharing or Requests': 1,
        'No insults or harassment of other subreddit members in the comments': 1
    }

    while True:
        try:
            rate_limiter.wait_if_needed()
            for comment in reddit.subreddit('PokeLeaks').mod.modqueue(limit=50):
                if isinstance(comment, praw.models.Comment) and getattr(comment, "user_reports", None):
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        rate_limiter.wait_if_needed()
                        comment.mod.remove()
                        print(f'Comment removed: {reason}')
            
            time.sleep(30)
        except Exception as e:
            print(f"Exception in handle_user_reports_and_removal: {e}")
            time.sleep(10)

def handle_submissions_based_on_user_reports():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            rate_limiter.wait_if_needed()
            for post in reddit.subreddit('PokeLeaks').mod.modqueue(limit=50):
                if isinstance(post, praw.models.Submission) and getattr(post, "user_reports", None):
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        rate_limiter.wait_if_needed()
                        post.mod.approve()
                        print(f'Post approved: {reason}')
            
            time.sleep(30)
        except Exception as e:
            print(f"Exception in handle_submissions_based_on_user_reports: {e}")
            time.sleep(10)

def handle_posts_based_on_removal():
    reddit = initialize_reddit()
    thresholds = {
        'Users Are Responsible for the Content They Post': 2,
        'Discussion-Only for Leaks, Not Distribution': 2,
        'No Linking to Downloadable Content in Posts or Comments': 1,
        'No ROMs, ISOs, or Game Files Sharing or Requests': 2,
        'Theories, Questions, Speculations must be commented in the Theory/Speculation/Question Megathread': 2,
        'Content Must Relate to Pokémon Leaks or News': 2,
        'Content must not contain any profanities, vulgarity, sexual content, slurs, be appropriate in nature': 2,
        'Post title should include sourcing and must be transparent': 2,
        'Posts with spoilers must have required spoiler flair, indicate spoiler alert in title, and be vague': 3,
        'No reposting of posts already up on the subreddit': 2,
        'No Self Advertisements or Promotion': 2,
        'No Memes, Fan Art, or Joke Posts': 2
    }

    while True:
        try:
            rate_limiter.wait_if_needed()
            for post in reddit.subreddit('PokeLeaks').mod.modqueue(limit=50):
                if isinstance(post, praw.models.Submission) and getattr(post, "user_reports", None):
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        rate_limiter.wait_if_needed()
                        post.mod.remove()
                        print(f'Post removed: {reason}')
            
            time.sleep(30)
        except Exception as e:
            print(f"Exception in handle_posts_based_on_removal: {e}")
            time.sleep(10)

def handle_comments_based_on_approval():
    reddit = initialize_reddit()
    thresholds = {'This is misinformation': 1, 'This is spam': 1}

    while True:
        try:
            rate_limiter.wait_if_needed()
            for comment in reddit.subreddit('PokeLeaks').mod.modqueue(limit=50):
                if getattr(comment, "user_reports", None):
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        rate_limiter.wait_if_needed()
                        comment.mod.approve()
                        print(f'Comment approved: {reason}')
            
            time.sleep(30)
        except Exception as e:
            print(f"Exception in handle_comments_based_on_approval: {e}")
            time.sleep(10)

def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    subreddit = reddit.subreddit('PokeLeaks')
    image_hashes = {}
    orb_descriptors = {}
    moderator_removed_hashes = set()
    processed_modqueue_submissions = set()
    ai_features = {}
    current_time = int(time.time())

    # AI model setup
    device = "cpu"
    resnet_model = models.resnet18(pretrained=True)
    resnet_model.eval()
    resnet_model.to(device)
    resnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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
        try:
            age_text = format_age(original_post_utc)
            comment_text = (
                "> **Duplicate detected**\n\n"
                "| Original Author | Title | Date | Age | Status |\n"
                "|:---------------:|:-----:|:----:|:---:|:------:|\n"
                f"| {original_post_author} | [{original_post_title}]({original_post_permalink}) | {original_post_date} | {age_text} | {original_status} |"
            )
            rate_limiter.wait_if_needed()
            comment = submission.reply(comment_text)
            comment.mod.distinguish(sticky=True)
            print("Duplicate removed and comment posted")
            return True
        except Exception as e:
            print(f"Error posting comment: {e}")
            return False

    def load_and_process_image(url):
        image_data = requests.get(url, timeout=10).content
        img = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        hash_value = str(imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))))
        descriptors = get_orb_descriptors_conditional(img)
        features = get_ai_features(img)
        return img, hash_value, descriptors, features

    def get_cached_ai_features(submission_id):
        if submission_id in ai_features:
            return ai_features[submission_id]
        
        rate_limiter.wait_if_needed()
        old_submission = reddit.submission(id=submission_id)
        old_image_data = requests.get(old_submission.url, timeout=10).content
        old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        old_features = get_ai_features(old_img)
        ai_features[submission_id] = old_features
        return old_features

    def calculate_ai_similarity(features1, features2):
        if features1 is not None and features2 is not None:
            return (features1 @ features2.T).item()
        return 0

    def check_hash_duplicate(submission, hash_value, new_features):
        if hash_value not in image_hashes:
            return False, None, None, None, None, None, None
        
        original_id, original_time = image_hashes[hash_value]
        
        if submission.id == original_id or submission.created_utc <= original_time:
            return False, None, None, None, None, None, None
        
        rate_limiter.wait_if_needed()
        original_submission = reddit.submission(id=original_id)
        original_features = get_cached_ai_features(original_submission.id)
        ai_score = calculate_ai_similarity(new_features, original_features)
        
        if ai_score > 0.70:
            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            original_status = "Removed by Moderator" if hash_value in moderator_removed_hashes else "Active"
            return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def check_orb_duplicate(submission, descriptors, new_features):
        for old_id, old_desc in list(orb_descriptors.items())[:100]:  # Limit checks
            sim = orb_similarity(descriptors, old_desc)
            
            if sim > 0.30:
                old_features = get_cached_ai_features(old_id)
                ai_score = calculate_ai_similarity(new_features, old_features)
                
                if ai_score > 0.70:
                    rate_limiter.wait_if_needed()
                    original_submission = reddit.submission(id=old_id)
                    original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    old_hash = next((h for h, v in image_hashes.items() if v[0] == old_id), None)
                    original_status = "Removed by Moderator" if old_hash and old_hash in moderator_removed_hashes else "Active"
                    
                    return True, original_submission.author.name, original_submission.title, original_post_date, original_submission.created_utc, original_status, original_submission.permalink
        
        return False, None, None, None, None, None, None

    def handle_duplicate(submission, detection_method, author, title, date, utc, status, permalink):
        if not submission.approved:
            rate_limiter.wait_if_needed()
            submission.mod.remove()
            post_comment(submission, author, title, date, utc, status, permalink)
            print(f"Duplicate removed by {detection_method}")
        return True

    def handle_moderator_removed_repost(submission, hash_value):
        if hash_value in moderator_removed_hashes and not submission.approved:
            rate_limiter.wait_if_needed()
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
            print("Repost of moderator-removed image removed")
            return True
        return False

    def process_submission_for_duplicates(submission, context="stream"):
        try:
            img, hash_value, descriptors, new_features = load_and_process_image(submission.url)
            ai_features[submission.id] = new_features
            
            if handle_moderator_removed_repost(submission, hash_value):
                return True
            
            is_duplicate, author, title, date, utc, status, permalink = check_hash_duplicate(
                submission, hash_value, new_features
            )
            if is_duplicate:
                return handle_duplicate(submission, "hash + AI", author, title, date, utc, status, permalink)
            
            is_duplicate, author, title, date, utc, status, permalink = check_orb_duplicate(
                submission, descriptors, new_features
            )
            if is_duplicate:
                return handle_duplicate(submission, "ORB + AI", author, title, date, utc, status, permalink)
            
            if context == "modqueue" and not submission.approved:
                rate_limiter.wait_if_needed()
                submission.mod.approve()
                print("Original submission approved")
            
            if hash_value not in image_hashes:
                image_hashes[hash_value] = (submission.id, submission.created_utc)
                orb_descriptors[submission.id] = descriptors
                ai_features[submission.id] = new_features
            
            return False
            
        except Exception as e:
            print(f"Error processing submission: {e}")
            return False

    def check_removed_original_posts():
        processed_log_items = set()
        last_checked = {}
        
        def monitor_mod_log():
            while True:
                try:
                    rate_limiter.wait_if_needed()
                    for log_entry in subreddit.mod.log(action='removelink', limit=25):
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
                            print(f"[MOD REMOVE] Submission {removed_submission_id} marked")
                        
                        if len(processed_log_items) > 1000:
                            processed_log_items.clear()
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    print(f"Error in mod log monitor: {e}")
                    time.sleep(60)
        
        threading.Thread(target=monitor_mod_log, daemon=True).start()
        
        while True:
            try:
                current_check_time = time.time()
                recent_submissions = []
                
                for hash_value, (submission_id, creation_time) in list(image_hashes.items()):
                    if hash_value in moderator_removed_hashes:
                        continue
                    
                    age = current_check_time - creation_time
                    last_check = last_checked.get(submission_id, 0)
                    
                    if age < 3600 and current_check_time - last_check >= 60:  # Check hourly posts every minute
                        recent_submissions.append((hash_value, submission_id))
                    elif age < 86400 and current_check_time - last_check >= 600:  # Check daily posts every 10 min
                        recent_submissions.append((hash_value, submission_id))
                
                for hash_value, submission_id in recent_submissions[:10]:  # Process max 10 at a time
                    try:
                        rate_limiter.wait_if_needed()
                        original_submission = reddit.submission(id=submission_id)
                        
                        if original_submission.author is None:
                            if hash_value in image_hashes:
                                del image_hashes[hash_value]
                            if submission_id in orb_descriptors:
                                del orb_descriptors[submission_id]
                            if submission_id in ai_features:
                                del ai_features[submission_id]
                            print(f"[USER DELETE] Submission {submission_id} deleted")
                        
                        last_checked[submission_id] = current_check_time
                        time.sleep(2)  # Small delay between checks
                        
                    except Exception as e:
                        print(f"Error checking submission {submission_id}: {e}")
                        last_checked[submission_id] = current_check_time
                
            except Exception as e:
                print(f"Error in check_removed_original_posts: {e}")
            
            time.sleep(30)  # Main loop delay
    
    threading.Thread(target=check_removed_original_posts, daemon=True).start()

    # Initial scan - limit to 100 to reduce initial load
    try:
        rate_limiter.wait_if_needed()
        for submission in subreddit.new(limit=100):
            if isinstance(submission, praw.models.Submission) and submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                print("Indexing submission (initial scan)")
                try:
                    img, hash_value, descriptors, features = load_and_process_image(submission.url)
                    if hash_value not in image_hashes:
                        image_hashes[hash_value] = (submission.id, submission.created_utc)
                        orb_descriptors[submission.id] = descriptors
                        ai_features[submission_id] = features
                    time.sleep(1)  # Delay between initial scans
                except Exception as e:
                    print(f"Error in initial scan: {e}")
    except Exception as e:
        print(f"Error in initial scan loop: {e}")

    # Mod Queue worker
    def modqueue_worker():
        while True:
            try:
                rate_limiter.wait_if_needed()
                modqueue_submissions = list(subreddit.mod.modqueue(only='submission', limit=50))
                modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                
                for submission in modqueue_submissions:
                    if not isinstance(submission, praw.models.Submission):
                        continue
                    
                    print("Scanning Mod Queue")
                    
                    if submission.num_reports > 0:
                        print("Skipping reported image")
                        image_hashes = {k: v for k, v in image_hashes.items() if v[0] != submission.id}
                        orb_descriptors.pop(submission.id, None)
                        ai_features.pop(submission.id, None)
                        continue
                    
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        process_submission_for_duplicates(submission, context="modqueue")
                        processed_modqueue_submissions.add(submission.id)
                    
                    time.sleep(2)  # Delay between submissions

            except Exception as e:
                print(f"Error in modqueue_worker: {e}")
            
            time.sleep(60)  # Check modqueue every minute

    threading.Thread(target=modqueue_worker, daemon=True).start()

    # Stream new submissions
    while True:
        try:
            rate_limiter.wait_if_needed()
            for submission in subreddit.stream.submissions(skip_existing=True, pause_after=5):
                if submission is None:
                    time.sleep(10)
                    continue
                    
                if submission.created_utc > current_time and isinstance(submission, praw.models.Submission):
                    if submission.id in processed_modqueue_submissions:
                        continue

                    print("Scanning new image/post")
                    
                    if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                        process_submission_for_duplicates(submission, context="stream")
                    
                    time.sleep(2)  # Delay between processing

            current_time = int(time.time())
        except Exception as e:
            print(f"Error in submission stream: {e}")
            time.sleep(10)
