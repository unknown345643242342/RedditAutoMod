import os
import time
import threading
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, Optional, Callable
from collections import defaultdict

import praw
import prawcore.exceptions
import requests
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import imagehash
import pytesseract
from openai import OpenAI


# =========================
# Configuration
# =========================
@dataclass
class BotConfig:
    """Centralized bot configuration"""
    # Reddit credentials (use environment variables in production)
    client_id: str = os.getenv('REDDIT_CLIENT_ID', 'jl-I3OHYH2_VZMC1feoJMQ')
    client_secret: str = os.getenv('REDDIT_CLIENT_SECRET', 'TCOIQBXqIskjWEbdH9i5lvoFavAJ1A')
    username: str = os.getenv('REDDIT_USERNAME', 'PokeLeakBot3')
    password: str = os.getenv('REDDIT_PASSWORD', 'testbot1')
    user_agent: str = os.getenv('REDDIT_USER_AGENT', 'testbot')
    
    # Timing constants
    rate_limit_sleep: int = 60
    modqueue_check_interval: int = 60
    spoiler_check_interval: int = 30
    reports_check_interval: int = 60
    invite_check_interval: int = 10
    modlog_check_interval: int = 30
    removal_check_interval: int = 60
    modqueue_worker_interval: int = 15
    stream_worker_interval: int = 20
    one_hour_seconds: int = 3600
    
    # Duplicate detection defaults
    default_hash_distance: int = 3
    default_hash_ai_similarity: float = 0.50
    default_orb_similarity: float = 0.50
    default_orb_ai_similarity: float = 0.75
    
    # Image processing
    initial_scan_limit: int = 15
    modqueue_limit: int = 100
    new_posts_limit: int = 10
    white_threshold: float = 0.7
    text_threshold: float = 0.05
    image_timeout: int = 10
    max_comment_retries: int = 3


# =========================
# Utility Functions
# =========================
def safe_run(target: Callable, *args, **kwargs) -> None:
    """
    Keeps a target function running forever.
    If the function raises, log the error, sleep briefly, and run it again.
    """
    while True:
        try:
            target(*args, **kwargs)
        except Exception as e:
            print(f"[FATAL] {target.__name__} crashed: {e}")
            traceback.print_exc()
            time.sleep(10)


def initialize_reddit(config: BotConfig) -> praw.Reddit:
    """Initialize Reddit API client"""
    return praw.Reddit(
        client_id=config.client_id,
        client_secret=config.client_secret,
        username=config.username,
        password=config.password,
        user_agent=config.user_agent
    )


def handle_exception(e: Exception) -> None:
    """Handle Reddit API exceptions"""
    if isinstance(e, prawcore.exceptions.ResponseException):
        if hasattr(e, 'response') and e.response.status_code == 429:
            print("Rate limited by Reddit API. Ignoring error.")


def format_age(utc_timestamp: float) -> str:
    """Format timestamp as human-readable age"""
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


# =========================
# Moderation Workers
# =========================
class ModerationWorker:
    """Base class for moderation workers"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.reddit = initialize_reddit(config)
    
    def run_with_error_handling(self, func: Callable, sleep_time: int) -> None:
        """Run a function repeatedly with error handling"""
        while True:
            try:
                func()
            except Exception as e:
                handle_exception(e)
                time.sleep(sleep_time)


class ReportedPostsMonitor(ModerationWorker):
    """Monitor and re-approve reported posts"""
    
    def monitor(self) -> None:
        subreddit = self.reddit.subreddit("PokeLeaks")
        while True:
            try:
                for post in subreddit.mod.reports():
                    if getattr(post, "approved", False):
                        post.mod.approve()
                        print(f"Post {post.id} has been approved again")
            except Exception as e:
                handle_exception(e)
                time.sleep(self.config.rate_limit_sleep)


class ModqueueHandler(ModerationWorker):
    """Handle modqueue items with timed approval"""
    
    def __init__(self, config: BotConfig):
        super().__init__(config)
        self.timers: Dict[str, float] = {}
    
    def handle(self) -> None:
        while True:
            try:
                for item in self.reddit.subreddit('PokeLeaks').mod.modqueue():
                    self._process_item(item)
            except Exception as e:
                handle_exception(e)
                time.sleep(self.config.modqueue_check_interval)
    
    def _process_item(self, item) -> None:
        """Process a single modqueue item"""
        num_reports = getattr(item, "num_reports", 0)
        
        # Start timer for single-report items
        if num_reports == 1 and item.id not in self.timers:
            created_time = getattr(item, "created_utc", time.time())
            self.timers[item.id] = time.time()
            print(f"Starting timer for post {item.id} (created {created_time})...")
        
        # Check if timer has expired
        if item.id in self.timers:
            start_time = self.timers[item.id]
            time_diff = time.time() - start_time
            
            if time_diff >= self.config.one_hour_seconds:
                try:
                    item.mod.approve()
                    print(f"Approved post {item.id} with one report")
                    del self.timers[item.id]
                except prawcore.exceptions.ServerError as se:
                    handle_exception(se)
            else:
                time_remaining = int(start_time + self.config.one_hour_seconds - time.time())
                print(f"Time remaining for post {item.id}: {time_remaining} seconds")


class SpoilerStatusHandler(ModerationWorker):
    """Handle spoiler status changes"""
    
    def __init__(self, config: BotConfig):
        super().__init__(config)
        self.previous_spoiler_status: Dict[str, bool] = {}
    
    def handle(self) -> None:
        subreddit = self.reddit.subreddit('PokeLeaks')
        
        while True:
            try:
                for submission in subreddit.new():
                    self._check_spoiler_status(submission, subreddit)
            except Exception as e:
                handle_exception(e)
                time.sleep(self.config.spoiler_check_interval)
    
    def _check_spoiler_status(self, submission, subreddit) -> None:
        """Check and enforce spoiler status"""
        if submission.id not in self.previous_spoiler_status:
            self.previous_spoiler_status[submission.id] = submission.spoiler
            return
        
        if self.previous_spoiler_status[submission.id] != submission.spoiler:
            is_moderator = self._is_moderator(submission.author, subreddit)
            
            if not submission.spoiler and not is_moderator:
                try:
                    print(f'Post {submission.id} unmarked as spoiler by non-mod. Re-marking.')
                    submission.mod.spoiler()
                except prawcore.exceptions.ServerError as se:
                    handle_exception(se)
            elif not submission.spoiler and is_moderator:
                print(f'Post {submission.id} unmarked as spoiler by a moderator. Leaving as-is.')
            
            self.previous_spoiler_status[submission.id] = submission.spoiler
    
    @staticmethod
    def _is_moderator(author, subreddit) -> bool:
        """Check if author is a moderator"""
        try:
            return author in subreddit.moderator()
        except Exception:
            return False


class ReportBasedRemovalHandler(ModerationWorker):
    """Handle automatic removal/approval based on user reports"""
    
    def __init__(self, config: BotConfig, thresholds: Dict[str, int], 
                 action: str = 'remove', content_type: str = 'comment'):
        super().__init__(config)
        self.thresholds = thresholds
        self.action = action  # 'remove' or 'approve'
        self.content_type = content_type  # 'comment' or 'submission'
    
    def handle(self) -> None:
        while True:
            try:
                for item in self.reddit.subreddit('PokeLeaks').mod.modqueue(limit=self.config.modqueue_limit):
                    self._process_reports(item)
            except Exception as e:
                handle_exception(e)
                time.sleep(self.config.reports_check_interval)
    
    def _process_reports(self, item) -> None:
        """Process user reports on an item"""
        # Type check
        if self.content_type == 'comment' and not isinstance(item, praw.models.Comment):
            return
        if self.content_type == 'submission' and not isinstance(item, praw.models.Submission):
            return
        
        user_reports = getattr(item, "user_reports", None)
        if not user_reports:
            return
        
        reason = user_reports[0][0]
        count = user_reports[0][1]
        
        if reason in self.thresholds and count >= self.thresholds[reason]:
            try:
                if self.action == 'remove':
                    item.mod.remove()
                    content = item.body if hasattr(item, 'body') else item.title
                    print(f'{self.content_type.capitalize()} "{content}" removed due to {count} reports for reason: {reason}')
                else:  # approve
                    item.mod.approve()
                    content = item.body if hasattr(item, 'body') else item.title
                    print(f'{self.content_type.capitalize()} "{content}" approved due to {count} reports for reason: {reason}')
            except prawcore.exceptions.ServerError as se:
                handle_exception(se)


# =========================
# Duplicate Detection System
# =========================
@dataclass
class DuplicateDetectionThresholds:
    """Thresholds for duplicate detection"""
    hash_distance: int = 3
    hash_ai_similarity: float = 0.50
    orb_similarity: float = 0.50
    orb_ai_similarity: float = 0.75
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'hash_distance': self.hash_distance,
            'hash_ai_similarity': self.hash_ai_similarity,
            'orb_similarity': self.orb_similarity,
            'orb_ai_similarity': self.orb_ai_similarity
        }


@dataclass
class SubredditData:
    """Data for a single subreddit"""
    subreddit: praw.models.Subreddit
    thresholds: DuplicateDetectionThresholds = field(default_factory=DuplicateDetectionThresholds)
    image_hashes: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    orb_descriptors: Dict[str, np.ndarray] = field(default_factory=dict)
    moderator_removed_hashes: Set[str] = field(default_factory=set)
    processed_modqueue_submissions: Set[str] = field(default_factory=set)
    approved_by_moderator: Set[str] = field(default_factory=set)
    ai_features: Dict[str, torch.Tensor] = field(default_factory=dict)
    current_time: int = field(default_factory=lambda: int(time.time()))
    processed_log_items: Set[str] = field(default_factory=set)
    last_checked: Dict[str, float] = field(default_factory=dict)


class ImageProcessor:
    """Handle image processing and feature extraction"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.device = "cpu"
        
        # Initialize AI model
        self.efficientnet_model = models.efficientnet_b0(pretrained=True)
        self.efficientnet_model.eval()
        self.efficientnet_model.to(self.device)
        
        self.efficientnet_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize ORB detector
        self.orb_detector = cv2.ORB_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def load_and_process_image(self, url: str) -> Tuple[np.ndarray, str, np.ndarray, torch.Tensor]:
        """Load image from URL and compute hash, descriptors, AI features"""
        image_data = requests.get(url, timeout=self.config.image_timeout).content
        img = np.asarray(bytearray(image_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hash_value = str(imagehash.phash(Image.fromarray(gray)))
        
        descriptors = self.get_orb_descriptors_conditional(img)
        features = self.get_ai_features(img)
        
        return img, hash_value, descriptors, features
    
    def get_ai_features(self, img: np.ndarray) -> Optional[torch.Tensor]:
        """Extract AI features from image"""
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_tensor = self.efficientnet_transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.efficientnet_model(img_tensor)
                feat = feat / feat.norm(dim=1, keepdim=True)
            return feat
        except Exception as e:
            print(f"AI feature extraction error: {e}")
            return None
    
    def is_problematic_image(self, img: np.ndarray) -> bool:
        """Check if image needs special processing (e.g., text-heavy)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_ratio = np.mean(gray > 240)
        
        if white_ratio > self.config.white_threshold:
            return True
        
        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = np.mean(edges > 0)
        return edge_ratio > self.config.text_threshold
    
    def preprocess_image_for_orb(self, img: np.ndarray) -> np.ndarray:
        """Preprocess problematic images for ORB detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 240, 255, cv2.THRESH_TOZERO_INV)
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    def get_orb_descriptors_conditional(self, img: np.ndarray) -> np.ndarray:
        """Get ORB descriptors with conditional preprocessing"""
        if self.is_problematic_image(img):
            processed_img = self.preprocess_image_for_orb(img)
        else:
            processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        kp, des = self.orb_detector.detectAndCompute(processed_img, None)
        return des
    
    def orb_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Calculate ORB similarity between two descriptor sets"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0
        matches = self.bf_matcher.match(desc1, desc2)
        return len(matches) / min(len(desc1), len(desc2))
    
    @staticmethod
    def calculate_ai_similarity(features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Calculate AI similarity score between two feature vectors"""
        if features1 is not None and features2 is not None:
            return (features1 @ features2.T).item()
        return 0


class DuplicateDetector:
    """Main duplicate detection logic"""
    
    def __init__(self, config: BotConfig, reddit: praw.Reddit, 
                 subreddit_data: Dict[str, SubredditData]):
        self.config = config
        self.reddit = reddit
        self.subreddit_data = subreddit_data
        self.image_processor = ImageProcessor(config)
    
    def post_duplicate_comment(self, submission, original_author: str, original_title: str,
                              original_date: str, original_utc: float, original_status: str,
                              original_permalink: str, subreddit_name: str) -> bool:
        """Post sticky comment about duplicate detection"""
        age_text = format_age(original_utc)
        
        for retry in range(self.config.max_comment_retries):
            try:
                comment_text = (
                    "> **Duplicate detected**\n\n"
                    "| Original Author | Title | Date | Age | Status |\n"
                    "|:---------------:|:-----:|:----:|:---:|:------:|\n"
                    f"| {original_author} | [{original_title}]({original_permalink}) | "
                    f"{original_date} | {age_text} | {original_status} |"
                )
                comment = submission.reply(comment_text)
                comment.mod.distinguish(sticky=True)
                print(f"[r/{subreddit_name}] Duplicate removed and comment posted: {submission.url}")
                return True
            except Exception as e:
                handle_exception(e)
                time.sleep(1)
        
        return False
    
    def check_hash_duplicate(self, submission, hash_value: str, new_features: torch.Tensor,
                            data: SubredditData, subreddit_name: str) -> Tuple[bool, ...]:
        """Check if submission is a hash-based duplicate"""
        matched_hash = None
        hash_dist_threshold = int(data.thresholds.hash_distance)
        
        for stored_hash in data.image_hashes.keys():
            if hash_value == stored_hash or \
               (imagehash.hex_to_hash(hash_value) - imagehash.hex_to_hash(stored_hash)) <= hash_dist_threshold:
                matched_hash = stored_hash
                break
        
        if matched_hash is None:
            return (False,) + (None,) * 6
        
        original_id, original_time = data.image_hashes[matched_hash]
        
        if submission.id == original_id or submission.created_utc <= original_time:
            return (False,) + (None,) * 6
        
        original_submission = self.reddit.submission(id=original_id)
        original_features = self._get_cached_ai_features(original_submission.id, data)
        
        ai_score = self.image_processor.calculate_ai_similarity(new_features, original_features)
        
        print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f}")
        
        if ai_score > data.thresholds.hash_ai_similarity:
            original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            original_status = "Removed by Moderator" if matched_hash in data.moderator_removed_hashes else "Active"
            return (True, original_submission.author.name, original_submission.title,
                   original_post_date, original_submission.created_utc, original_status,
                   original_submission.permalink)
        
        return (False,) + (None,) * 6
    
    def check_orb_duplicate(self, submission, descriptors: np.ndarray, new_features: torch.Tensor,
                           data: SubredditData, subreddit_name: str) -> Tuple[bool, ...]:
        """Check if submission is an ORB-based duplicate"""
        for old_id, old_desc in data.orb_descriptors.items():
            sim = self.image_processor.orb_similarity(descriptors, old_desc)
            
            if sim > data.thresholds.orb_similarity:
                old_features = self._get_cached_ai_features(old_id, data)
                ai_score = self.image_processor.calculate_ai_similarity(new_features, old_features)
                
                if ai_score > data.thresholds.orb_ai_similarity:
                    print(f"[r/{subreddit_name}] ORB duplicate found! AI similarity: {ai_score:.2f}")
                    
                    original_submission = self.reddit.submission(id=old_id)
                    original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                    old_hash = next((h for h, v in data.image_hashes.items() if v[0] == old_id), None)
                    original_status = "Removed by Moderator" if old_hash and old_hash in data.moderator_removed_hashes else "Active"
                    
                    return (True, original_submission.author.name, original_submission.title,
                           original_post_date, original_submission.created_utc, original_status,
                           original_submission.permalink)
        
        return (False,) + (None,) * 6
    
    def _get_cached_ai_features(self, submission_id: str, data: SubredditData) -> torch.Tensor:
        """Get AI features from cache or compute them"""
        if submission_id in data.ai_features:
            return data.ai_features[submission_id]
        
        old_submission = self.reddit.submission(id=submission_id)
        old_image_data = requests.get(old_submission.url, timeout=self.config.image_timeout).content
        old_img = cv2.imdecode(np.asarray(bytearray(old_image_data), dtype=np.uint8), cv2.IMREAD_COLOR)
        old_features = self.image_processor.get_ai_features(old_img)
        data.ai_features[submission_id] = old_features
        return old_features
    
    def process_submission(self, submission, data: SubredditData, 
                          subreddit_name: str, context: str = "stream") -> bool:
        """Main duplicate detection logic - works for both mod queue and stream"""
        try:
            img, hash_value, descriptors, new_features = self.image_processor.load_and_process_image(submission.url)
            data.ai_features[submission.id] = new_features
            
            # Check for moderator-removed repost
            if hash_value in data.moderator_removed_hashes and not submission.approved:
                submission.mod.remove()
                original_submission = self.reddit.submission(id=data.image_hashes[hash_value][0])
                self.post_duplicate_comment(
                    submission, original_submission.author.name, original_submission.title,
                    datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    original_submission.created_utc, "Removed by Moderator",
                    original_submission.permalink, subreddit_name
                )
                print(f"[r/{subreddit_name}] Repost of a moderator-removed image removed: {submission.url}")
                return True
            
            # Check hash duplicate
            is_duplicate, author, title, date, utc, status, permalink = self.check_hash_duplicate(
                submission, hash_value, new_features, data, subreddit_name
            )
            if is_duplicate and not submission.approved:
                submission.mod.remove()
                self.post_duplicate_comment(submission, author, title, date, utc, status, permalink, subreddit_name)
                print(f"[r/{subreddit_name}] Duplicate removed by hash + AI: {submission.url}")
                return True
            
            # Check ORB duplicate
            is_duplicate, author, title, date, utc, status, permalink = self.check_orb_duplicate(
                submission, descriptors, new_features, data, subreddit_name
            )
            if is_duplicate and not submission.approved:
                submission.mod.remove()
                self.post_duplicate_comment(submission, author, title, date, utc, status, permalink, subreddit_name)
                print(f"[r/{subreddit_name}] Duplicate removed by ORB + AI: {submission.url}")
                return True
            
            # Store as new original
            if hash_value not in data.image_hashes:
                data.image_hashes[hash_value] = (submission.id, submission.created_utc)
                data.orb_descriptors[submission.id] = descriptors
                data.ai_features[submission.id] = new_features
                print(f"[r/{subreddit_name}] Stored new original: {submission.url}")
            
            # Approve if from modqueue
            if context == "modqueue" and not submission.approved:
                submission.mod.approve()
                print(f"[r/{subreddit_name}] Original submission approved: {submission.url}")
            
            return False
            
        except Exception as e:
            handle_exception(e)
            return False


class DuplicateBotManager:
    """Manages the duplicate detection bot across multiple subreddits"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.reddit = initialize_reddit(config)
        self.subreddit_data: Dict[str, SubredditData] = {}
        self.subreddit_data_lock = threading.Lock()
        self.detector = DuplicateDetector(config, self.reddit, self.subreddit_data)
    
    def setup_subreddit(self, subreddit_name: str) -> None:
        """Initialize data structures for a specific subreddit"""
        original_name = subreddit_name
        subreddit_name = subreddit_name.lower()
        print(f"\n=== Setting up bot for r/{subreddit_name} ===")
        
        subreddit = self.reddit.subreddit(original_name)
        
        data = SubredditData(
            subreddit=subreddit,
            thresholds=DuplicateDetectionThresholds(
                hash_distance=self.config.default_hash_distance,
                hash_ai_similarity=self.config.default_hash_ai_similarity,
                orb_similarity=self.config.default_orb_similarity,
                orb_ai_similarity=self.config.default_orb_ai_similarity
            )
        )
        
        with self.subreddit_data_lock:
            self.subreddit_data[subreddit_name] = data
        
        print(f"[r/{subreddit_name}] Loaded thresholds: {data.thresholds.to_dict()}")
        
        # Initial scan
        self._initial_scan(subreddit, data, subreddit_name)
        
        print(f"[r/{subreddit_name}] Bot setup complete!\n")
    
    def _initial_scan(self, subreddit, data: SubredditData, subreddit_name: str) -> None:
        """Perform initial scan of recent submissions"""
        print(f"[r/{subreddit_name}] Starting initial scan...")
        
        try:
            for submission in subreddit.new(limit=self.config.initial_scan_limit):
                if isinstance(submission, praw.models.Submission) and \
                   submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                    print(f"[r/{subreddit_name}] Indexing submission (initial scan): {submission.url}")
                    try:
                        img, hash_value, descriptors, features = self.detector.image_processor.load_and_process_image(submission.url)
                        if hash_value not in data.image_hashes:
                            data.image_hashes[hash_value] = (submission.id, submission.created_utc)
                            data.orb_descriptors[submission.id] = descriptors
                            data.ai_features[submission.id] = features
                    except Exception as e:
                        handle_exception(e)
        except Exception as e:
            handle_exception(e)
        
        print(f"[r/{subreddit_name}] Initial scan complete. Indexed {len(data.image_hashes)} images.")
    
    def shared_mod_log_monitor(self) -> None:
        """Single thread monitoring mod logs for ALL subreddits"""
        while True:
            try:
                with self.subreddit_data_lock:
                    subreddits = list(self.subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = self.subreddit_data[subreddit_name]
                        subreddit = data.subreddit
                        
                        for log_entry in subreddit.mod.log(action='removelink', limit=50):
                            if log_entry.id in data.processed_log_items:
                                continue
                            
                            data.processed_log_items.add(log_entry.id)
                            removed_submission_id = log_entry.target_fullname.replace('t3_', '')
                            
                            hash_to_process = None
                            for hash_value, (submission_id, creation_time) in list(data.image_hashes.items()):
                                if submission_id == removed_submission_id:
                                    hash_to_process = hash_value
                                    break
                            
                            if hash_to_process and hash_to_process not in data.moderator_removed_hashes:
                                data.moderator_removed_hashes.add(hash_to_process)
                                print(f"[r/{subreddit_name}] [MOD REMOVE] Submission {removed_submission_id} removed by moderator. Hash kept for future duplicate detection.")
                            
                            if len(data.processed_log_items) > 1000:
                                data.processed_log_items.clear()
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in mod log check: {e}")
                
            except Exception as e:
                print(f"Error in shared mod log monitor: {e}")
            
            time.sleep(self.config.modlog_check_interval)
    
    def shared_removal_checker(self) -> None:
        """Single thread checking for removed posts across ALL subreddits"""
        while True:
            try:
                with self.subreddit_data_lock:
                    subreddits = list(self.subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = self.subreddit_data[subreddit_name]
                        current_check_time = time.time()
                        checked_this_cycle = 0
                        
                        recent_submissions = []
                        medium_submissions = []
                        old_submissions = []
                        
                        for hash_value, (submission_id, creation_time) in list(data.image_hashes.items()):
                            if hash_value in data.moderator_removed_hashes:
                                continue
                            
                            age = current_check_time - creation_time
                            last_check = data.last_checked.get(submission_id, 0)
                            
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
                                original_submission = self.reddit.submission(id=submission_id)
                                original_author = original_submission.author
                                
                                if original_author is None:
                                    if hash_value in data.image_hashes:
                                        del data.image_hashes[hash_value]
                                    if submission_id in data.orb_descriptors:
                                        del data.orb_descriptors[submission_id]
                                    if submission_id in data.ai_features:
                                        del data.ai_features[submission_id]
                                    if submission_id in data.last_checked:
                                        del data.last_checked[submission_id]
                                    print(f"[r/{subreddit_name}] [USER DELETE] Submission {submission_id} deleted by user. Hash removed.")
                                else:
                                    data.last_checked[submission_id] = current_check_time
                                
                                checked_this_cycle += 1
                                
                                if checked_this_cycle >= 10:
                                    time.sleep(60)
                                    checked_this_cycle = 0
                                
                            except Exception as e:
                                print(f"[r/{subreddit_name}] Error checking submission {submission_id}: {e}")
                                data.last_checked[submission_id] = current_check_time
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in removal check: {e}")
                
            except Exception as e:
                print(f"Error in shared removal checker: {e}")
            
            time.sleep(self.config.removal_check_interval)
    
    def shared_modqueue_worker(self) -> None:
        """Single thread processing mod queue for ALL subreddits"""
        while True:
            try:
                with self.subreddit_data_lock:
                    subreddits = list(self.subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = self.subreddit_data[subreddit_name]
                        subreddit = data.subreddit
                        
                        modqueue_submissions = subreddit.mod.modqueue(only='submission', limit=None)
                        modqueue_submissions = sorted(modqueue_submissions, key=lambda x: x.created_utc)
                        
                        for submission in modqueue_submissions:
                            if not isinstance(submission, praw.models.Submission):
                                continue
                            
                            print(f"[r/{subreddit_name}] Scanning Mod Queue: {submission.url}")
                            
                            if submission.num_reports > 0:
                                print(f"[r/{subreddit_name}] Skipping reported image: {submission.url}")
                                data.image_hashes = {k: v for k, v in data.image_hashes.items() if v[0] != submission.id}
                                data.orb_descriptors.pop(submission.id, None)
                                data.ai_features.pop(submission.id, None)
                                continue
                            
                            if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                self.detector.process_submission(submission, data, subreddit_name, context="modqueue")
                                data.processed_modqueue_submissions.add(submission.id)
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in modqueue worker: {e}")
            
            except Exception as e:
                print(f"Error in shared modqueue worker: {e}")
            
            time.sleep(self.config.modqueue_worker_interval)
    
    def shared_stream_worker(self) -> None:
        """Single thread streaming new submissions for ALL subreddits"""
        while True:
            try:
                with self.subreddit_data_lock:
                    subreddits = list(self.subreddit_data.keys())
                
                for subreddit_name in subreddits:
                    try:
                        data = self.subreddit_data[subreddit_name]
                        subreddit = data.subreddit
                        
                        for submission in subreddit.new(limit=self.config.new_posts_limit):
                            if submission.created_utc > data.current_time and isinstance(submission, praw.models.Submission):
                                if submission.id in data.processed_modqueue_submissions:
                                    continue

                                print(f"[r/{subreddit_name}] Scanning new image/post: {submission.url}")
                                
                                if submission.url.endswith(('jpg', 'jpeg', 'png', 'gif')):
                                    self.detector.process_submission(submission, data, subreddit_name, context="stream")
                        
                        data.current_time = int(time.time())
                    
                    except Exception as e:
                        print(f"[r/{subreddit_name}] Error in stream worker: {e}")
            
            except Exception as e:
                print(f"Error in shared stream worker: {e}")
            
            time.sleep(self.config.stream_worker_interval)
    
    def check_for_invites_and_messages(self) -> None:
        """Check for moderator invites and threshold adjustment messages"""
        while True:
            try:
                # Check unread messages for mod invites AND threshold commands
                for message in self.reddit.inbox.unread(limit=None):
                    # Handle mod invites
                    if "invitation to moderate" in message.subject.lower():
                        subreddit_name = message.subreddit.display_name
                        print(f"\n*** Found mod invite for r/{subreddit_name} ***")
                        try:
                            message.subreddit.mod.accept_invite()
                            print(f"✅ Accepted mod invite for r/{subreddit_name}")
                            self.setup_subreddit(subreddit_name)
                        except Exception as e:
                            print(f"Error accepting invite for r/{subreddit_name}: {e}")
                        message.mark_read()
                    
                    # Handle threshold adjustment messages
                    else:
                        self._process_threshold_message(message)
                
                # Check for already accepted subreddits
                for subreddit in self.reddit.user.moderator_subreddits(limit=None):
                    subreddit_name = subreddit.display_name.lower()
                    if subreddit_name not in self.subreddit_data:
                        print(f"\n*** Already moderating r/{subreddit_name}, setting up bot ***")
                        self.setup_subreddit(subreddit_name)
            
            except Exception as e:
                print(f"Error checking for invites and messages: {e}")
        
            time.sleep(self.config.invite_check_interval)
    
    def _process_threshold_message(self, message) -> None:
        """Process threshold adjustment commands"""
        try:
            body = message.body.strip()
            body_lower = body.lower()
            author = message.author.name
            
            if '!showthresholds' in body_lower:
                self._handle_show_thresholds(message, body_lower, author)
            elif '!resetthresholds' in body_lower:
                self._handle_reset_thresholds(message, body_lower, author)
            elif '!setthreshold' in body_lower:
                self._handle_set_threshold(message, body, body_lower, author)
        
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def _handle_show_thresholds(self, message, body_lower: str, author: str) -> None:
        """Handle !showthresholds command"""
        parts = body_lower.split()
        
        if len(parts) >= 2 and parts[1].startswith('r/'):
            target_subreddit = parts[1].replace('r/', '').lower()
            
            if target_subreddit in self.subreddit_data:
                target_data = self.subreddit_data[target_subreddit]
                try:
                    if target_data.subreddit.moderator(author):
                        thresholds = target_data.thresholds
                        response = f"""**Current Threshold Settings for r/{target_subreddit}:**

- hash_distance: {thresholds.hash_distance}
- hash_ai_similarity: {thresholds.hash_ai_similarity}
- orb_similarity: {thresholds.orb_similarity}
- orb_ai_similarity: {thresholds.orb_ai_similarity}

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
            moderated_subs = self._get_moderated_subs(author)
            if moderated_subs:
                subs_list = ', '.join([f"r/{s}" for s in moderated_subs])
                message.reply(f"❌ Please specify a subreddit.\n\nUsage: `!showthresholds r/subredditname`\n\nYou moderate: {subs_list}")
            else:
                message.reply(f"❌ You do not moderate any subreddits where this bot is running.")
        
        message.mark_read()
    
    def _handle_reset_thresholds(self, message, body_lower: str, author: str) -> None:
        """Handle !resetthresholds command"""
        parts = body_lower.split()
        
        if len(parts) >= 2 and parts[1].startswith('r/'):
            target_subreddit = parts[1].replace('r/', '').lower()
            
            if target_subreddit in self.subreddit_data:
                target_data = self.subreddit_data[target_subreddit]
                try:
                    if target_data.subreddit.moderator(author):
                        target_data.thresholds = DuplicateDetectionThresholds(
                            hash_distance=self.config.default_hash_distance,
                            hash_ai_similarity=self.config.default_hash_ai_similarity,
                            orb_similarity=self.config.default_orb_similarity,
                            orb_ai_similarity=self.config.default_orb_ai_similarity
                        )
                        message.reply(f"✅ Thresholds reset to defaults for r/{target_subreddit}")
                        print(f"[r/{target_subreddit}] Thresholds reset to defaults by {author}")
                        print(f"[r/{target_subreddit}] Current thresholds: {target_data.thresholds.to_dict()}")
                    else:
                        message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                except:
                    message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
            else:
                message.reply(f"❌ Bot is not running on r/{target_subreddit}")
        else:
            moderated_subs = self._get_moderated_subs(author)
            if moderated_subs:
                subs_list = ', '.join([f"r/{s}" for s in moderated_subs])
                message.reply(f"❌ Please specify a subreddit.\n\nUsage: `!resetthresholds r/subredditname`\n\nYou moderate: {subs_list}")
            else:
                message.reply(f"❌ You do not moderate any subreddits where this bot is running.")
        
        message.mark_read()
    
    def _handle_set_threshold(self, message, body: str, body_lower: str, author: str) -> None:
        """Handle !setthreshold command"""
        parts = body.split()
        
        if len(parts) >= 4 and parts[1].lower().startswith('r/'):
            target_subreddit = parts[1].lower().replace('r/', '')
            param = parts[2].lower()
            
            try:
                value = float(parts[3])
                
                if target_subreddit in self.subreddit_data:
                    target_data = self.subreddit_data[target_subreddit]
                    try:
                        if target_data.subreddit.moderator(author):
                            if hasattr(target_data.thresholds, param):
                                old_value = getattr(target_data.thresholds, param)
                                setattr(target_data.thresholds, param, value)
                                message.reply(f"✅ Updated `{param}` from `{old_value}` to `{value}` for r/{target_subreddit}")
                                print(f"[r/{target_subreddit}] Threshold {param} updated: {old_value} → {value} by {author}")
                                print(f"[r/{target_subreddit}] Current thresholds: {target_data.thresholds.to_dict()}")
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
            moderated_subs = self._get_moderated_subs(author)
            if moderated_subs:
                subs_list = ', '.join([f"r/{s}" for s in moderated_subs])
                message.reply(f"❌ Usage: `!setthreshold r/subredditname <parameter> <value>`\n\nValid parameters: hash_distance, hash_ai_similarity, orb_similarity, orb_ai_similarity\n\nYou moderate: {subs_list}")
            else:
                message.reply(f"❌ You do not moderate any subreddits where this bot is running.")
        
        message.mark_read()
    
    def _get_moderated_subs(self, author: str) -> list:
        """Get list of subreddits where author is a moderator"""
        moderated_subs = []
        for sub_name, sub_data in self.subreddit_data.items():
            try:
                if sub_data.subreddit.moderator(author):
                    moderated_subs.append(sub_name)
            except:
                continue
        return moderated_subs
    
    def start_workers(self) -> None:
        """Start all worker threads"""
        threading.Thread(target=self.check_for_invites_and_messages, daemon=True).start()
        threading.Thread(target=self.shared_mod_log_monitor, daemon=True).start()
        threading.Thread(target=self.shared_removal_checker, daemon=True).start()
        threading.Thread(target=self.shared_modqueue_worker, daemon=True).start()
        threading.Thread(target=self.shared_stream_worker, daemon=True).start()
        
        print("=== Multi-subreddit duplicate bot started ===")
        print("Running with 5 shared worker threads for all subreddits")
        print("Monitoring for mod invites and threshold adjustment messages...")
        
        # Keep main thread alive
        while True:
            time.sleep(10)


# =========================
# Main Entry Point
# =========================
def main():
    """Main entry point for the bot"""
    config = BotConfig()
    threads = {}

    def add_thread(name: str, func: Callable, *args, **kwargs):
        """Add and start a thread with crash protection"""
        t = threading.Thread(target=safe_run, args=(func,) + args, kwargs=kwargs, daemon=True)
        t.start()
        threads[name] = t
        print(f"[STARTED] {name}")

    # Start duplicate detection bot
    duplicate_bot = DuplicateBotManager(config)
    add_thread('run_pokemon_duplicate_bot_thread', duplicate_bot.start_workers)
    
    # Start moderation workers
    add_thread('modqueue_thread', ModqueueHandler(config).handle)
    add_thread('reported_posts_thread', ReportedPostsMonitor(config).monitor)
    add_thread('spoiler_status_thread', SpoilerStatusHandler(config).handle)
    
    # Comment removal based on reports
    comment_removal_thresholds = {
        'No Linking to Downloadable Content in Posts or Comments': 1,
        'No ROMs, ISOs, or Game Files Sharing or Requests': 1,
        'No insults or harassment of other subreddit members in the comments': 1
    }
    add_thread('user_reports_removal_thread', 
               ReportBasedRemovalHandler(config, comment_removal_thresholds, 'remove', 'comment').handle)
    
    # Submission approval based on reports
    submission_approval_thresholds = {
        'This is misinformation': 1,
        'This is spam': 1
    }
    add_thread('submissions_based_on_user_reports_thread',
               ReportBasedRemovalHandler(config, submission_approval_thresholds, 'approve', 'submission').handle)
    
    # Post removal based on reports
    post_removal_thresholds = {
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
    add_thread('posts_based_on_removal_thread',
               ReportBasedRemovalHandler(config, post_removal_thresholds, 'remove', 'submission').handle)
    
    # Comment approval based on reports
    comment_approval_thresholds = {
        'This is misinformation': 1,
        'This is spam': 1
    }
    add_thread('comments_based_on_approval_thread',
               ReportBasedRemovalHandler(config, comment_approval_thresholds, 'approve', 'comment').handle)

    # Keep the main thread alive indefinitely so daemon threads keep running
    print("\n=== All bot threads started successfully ===")
    print("Bot is now running. Press Ctrl+C to stop.\n")
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
