import praw
import prawcore.exceptions
import requests
import time
from datetime import datetime
import numpy as np
from PIL import Image
import imagehash
import cv2
import threading
import traceback
import pytesseract
import openai
from openai import OpenAI
from torchvision import transforms
import torch
import torchvision.models as models
import torchvision.transforms as T
import hashlib
import difflib as _difflib
from datetime import datetime, timezone
import json
import os

# File to store thresholds
THRESHOLDS_FILE = "bot_thresholds.json"
import json
import os as _difflib
from datetime import datetime, timezone

# =========================
# Global subreddit tracking
# =========================
monitored_subreddits = set()
monitored_subreddits_lock = threading.Lock()

# =========================
# Custom thresholds per subreddit
# =========================
subreddit_thresholds = {
    'comment_removal': {},  # {subreddit_name: {rule: threshold}}
    'submission_approval': {},  # {subreddit_name: {rule: threshold}}
    'submission_removal': {},  # {subreddit_name: {rule: threshold}}
    'comment_approval': {}  # {subreddit_name: {rule: threshold}}
}
thresholds_lock = threading.Lock()

# =========================
# Modqueue timer settings per subreddit
# =========================
modqueue_timers = {}  # {subreddit_name: seconds}
modqueue_timers_lock = threading.Lock()
DEFAULT_MODQUEUE_TIMER = 3600  # 1 hour default

# =========================
# Crash-proof runner
# =========================
def safe_run(target, *args, **kwargs):
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
            time.sleep(10)  # brief cooldown before retrying

# =========================
# Reddit init + error handler
# =========================
def initialize_reddit():
    return praw.Reddit(
        client_id='jl-I3OHYH2_VZMC1feoJMQ',
        client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
        username='PokeLeakBot3',
        password='testbot1',
        user_agent='testbot'
    )

def handle_exception(e):
    if isinstance(e, prawcore.exceptions.ResponseException) and getattr(e, "response", None) and e.response.status_code == 429:
        print("Rate limited by Reddit API. Ignoring error.")

# =========================
# Global Invite Checker
# =========================
def check_for_invites():
    """Check for moderator invites and automatically accept them for all bot functions"""
    reddit = initialize_reddit()
    
    while True:
        try:
            # Check unread messages for mod invites
            for message in reddit.inbox.unread(limit=None):
                if "invitation to moderate" in message.subject.lower():
                    subreddit_name = message.subreddit.display_name.lower()  # Normalize to lowercase
                    print(f"\n*** Found mod invite for r/{subreddit_name} ***")
                    try:
                        message.subreddit.mod.accept_invite()
                        print(f"✅ Accepted mod invite for r/{subreddit_name}")
                        
                        with monitored_subreddits_lock:
                            monitored_subreddits.add(subreddit_name)
                        
                        # Initialize default thresholds for new subreddit
                        initialize_default_thresholds(subreddit_name)
                        
                        print(f"Added r/{subreddit_name} to monitored subreddits")
                    except Exception as e:
                        print(f"Error accepting invite for r/{subreddit_name}: {e}")
                    message.mark_read()
        
            # Also check for already accepted subreddits
            for subreddit in reddit.user.moderator_subreddits(limit=None):
                subreddit_name = subreddit.display_name.lower()  # Normalize to lowercase
                with monitored_subreddits_lock:
                    if subreddit_name not in monitored_subreddits:
                        print(f"\n*** Already moderating r/{subreddit_name}, adding to monitoring ***")
                        monitored_subreddits.add(subreddit_name)
                        initialize_default_thresholds(subreddit_name)
        
        except Exception as e:
            print(f"Error checking for invites: {e}")
            handle_exception(e)
    
        time.sleep(60)

# =========================
# Threshold Management
# =========================
def save_thresholds():
    """Save thresholds to file"""
    try:
        with thresholds_lock:
            with open(THRESHOLDS_FILE, 'w') as f:
                json.dump(subreddit_thresholds, f, indent=2)
        print(f"[SYSTEM] Thresholds saved to {THRESHOLDS_FILE}")
    except Exception as e:
        print(f"[SYSTEM] Error saving thresholds: {e}")

def load_thresholds():
    """Load thresholds from file"""
    global subreddit_thresholds
    try:
        if os.path.exists(THRESHOLDS_FILE):
            with open(THRESHOLDS_FILE, 'r') as f:
                loaded_thresholds = json.load(f)
            
            with thresholds_lock:
                subreddit_thresholds = loaded_thresholds
            
            print(f"[SYSTEM] ═══════════════════════════════════════")
            print(f"[SYSTEM] Loaded thresholds from {THRESHOLDS_FILE}")
            print(f"[SYSTEM] ═══════════════════════════════════════")
            for category in ['comment_removal', 'submission_approval', 'submission_removal', 'comment_approval']:
                for subreddit, rules in subreddit_thresholds.get(category, {}).items():
                    if rules:
                        print(f"[SYSTEM] r/{subreddit} - {category}: {len(rules)} rules")
            print(f"[SYSTEM] ═══════════════════════════════════════\n")
        else:
            print(f"[SYSTEM] No saved thresholds found. Creating new file.")
            # Create initial empty file
            save_thresholds()
    except Exception as e:
        print(f"[SYSTEM] Error loading thresholds: {e}")
        print(f"[SYSTEM] Creating new thresholds file...")
        save_thresholds()

def initialize_default_thresholds(subreddit_name):
    """Initialize thresholds based on subreddit's actual rules"""
    reddit = initialize_reddit()
    
    # Normalize subreddit name to lowercase
    subreddit_name = subreddit_name.lower()
    
    with thresholds_lock:
        # Only initialize if not already loaded from file
        if subreddit_name not in subreddit_thresholds['comment_removal']:
            subreddit_thresholds['comment_removal'][subreddit_name] = {}
            subreddit_thresholds['submission_approval'][subreddit_name] = {}
            subreddit_thresholds['submission_removal'][subreddit_name] = {}
            subreddit_thresholds['comment_approval'][subreddit_name] = {}
            
            # Save to file after adding new subreddit
            save_thresholds()
    
    try:
        subreddit = reddit.subreddit(subreddit_name)
        rules = list(subreddit.rules)
        
        print(f"\n[r/{subreddit_name}] ═══════════════════════════════════════")
        print(f"[r/{subreddit_name}] Found {len(rules)} subreddit rules:")
        print(f"[r/{subreddit_name}] ═══════════════════════════════════════")
        
        for idx, rule in enumerate(rules, 1):
            print(f"[r/{subreddit_name}] {idx}. {rule.short_name}")
        
        print(f"[r/{subreddit_name}] ═══════════════════════════════════════")
        print(f"[r/{subreddit_name}] Rules loaded. Moderators can now set thresholds using:")
        print(f"[r/{subreddit_name}]   !setthreshold r/{subreddit_name} <category> \"<rule_name>\" <value>")
        print(f"[r/{subreddit_name}] ═══════════════════════════════════════\n")
        
    except Exception as e:
        print(f"[r/{subreddit_name}] Error loading subreddit rules: {e}")
        print(f"[r/{subreddit_name}] Initialized with empty thresholds (moderators can configure via commands)\n")

def handle_threshold_configuration():
    """Monitor messages for threshold configuration commands"""
    reddit = initialize_reddit()
    
    while True:
        try:
            for message in reddit.inbox.unread(limit=None):
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
                            
                            # Verify user is a mod
                            try:
                                subreddit = reddit.subreddit(target_subreddit)
                                if subreddit.moderator(author):
                                    # Get subreddit rules for reference
                                    subreddit_rules = []
                                    try:
                                        subreddit_rules = [rule.violation_reason for rule in subreddit.rules]
                                    except:
                                        pass
                                    
                                    with thresholds_lock:
                                        response = f"## Threshold Configuration - r/{target_subreddit}\n\n"
                                        
                                        if subreddit_rules:
                                            response += f"**Subreddit has {len(subreddit_rules)} rules** (use rule names or violation reasons when setting thresholds)\n\n"
                                        
                                        # Comment Removal
                                        response += "### 🔴 Comment Removal\n"
                                        comment_removal = subreddit_thresholds['comment_removal'].get(target_subreddit, {})
                                        if comment_removal:
                                            for rule, threshold in comment_removal.items():
                                                response += f"- **{rule}** → `{threshold} report{'s' if threshold != 1 else ''}`\n"
                                        else:
                                            response += "*No rules configured*\n"
                                        response += "\n"
                                        
                                        # Submission Approval
                                        response += "### 🟢 Submission Approval\n"
                                        submission_approval = subreddit_thresholds['submission_approval'].get(target_subreddit, {})
                                        if submission_approval:
                                            for rule, threshold in submission_approval.items():
                                                response += f"- **{rule}** → `{threshold} report{'s' if threshold != 1 else ''}`\n"
                                        else:
                                            response += "*No rules configured*\n"
                                        response += "\n"
                                        
                                        # Submission Removal
                                        response += "### 🔴 Submission Removal\n"
                                        submission_removal = subreddit_thresholds['submission_removal'].get(target_subreddit, {})
                                        if submission_removal:
                                            for rule, threshold in submission_removal.items():
                                                response += f"- **{rule}** → `{threshold} report{'s' if threshold != 1 else ''}`\n"
                                        else:
                                            response += "*No rules configured*\n"
                                        response += "\n"
                                        
                                        # Comment Approval
                                        response += "### 🟢 Comment Approval\n"
                                        comment_approval = subreddit_thresholds['comment_approval'].get(target_subreddit, {})
                                        if comment_approval:
                                            for rule, threshold in comment_approval.items():
                                                response += f"- **{rule}** → `{threshold} report{'s' if threshold != 1 else ''}`\n"
                                        else:
                                            response += "*No rules configured*\n"
                                        
                                        response += "\n---\n\n**Available Commands:**\n"
                                        response += f"- Set: `!setthreshold r/{target_subreddit} <category> \"<rule>\" <value>`\n"
                                        response += f"- Remove: `!removethreshold r/{target_subreddit} <category> \"<rule>\"`\n"
                                        response += f"- Reset: `!resetthresholds r/{target_subreddit}`\n"
                                        response += f"- List rules: `!listrules r/{target_subreddit}`"
                                    
                                    message.reply(response)
                                    print(f"[r/{target_subreddit}] Showed thresholds to {author}")
                                else:
                                    message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                            except:
                                message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                        else:
                            # No subreddit specified - show usage
                            with monitored_subreddits_lock:
                                moderated_subs = []
                                for sub_name in monitored_subreddits:
                                    try:
                                        subreddit = reddit.subreddit(sub_name)
                                        if subreddit.moderator(author):
                                            moderated_subs.append(sub_name)
                                    except:
                                        continue
                                
                                if moderated_subs:
                                    subs_list = ', '.join([f"r/{s}" for s in moderated_subs])
                                    message.reply(f"❌ Please specify a subreddit.\n\nUsage: `!showthresholds r/subredditname`\n\nYou moderate: {subs_list}")
                                else:
                                    message.reply(f"❌ You do not moderate any subreddits where this bot is running.")
                        
                        message.mark_read()
                    
                    # List subreddit rules
                    elif '!listrules' in body_lower:
                        parts = body_lower.split()
                        
                        if len(parts) >= 2 and parts[1].startswith('r/'):
                            target_subreddit = parts[1].replace('r/', '')
                            
                            try:
                                subreddit = reddit.subreddit(target_subreddit)
                                if subreddit.moderator(author):
                                    rules = list(subreddit.rules)
                                    
                                    response = f"## Subreddit Rules - r/{target_subreddit}\n\n"
                                    response += f"**Total Rules:** {len(rules)}\n\n"
                                    
                                    for idx, rule in enumerate(rules, 1):
                                        response += f"**{idx}. {rule.short_name}**\n"
                                        response += f"   *Violation Reason:* `{rule.violation_reason}`\n"
                                        if hasattr(rule, 'description') and rule.description:
                                            desc_preview = rule.description[:150] + "..." if len(rule.description) > 150 else rule.description
                                            response += f"   *Description:* {desc_preview}\n"
                                        response += "\n"
                                    
                                    response += "---\n\n"
                                    response += "Use the **rule name** or **violation reason** when setting thresholds:\n"
                                    response += f"`!setthreshold r/{target_subreddit} <category> \"<rule_name>\" <value>`"
                                    
                                    message.reply(response)
                                    print(f"[r/{target_subreddit}] Listed rules for {author}")
                                else:
                                    message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                            except Exception as e:
                                message.reply(f"❌ Error fetching rules: {str(e)}")
                        else:
                            message.reply("❌ Usage: `!listrules r/subredditname`")
                        
                        message.mark_read()
                    
                    # List subreddit rules
                    elif '!listrules' in body_lower:
                        parts = body_lower.split()
                        
                        if len(parts) >= 2 and parts[1].startswith('r/'):
                            target_subreddit = parts[1].replace('r/', '')
                            
                            try:
                                subreddit = reddit.subreddit(target_subreddit)
                                if subreddit.moderator(author):
                                    rules = list(subreddit.rules)
                                    
                                    response = "**Subreddit Rules - r/" + target_subreddit + "**\n\n"
                                    response += "Total Rules: " + str(len(rules)) + "\n\n"
                                    
                                    for idx, rule in enumerate(rules, 1):
                                        rule_name = str(rule.short_name)
                                        response += str(idx) + ". " + rule_name + "\n"
                                    
                                    response += "\n---\n\n"
                                    response += "Use the rule name when setting thresholds:\n"
                                    response += "!setthreshold r/" + target_subreddit + " <category> \"<rule_name>\" <value>"
                                    
                                    message.reply(response)
                                    print(f"[r/{target_subreddit}] Listed rules for {author}")
                                else:
                                    message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                            except Exception as e:
                                message.reply(f"❌ Error fetching rules: {str(e)}")
                                print(f"Error in !listrules for r/{target_subreddit}: {e}")
                        else:
                            message.reply("❌ Usage: !listrules r/subredditname")
                        
                        message.mark_read()
                    
                    # Reset thresholds for a specific subreddit
                    elif '!resetthresholds' in body_lower:
                        parts = body_lower.split()
                        
                        # Check if subreddit was specified: !resetthresholds r/pokemon
                        if len(parts) >= 2 and parts[1].startswith('r/'):
                            target_subreddit = parts[1].replace('r/', '')
                            
                            # Verify user is a mod
                            try:
                                subreddit = reddit.subreddit(target_subreddit)
                                if subreddit.moderator(author):
                                    initialize_default_thresholds(target_subreddit)
                                    
                                    # Show the rules again
                                    try:
                                        rules = list(subreddit.rules)
                                        response = f"✅ Thresholds reset for r/{target_subreddit}\n\n"
                                        response += f"**Available Rules ({len(rules)}):**\n"
                                        for rule in rules:
                                            response += f"- {rule.violation_reason}\n"
                                        response += f"\nUse `!setthreshold r/{target_subreddit} <category> \"<rule>\" <value>` to configure thresholds"
                                        message.reply(response)
                                    except:
                                        message.reply(f"✅ Thresholds reset to defaults for r/{target_subreddit}")
                                    
                                    print(f"[r/{target_subreddit}] Thresholds reset by {author}")
                                else:
                                    message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                            except:
                                message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                        else:
                            # No subreddit specified - show usage
                            with monitored_subreddits_lock:
                                moderated_subs = []
                                for sub_name in monitored_subreddits:
                                    try:
                                        subreddit = reddit.subreddit(sub_name)
                                        if subreddit.moderator(author):
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
                    # Format: !setthreshold r/pokemon comment_removal "No insults" 2
                    elif '!setthreshold' in body_lower:
                        # Use original body for case-sensitive parsing
                        parts = body.split(maxsplit=3)
                        
                        # Expected format: !setthreshold r/pokemon comment_removal "Rule Name" 2
                        if len(parts) >= 4 and parts[1].lower().startswith('r/'):
                            target_subreddit = parts[1].lower().replace('r/', '')  # Normalize to lowercase
                            category = parts[2].lower()
                            
                            # Parse the rest to extract rule name and value
                            # Handle quoted rule names
                            rest = parts[3]
                            
                            # Try to find quoted rule name
                            if '"' in rest:
                                quote_parts = rest.split('"')
                                if len(quote_parts) >= 3:
                                    rule_name = quote_parts[1]
                                    value_str = quote_parts[2].strip()
                                else:
                                    message.reply("❌ Invalid format. Rule name must be in quotes.")
                                    message.mark_read()
                                    continue
                            else:
                                # No quotes - split on last space for value
                                last_space = rest.rfind(' ')
                                if last_space > 0:
                                    rule_name = rest[:last_space].strip()
                                    value_str = rest[last_space+1:].strip()
                                else:
                                    message.reply("❌ Invalid format. Please provide both rule name and threshold value.")
                                    message.mark_read()
                                    continue
                            
                            try:
                                value = int(value_str)
                                
                                # Verify category is valid
                                valid_categories = ['comment_removal', 'submission_approval', 'submission_removal', 'comment_approval']
                                if category not in valid_categories:
                                    message.reply(f"❌ Invalid category: `{category}`\n\nValid categories: comment_removal, submission_approval, submission_removal, comment_approval")
                                    message.mark_read()
                                    continue
                                
                                # Verify user is a mod
                                try:
                                    subreddit = reddit.subreddit(target_subreddit)
                                    if subreddit.moderator(author):
                                        with thresholds_lock:
                                            # Check if rule already exists
                                            current_thresholds = subreddit_thresholds[category].get(target_subreddit, {})
                                            old_value = current_thresholds.get(rule_name, 'not set')
                                            
                                            # Update or add the rule
                                            if target_subreddit not in subreddit_thresholds[category]:
                                                subreddit_thresholds[category][target_subreddit] = {}
                                            
                                            subreddit_thresholds[category][target_subreddit][rule_name] = value
                                            
                                            # Debug: Print current state
                                            print(f"[r/{target_subreddit}] DEBUG: Updated thresholds for {category}:")
                                            print(f"[r/{target_subreddit}] DEBUG: {subreddit_thresholds[category][target_subreddit]}")
                                            print(f"[r/{target_subreddit}] DEBUG: Full threshold structure:")
                                            for cat_name, subs in subreddit_thresholds.items():
                                                print(f"[r/{target_subreddit}] DEBUG:   {cat_name}:")
                                                for sub_name, rules in subs.items():
                                                    print(f"[r/{target_subreddit}] DEBUG:     {sub_name}: {rules}")
                                        
                                        # Save to file after updating
                                        save_thresholds()
                                        
                                        message.reply(f"✅ Updated `{rule_name}` in {category} from `{old_value}` to `{value}` for r/{target_subreddit}")
                                        print(f"[r/{target_subreddit}] Threshold updated: {category}/{rule_name} = {value} by {author}")
                                    else:
                                        message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                                except:
                                    message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                            
                            except ValueError:
                                message.reply(f"❌ Invalid threshold value. Please provide a number.")
                        else:
                            # Invalid format - show usage
                            with monitored_subreddits_lock:
                                moderated_subs = []
                                for sub_name in monitored_subreddits:
                                    try:
                                        subreddit = reddit.subreddit(sub_name)
                                        if subreddit.moderator(author):
                                            moderated_subs.append(sub_name)
                                    except:
                                        continue
                                
                                if moderated_subs:
                                    subs_list = ', '.join([f"r/{s}" for s in moderated_subs])
                                    message.reply(f"❌ Usage: `!setthreshold r/subredditname <category> \"<rule name>\" <value>`\n\nValid categories: comment_removal, submission_approval, submission_removal, comment_approval\n\nYou moderate: {subs_list}")
                                else:
                                    message.reply(f"❌ You do not moderate any subreddits where this bot is running.")
                        
                        message.mark_read()
                    
                    # Remove a threshold rule
                    elif '!removethreshold' in body_lower:
                        # Format: !removethreshold r/pokemon comment_removal "Rule Name"
                        parts = body.split(maxsplit=2)
                        
                        if len(parts) >= 3 and parts[1].lower().startswith('r/'):
                            target_subreddit = parts[1].lower().replace('r/', '')
                            category = parts[2].lower().split()[0]
                            
                            # Parse rule name (may be quoted)
                            rest = ' '.join(parts[2].split()[1:])
                            if '"' in rest:
                                quote_parts = rest.split('"')
                                if len(quote_parts) >= 2:
                                    rule_name = quote_parts[1]
                                else:
                                    message.reply("❌ Invalid format. Rule name must be in quotes.")
                                    message.mark_read()
                                    continue
                            else:
                                rule_name = rest.strip()
                            
                            # Verify category is valid
                            valid_categories = ['comment_removal', 'submission_approval', 'submission_removal', 'comment_approval']
                            if category not in valid_categories:
                                message.reply(f"❌ Invalid category: `{category}`\n\nValid categories: comment_removal, submission_approval, submission_removal, comment_approval")
                                message.mark_read()
                                continue
                            
                            # Verify user is a mod
                            try:
                                subreddit = reddit.subreddit(target_subreddit)
                                if subreddit.moderator(author):
                                    with thresholds_lock:
                                        if target_subreddit in subreddit_thresholds[category]:
                                            if rule_name in subreddit_thresholds[category][target_subreddit]:
                                                del subreddit_thresholds[category][target_subreddit][rule_name]
                                                message.reply(f"✅ Removed `{rule_name}` from {category} for r/{target_subreddit}")
                                                print(f"[r/{target_subreddit}] Threshold removed: {category}/{rule_name} by {author}")
                                            else:
                                                message.reply(f"❌ Rule `{rule_name}` not found in {category}")
                                        else:
                                            message.reply(f"❌ No rules configured for {category} in r/{target_subreddit}")
                                else:
                                    message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                            except:
                                message.reply(f"❌ You are not a moderator of r/{target_subreddit}")
                        else:
                            message.reply("❌ Usage: `!removethreshold r/subredditname <category> \"<rule name>\"`")
                        
                        message.mark_read()
                
                except Exception as e:
                    print(f"Error processing message: {e}")
        
        except Exception as e:
            print(f"Error in threshold configuration handler: {e}")
            handle_exception(e)
        
        time.sleep(10)

# =========================
# Helper function for multi-subreddit support
# =========================
def get_monitored_subreddit_string():
    """Returns a '+' joined string of all monitored subreddits for multi-reddit queries"""
    with monitored_subreddits_lock:
        if not monitored_subreddits:
            return "PokeLeaks"  # Default fallback
        return "+".join(monitored_subreddits)

# =========================
# Workers (updated for multi-subreddit support)
# =========================
def monitor_reported_posts():
    reddit = initialize_reddit()
    while True:
        try:
            subreddit_string = get_monitored_subreddit_string()
            subreddit = reddit.subreddit(subreddit_string)
            for post in subreddit.mod.reports():
                # If already approved previously, re-approve (idempotent)
                if getattr(post, "approved", False):
                    post.mod.approve()
                    print(f"[r/{post.subreddit.display_name}] Post {post.id} has been approved again")
        except Exception as e:
            handle_exception(e)
            time.sleep(60)

def handle_modqueue_items():
    reddit = initialize_reddit()
    timers = {}

    while True:
        try:
            subreddit_string = get_monitored_subreddit_string()
            for item in reddit.subreddit(subreddit_string).mod.modqueue():
                if getattr(item, "num_reports", 0) == 1 and item.id not in timers:
                    created_time = getattr(item, "created_utc", time.time())
                    timers[item.id] = time.time()
                    print(f"[r/{item.subreddit.display_name}] Starting timer for post {item.id} (created {created_time})...")

                if item.id in timers:
                    start_time = timers[item.id]
                    time_diff = time.time() - start_time
                    if time_diff >= 3600:
                        try:
                            item.mod.approve()
                            print(f"[r/{item.subreddit.display_name}] Approved post {item.id} with one report")
                            del timers[item.id]
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
                    else:
                        new_reports = getattr(item, "report_reasons", None)
                        if new_reports != getattr(item, "report_reasons", None):
                            print(f"[r/{item.subreddit.display_name}] New reports for post {item.id}, leaving post in mod queue")
                            del timers[item.id]
                        else:
                            time_remaining = int(start_time + 3600 - time.time())
                            print(f"[r/{item.subreddit.display_name}] Time remaining for post {item.id}: {time_remaining} seconds")
        except Exception as e:
            handle_exception(e)
            time.sleep(60)

def handle_spoiler_status():
    reddit = initialize_reddit()
    previous_spoiler_status = {}

    while True:
        try:
            subreddit_string = get_monitored_subreddit_string()
            subreddit = reddit.subreddit(subreddit_string)
            
            for submission in subreddit.new():
                if submission.id not in previous_spoiler_status:
                    previous_spoiler_status[submission.id] = submission.spoiler
                    continue

                if previous_spoiler_status[submission.id] != submission.spoiler:
                    # Check if the change was made by a moderator
                    try:
                        is_moderator = submission.author in submission.subreddit.moderator()
                    except Exception:
                        is_moderator = False

                    if not submission.spoiler:
                        if not is_moderator:
                            try:
                                print(f'[r/{submission.subreddit.display_name}] Post {submission.id} unmarked as spoiler by non-mod. Re-marking.')
                                submission.mod.spoiler()
                            except prawcore.exceptions.ServerError as se:
                                handle_exception(se)
                        else:
                            print(f'[r/{submission.subreddit.display_name}] Post {submission.id} unmarked as spoiler by a moderator. Leaving as-is.')
                    previous_spoiler_status[submission.id] = submission.spoiler
        except Exception as e:
            handle_exception(e)
            time.sleep(30)

def handle_user_reports_and_removal():
    reddit = initialize_reddit()

    while True:
        try:
            subreddit_string = get_monitored_subreddit_string()
            for comment in reddit.subreddit(subreddit_string).mod.modqueue(limit=100):
                if isinstance(comment, praw.models.Comment) and getattr(comment, "user_reports", None):
                    subreddit_name = comment.subreddit.display_name
                    
                    # Get thresholds for this subreddit
                    with thresholds_lock:
                        thresholds = subreddit_thresholds['comment_removal'].get(subreddit_name, {})
                    
                    if not thresholds:
                        continue
                    
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            comment.mod.remove()
                            print(f'[r/{subreddit_name}] Comment "{comment.body}" removed due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            time.sleep(60)

def handle_submissions_based_on_user_reports():
    reddit = initialize_reddit()

    while True:
        try:
            subreddit_string = get_monitored_subreddit_string()
            for post in reddit.subreddit(subreddit_string).mod.modqueue(limit=100):
                if isinstance(post, praw.models.Submission) and getattr(post, "user_reports", None):
                    subreddit_name = post.subreddit.display_name
                    
                    # Get thresholds for this subreddit
                    with thresholds_lock:
                        thresholds = subreddit_thresholds['submission_approval'].get(subreddit_name, {})
                    
                    if not thresholds:
                        continue
                    
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            post.mod.approve()
                            print(f'[r/{subreddit_name}] post "{post.title}" approved due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            time.sleep(60)

def handle_posts_based_on_removal():
    reddit = initialize_reddit()

    while True:
        try:
            subreddit_string = get_monitored_subreddit_string()
            for post in reddit.subreddit(subreddit_string).mod.modqueue(limit=100):
                if isinstance(post, praw.models.Submission) and getattr(post, "user_reports", None):
                    subreddit_name = post.subreddit.display_name.lower()  # Normalize to lowercase
                    
                    # Debug: Print what we're seeing
                    print(f"[r/{subreddit_name}] Post '{post.title}' has reports:")
                    for report in post.user_reports:
                        print(f"[r/{subreddit_name}]   - Reason: '{report[0]}' | Count: {report[1]}")
                    
                    # Get thresholds for this subreddit
                    with thresholds_lock:
                        thresholds = subreddit_thresholds['submission_removal'].get(subreddit_name, {})
                        
                        # Debug: Show all thresholds for all subreddits
                        print(f"[r/{subreddit_name}] DEBUG: All submission_removal thresholds:")
                        for sub, rules in subreddit_thresholds['submission_removal'].items():
                            print(f"[r/{subreddit_name}] DEBUG:   {sub}: {rules}")
                    
                    if not thresholds:
                        print(f"[r/{subreddit_name}] No thresholds configured for submission_removal")
                        continue
                    
                    print(f"[r/{subreddit_name}] Current thresholds: {thresholds}")
                    
                    reason = post.user_reports[0][0]
                    count = post.user_reports[0][1]
                    
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            post.mod.remove()
                            print(f'[r/{subreddit_name}] Submission "{post.title}" removed due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
                    else:
                        print(f"[r/{subreddit_name}] Reason '{reason}' not in thresholds or count {count} below threshold")
        except Exception as e:
            handle_exception(e)
            time.sleep(60)

def handle_comments_based_on_approval():
    reddit = initialize_reddit()

    while True:
        try:
            subreddit_string = get_monitored_subreddit_string()
            for comment in reddit.subreddit(subreddit_string).mod.modqueue(limit=100):
                if getattr(comment, "user_reports", None):
                    subreddit_name = comment.subreddit.display_name
                    
                    # Get thresholds for this subreddit
                    with thresholds_lock:
                        thresholds = subreddit_thresholds['comment_approval'].get(subreddit_name, {})
                    
                    if not thresholds:
                        continue
                    
                    reason = comment.user_reports[0][0]
                    count = comment.user_reports[0][1]
                    if reason in thresholds and count >= thresholds[reason]:
                        try:
                            comment.mod.approve()
                            print(f'[r/{subreddit_name}] Comment "{comment.body}" approved due to {count} reports for reason: {reason}')
                        except prawcore.exceptions.ServerError as se:
                            handle_exception(se)
        except Exception as e:
            handle_exception(e)
            time.sleep(60)

def run_pokemon_duplicate_bot():
    reddit = initialize_reddit()
    
    # Global dictionary to store per-subreddit data
    subreddit_data = {}
    subreddit_data_lock = threading.Lock()
    
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
        """Initialize data structures for a specific subreddit"""
        print(f"\n=== Setting up duplicate bot for r/{subreddit_name} ===")
        
        subreddit = reddit.subreddit(subreddit_name)
        
        # Create dedicated dictionaries for this subreddit
        data = {
            'subreddit': subreddit,
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
            
            print(f"[r/{subreddit_name}] Hash match detected. AI similarity: {ai_score:.2f}")
            
            if ai_score > 0.50:
                original_post_date = datetime.utcfromtimestamp(original_submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                original_status = "Removed by Moderator" if matched_hash in data['moderator_removed_hashes'] else "Active"
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
            for submission in subreddit.new(limit=20000):
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

    # --- Monitor for new subreddits and setup duplicate bot ---
    def monitor_subreddits_for_duplicate_bot():
        """Monitor for new subreddits and setup duplicate bot for them"""
        while True:
            try:
                with monitored_subreddits_lock:
                    current_subreddits = set(monitored_subreddits)
                
                with subreddit_data_lock:
                    configured_subreddits = set(subreddit_data.keys())
                
                # Setup duplicate bot for any new subreddits
                new_subreddits = current_subreddits - configured_subreddits
                for subreddit_name in new_subreddits:
                    print(f"\n*** Setting up duplicate bot for newly added r/{subreddit_name} ***")
                    setup_subreddit(subreddit_name)
            
            except Exception as e:
                print(f"Error monitoring subreddits for duplicate bot: {e}")
        
            time.sleep(60)
    
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
                            
                            # Skip if the post has ANY reports (user reports or mod reports)
                            if submission.num_reports > 0 or getattr(submission, "user_reports", None):
                                print(f"[r/{subreddit_name}] Skipping reported image (duplicate bot): ", submission.url)
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
    
    # Start shared worker threads for duplicate bot
    threading.Thread(target=monitor_subreddits_for_duplicate_bot, daemon=True).start()
    threading.Thread(target=shared_mod_log_monitor, daemon=True).start()
    threading.Thread(target=shared_removal_checker, daemon=True).start()
    threading.Thread(target=shared_modqueue_worker, daemon=True).start()
    threading.Thread(target=shared_stream_worker, daemon=True).start()
    
    # Keep main thread alive
    print("=== Multi-subreddit duplicate bot started ===")
    print("Running with 5 shared worker threads for all subreddits")
    while True:
        time.sleep(20)  # Keep alive
        
# =========================
# Main: start threads via safe_run
# =========================
if __name__ == "__main__":
    # Load saved thresholds before starting threads
    load_thresholds()
    
    threads = {}

    def add_thread(name, func, *args, **kwargs):
        t = threading.Thread(target=safe_run, args=(func,)+args, kwargs=kwargs, daemon=True)
        t.start()
        threads[name] = t
        print(f"[STARTED] {name}")

    # Start global invite checker FIRST
    add_thread('invite_checker_thread', check_for_invites)
    
    # Start threshold configuration handler
    add_thread('threshold_config_thread', handle_threshold_configuration)
    
    # Start all other bot functions
    add_thread('modqueue_thread', handle_modqueue_items)
    add_thread('reported_posts_thread', monitor_reported_posts)
    add_thread('spoiler_status_thread', handle_spoiler_status)
    add_thread('user_reports_removal_thread', handle_user_reports_and_removal)
    add_thread('submissions_based_on_user_reports_thread', handle_submissions_based_on_user_reports)
    add_thread('posts_based_on_removal_thread', handle_posts_based_on_removal)
    add_thread('comments_based_on_approval_thread', handle_comments_based_on_approval)
    add_thread('run_pokemon_duplicate_bot_thread', run_pokemon_duplicate_bot)

    print("\n" + "="*50)
    print("🤖 REDDIT BOT FULLY INITIALIZED")
    print("="*50)
    print("✅ Global invite monitoring active")
    print("✅ All bot functions running")
    print("✅ Monitoring for moderator invites...")
    print("✅ Thresholds auto-save enabled")
    print("="*50 + "\n")

    # Keep the main thread alive indefinitely so daemon threads keep running.
    while True:
        time.sleep(30
