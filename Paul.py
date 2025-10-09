import praw
import openai
import time
import re
import random
import traceback
import prawcore
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
from openai import OpenAI
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# --- CONFIGURATION ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID", "jl-I3OHYH2_VZMC1feoJMQ"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET", "TCOIQBXqIskjWEbdH9i5lvoFavAJ1A"),
    username=os.getenv("REDDIT_USERNAME", "PokeLeakBot3"),
    password=os.getenv("REDDIT_PASSWORD", "testbot1"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "testbot")
)

subreddit_name = "PokeLeaks"
human_moderators = ["u/Gismo69", "u/vagrantwade", "u/Aether13", "u/Cmholde2"]

# --- BOT DECISIONS LOG ---
bot_decisions = []

# --- LOGGING SETUP ---
def log_decision(comment_id, author, decision, reason, action_taken):
    """Log all bot decisions for review"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] ID:{comment_id} | Author:{author} | Decision:{decision} | Reason:{reason} | Action:{action_taken}"
    print(log_entry)
    
    # Also write to file for persistence
    try:
        with open("bot_decisions.log", "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"⚠️ Failed to write to log file: {e}")

# --- SAFE RUN + BACKOFF ---
def backoff_sleep(attempt, base=5, cap=120):
    delay = min(cap, base * (2 ** attempt)) + random.random()
    print(f"⏳ Backing off for {delay:.1f} seconds (attempt {attempt})")
    time.sleep(delay)

def safe_run(func, *args, **kwargs):
    """Run a function forever with retry logic"""
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt detected. Shutting down...")
            raise
        except Exception as e:
            attempt += 1
            print(f"\n[SAFE_RUN] {func.__name__} crashed with: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            backoff_sleep(attempt)

# --- AUTOMATIC REMOVAL TRIGGERS ---
def should_auto_remove(comment_text):
    """Check for hard violation triggers before GPT analysis"""
    
    # Critical triggers that bypass GPT
    critical_triggers = [
        (r'\bkhu\b', "Mention of Khu (banned leaker)"),
        (r'riddler\s*khu', "Mention of Riddler Khu"),
        (r'\b(rom|iso|xci)\s*(file|download)?', "Piracy file format request"),
        (r'\b(pirat(e|ed|ing)|crack(ed)?|torrent)\b', "Piracy terminology"),
        (r'(where|how)\s+(can|do|to)\s+.{0,30}(download|get|find).{0,30}(rom|iso|xci|game|file)', "Piracy request"),
        (r'discord\s*(server|link|invite|channel)', "Discord server request"),
        (r'(join|invite|link).{0,20}discord', "Discord invite request"),
        (r'\bsex(ual|y|ually)?\b(?!.*pokemon)', "Sexual content (not Pokemon-related)"),
        (r'\b(porn|nsfw|xxx|nude|naked)\b', "Explicit content"),
        (r'(politics|political|democrat|republican|liberal|conservative)\b', "Political discussion"),
    ]
    
    text_lower = comment_text.lower()
    
    for pattern, reason in critical_triggers:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True, reason
    
    return False, ""

# --- DOXXING/LINK FILTER ---
def is_doxxing_or_link(comment_text):
    """Check for links and personal information"""
    patterns = [
        r'(https?://\S+|www\.\S+)',  # Normal URLs
        r'(hxxps?|https?|ftp|www|\[dot\]|\(dot\)|dot)\s*[:\.]?\s*//?',  # Obfuscated URLs
        r'[\w\-]+\.(com|net|org|info|io|co|gg|to|xyz|gov|edu|biz|us|ca|me)',
        r'([a-z0-9\-]+\s*\[?dot\]?\s*[a-z]+)',
        r'(pastebin\.com|mega\.nz|anonfiles\.com|drive\.google\.com|dropbox\.com)',
        r'(discord\.gg|discordapp\.com|discord\.com)',
        r'(t\.me|telegram\.me)',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b',  # Phone
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP address
        r'u\/[A-Za-z0-9_-]{3,20}',  # Reddit usernames
        r'r\/[A-Za-z0-9_]{3,21}'    # Subreddit mentions
    ]
    
    for pattern in patterns:
        if re.search(pattern, comment_text, re.IGNORECASE):
            return True, f"Contains link/personal info (pattern matched: {pattern[:30]}...)"
    
    return False, ""

# --- FETCH AUTOMOD RULES ---
def fetch_automod_rules():
    """Fetch AutoModerator configuration from wiki"""
    try:
        wiki_page = reddit.subreddit(subreddit_name).wiki["config/automoderator"]
        return wiki_page.content_md
    except Exception as e:
        print(f"❌ Error fetching AutoMod rules: {e}")
        return ""

def parse_automod_rules(content):
    """Parse AutoMod rules into structured format"""
    rules = {}
    rule_pattern = r"Rule #(\d+):\s*(.*?)(?=\nRule #|\Z)"
    matches = re.findall(rule_pattern, content, re.DOTALL)
    
    for number, text in matches:
        rules[int(number)] = text.strip()
    
    return rules

# --- GPT AI MODERATOR (IMPROVED) ---
def gpt_decision(comment_text, rules_summary):
    """Use GPT to analyze comment with strict rules"""
    
    prompt = f"""You are a strict Reddit moderator for r/PokeLeaks. Follow these rules EXACTLY in order:

AUTOMATIC REMOVAL - NO EXCEPTIONS:
1. ANY mention of "Khu" or "Riddler Khu" anywhere in comment
2. Requests for Discord servers, invites, or links
3. Piracy requests: ROM, ISO, XCI files, game downloads, "where to download"
4. Sexual or explicit content
5. Political discussions or inflammatory politics
6. Personal attacks or harassment directed at specific users

ALLOWED - May approve if civil and on-topic:
- Passionate opinions about Pokémon games, leaks, or designs
- Criticism of Game Freak, Nintendo, or game quality
- Speculation about future Pokémon content
- Debates about leaker credibility (EXCEPT Khu)
- Casual profanity used in excitement/humor (NOT directed at people)
- Respectful disagreements about Pokémon topics

COMMENT TO ANALYZE:
"{comment_text}"

INSTRUCTIONS:
1. Check if comment violates ANY automatic removal rule
2. If it violates a rule, respond: REMOVE: <specific rule number violated>
3. If it doesn't violate rules and is civil, respond: APPROVE: <brief reason>
4. When uncertain, choose REMOVE

RESPOND WITH ONLY ONE LINE - REMOVE or APPROVE:"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a strict moderation assistant. Always prioritize rule violations. When in doubt, REMOVE. Reply with only 'REMOVE: <reason>' or 'APPROVE: <reason>'."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.2,  # Lower = more consistent
            max_tokens=150
        )
        
        content = response.choices[0].message.content.strip()
        print(f"🧠 GPT Response: {content}")
        
        # Strict parsing - look for keywords
        content_upper = content.upper()
        
        if "REMOVE" in content_upper or "DENY" in content_upper:
            return "REMOVE", content
        elif "APPROVE" in content_upper or "ALLOW" in content_upper:
            return "APPROVE", content
        else:
            # Default to REMOVE if response is unclear
            return "REMOVE", f"Unclear GPT response (defaulting to remove): {content}"
            
    except Exception as e:
        print(f"❌ GPT Error: {e}")
        # On error, default to REMOVE for safety
        return "REMOVE", f"GPT API error - removing for safety: {str(e)}"

# --- SIMILARITY ANALYSIS ---
def analyze_similarity(comment, approved, removed, result_holder):
    """Analyze comment similarity to previously moderated comments"""
    try:
        all_comments = approved + removed
        if not all_comments:
            result_holder["similar"] = False
            return
        
        texts = [comment.body] + [c.body for c in all_comments]
        tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = tfidf.fit_transform(texts)
        
        sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).max()
        result_holder["similar"] = sim_score > 0.5
        result_holder["similarity_score"] = sim_score
        
    except Exception as e:
        print(f"❌ Similarity analysis failed: {e}")
        result_holder["similar"] = False

# --- LEARNING FROM MOD LOGS ---
def learn_from_mods(subreddit):
    """Learn from human moderator actions (OPTIMIZED)"""
    approved = []
    removed = []
    
    print("📘 Learning from recent mod logs...")
    
    # REDUCED from 100000 to 1000 for performance
    try:
        for log in subreddit.mod.log(limit=1000, action="approvecomment"):
            try:
                comment = reddit.comment(log.target_fullname.split("_")[1])
                approved.append(comment)
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ Error fetching approved comments: {e}")
    
    try:
        for log in subreddit.mod.log(limit=1000, action="removecomment"):
            try:
                comment = reddit.comment(log.target_fullname.split("_")[1])
                removed.append(comment)
            except Exception:
                continue
    except Exception as e:
        print(f"⚠️ Error fetching removed comments: {e}")
    
    print(f"✅ Learned from {len(approved)} approved and {len(removed)} removed comments.")
    return approved, removed

def was_removed_by_automod(comment):
    """Check if comment was removed by AutoModerator (OPTIMIZED)"""
    try:
        # REDUCED from 100000 to 50 - only check recent logs
        logs = list(comment.subreddit.mod.log(limit=50, action="removecomment"))
        
        for log in logs:
            if log.target_fullname == comment.fullname:
                if isinstance(log.mod, praw.models.Redditor) and log.mod.name.lower() == "automoderator":
                    print("📛 Removed by AutoMod confirmed.")
                    return True
                    
    except Exception as e:
        print(f"⚠️ Could not check mod log: {e}")
    
    return False

# --- STORE BOT DECISIONS ---
def store_bot_decision(comment, decision, explanation, human_action):
    """Store bot decision for later analysis"""
    bot_decisions.append({
        "comment_id": comment.id,
        "timestamp": datetime.now().isoformat(),
        "bot_decision": decision,
        "bot_reason": explanation,
        "human_action": human_action
    })

# --- PROCESS A SINGLE COMMENT ---
def process_comment(comment, approved, removed, rules_summary):
    """Process a single comment with all checks"""
    
    # Guard: Check if comment is accessible
    try:
        comment_body = comment.body
        comment_author = getattr(comment.author, 'name', '[deleted]')
    except Exception as e:
        print(f"⚠️ Skipping comment (unavailable): {e}")
        return
    
    # Only process AutoMod-removed comments
    if not was_removed_by_automod(comment):
        return
    
    print(f"\n{'='*60}")
    print(f"🗨️ Processing Comment ID: {comment.id}")
    print(f"👤 Author: {comment_author}")
    print(f"📝 Body: {comment_body[:200]}{'...' if len(comment_body) > 200 else ''}")
    print(f"{'='*60}")
    
    # STEP 1: Check for hard violations (auto-remove triggers)
    should_remove, remove_reason = should_auto_remove(comment_body)
    if should_remove:
        try:
            comment.mod.remove()
            log_decision(comment.id, comment_author, "AUTO_REMOVE", remove_reason, "REMOVED")
            print(f"❌ AUTO-REMOVED: {remove_reason}")
        except Exception as e:
            print(f"❌ Failed to remove (auto-trigger): {e}")
        return
    
    # STEP 2: Check for doxxing/links
    is_doxxing, doxx_reason = is_doxxing_or_link(comment_body)
    if is_doxxing:
        try:
            comment.mod.remove()
            log_decision(comment.id, comment_author, "DOXXING_LINK", doxx_reason, "REMOVED")
            print(f"❌ REMOVED (Link/Doxxing): {doxx_reason}")
        except Exception as e:
            print(f"❌ Failed to remove (doxxing): {e}")
        return
    
    # STEP 3: Run similarity analysis in parallel
    result_holder = {}
    sim_thread = Thread(
        target=analyze_similarity, 
        args=(comment, approved, removed, result_holder), 
        daemon=True
    )
    sim_thread.start()
    
    # STEP 4: Get GPT decision
    decision, explanation = gpt_decision(comment_body, rules_summary)
    result_holder["gpt_decision"] = decision
    result_holder["gpt_reason"] = explanation
    
    # Wait for similarity analysis
    sim_thread.join(timeout=5)  # Don't wait forever
    is_similar = result_holder.get("similar", False)
    similarity_score = result_holder.get("similarity_score", 0)
    
    print(f"🔎 Similarity Score: {similarity_score:.3f} - {'✅ Similar' if is_similar else '❌ Not similar'}")
    print(f"🤖 GPT Decision: {decision}")
    print(f"💭 GPT Reasoning: {explanation}")
    
    # STEP 5: Check for human moderator action
    human_action = None
    try:
        for log in comment.subreddit.mod.log(limit=50):
            if log.target_fullname == comment.fullname:
                human_action = log.action
                break
    except Exception as e:
        print(f"⚠️ Could not fetch mod log: {e}")
    
    # STEP 6: Store decision for analysis
    store_bot_decision(comment, decision, explanation, human_action)
    
    # STEP 7: Take action
    try:
        # Only approve if BOTH GPT says approve AND similar to approved content
        # This makes the bot more conservative
        if decision == "APPROVE" and is_similar:
            comment.mod.approve()
            log_decision(comment.id, comment_author, "APPROVE", explanation, "APPROVED")
            print("✅ ACTION: APPROVED (GPT + Similarity)")
        elif decision == "APPROVE" and not is_similar:
            # GPT wants to approve but no similar precedent - be cautious
            log_decision(comment.id, comment_author, "APPROVE_HELD", "GPT approved but no similar precedent", "NO_ACTION")
            print("⏸️ ACTION: HELD (GPT approved but no precedent)")
        else:
            log_decision(comment.id, comment_author, "REMOVE", explanation, "KEPT_REMOVED")
            print("🚫 ACTION: KEPT REMOVED")
            
    except prawcore.exceptions.TooManyRequests as e:
        print(f"⏳ Rate limited: {e}")
        time.sleep(60)
    except Exception as e:
        print(f"❌ Failed to apply action: {e}")
    
    print(f"{'='*60}\n")

# --- MAIN MONITOR FUNCTION ---
def monitor_comments():
    """Main monitoring loop"""
    sub = reddit.subreddit(subreddit_name)
    
    print("🚀 Starting Reddit Moderation Bot")
    print(f"📍 Subreddit: r/{subreddit_name}")
    print(f"🤖 Bot: {reddit.user.me()}")
    print()
    
    # Learn from logs & rules
    approved, removed = safe_run(learn_from_mods, sub)
    rules_raw = safe_run(fetch_automod_rules)
    parsed_rules = parse_automod_rules(rules_raw)
    rules_summary = "\n".join([f"Rule #{k}: {v[:100]}..." for k, v in parsed_rules.items()])
    
    print("\n🎯 Bot is now monitoring for AutoMod-removed comments...")
    print("📋 Check 'bot_decisions.log' for detailed decision logs\n")
    
    attempt = 0
    while True:
        try:
            for comment in sub.stream.comments(skip_existing=True):
                # Process each comment in a separate thread with safe_run
                t = Thread(
                    target=safe_run,
                    args=(process_comment, comment, approved, removed, rules_summary),
                    daemon=True
                )
                t.start()
                time.sleep(2)  # Throttle API calls
                
            # If loop exits naturally, reset attempt counter
            attempt = 0
            
        except (prawcore.exceptions.ServerError,
                prawcore.exceptions.ResponseException,
                prawcore.exceptions.RequestException,
                prawcore.exceptions.TooManyRequests) as e:
            attempt += 1
            print(f"⚠️ Stream error ({e.__class__.__name__}): {e}")
            backoff_sleep(attempt, base=3, cap=90)
            
        except Exception as e:
            attempt += 1
            print(f"❌ Unexpected stream error: {e}")
            traceback.print_exc()
            backoff_sleep(attempt, base=5, cap=120)

# --- MAIN ENTRY POINT ---
def main():
    """Run the bot with safe_run wrapper"""
    try:
        safe_run(monitor_comments)
    except KeyboardInterrupt:
        print("\n👋 Bot shutting down gracefully...")
        print(f"📊 Processed {len(bot_decisions)} comments this session")

if __name__ == "__main__":
    main()
