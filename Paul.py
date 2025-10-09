import praw
import openai
import time
import re
import random
import traceback
import prawcore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
from openai import OpenAI

# --- CONFIGURATION (kept exactly as you provided) ---
openai_client = OpenAI(api_key="")

reddit = praw.Reddit(
    client_id='jl-I3OHYH2_VZMC1feoJMQ',
    client_secret='TCOIQBXqIskjWEbdH9i5lvoFavAJ1A',
    username='PokeLeakBot3',
    password='testbot1',
    user_agent='testbot'
)

subreddit_name = "PokeLeaks"
human_moderators = ["u/Gismo69", "u/vagrantwade", "u/Aether13", "u/Cmholde2"]

# --- BOT DECISIONS LOG ---
bot_decisions = []  # Store GPT's decisions to compare with human mods

# --- SAFE RUN + BACKOFF ---
def backoff_sleep(attempt, base=5, cap=120):
    # Exponential backoff with jitter: min(cap, base * 2^attempt) plus 0-1s jitter
    delay = min(cap, base * (2 ** attempt)) + random.random()
    time.sleep(delay)

def safe_run(func, *args, **kwargs):
    """
    Run a function forever. If it raises, log the error, back off, and retry.
    This keeps the worker alive even across network/API glitches.
    """
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            attempt += 1
            print(f"\n[SAFE_RUN] {func.__name__} crashed with: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            backoff_sleep(attempt)

# --- DOXXING/LINK FILTER (unchanged) ---
def is_doxxing_or_link(comment_text):
    patterns = [
        r'(https?://\S+|www\.\S+)',  # Normal URLs
        r'(hxxps?|https?|ftp|www|\[dot\]|\(dot\)|dot)\s*[:\.]?\s*//?',  # Obfuscated URLs
        r'[\w\-]+\.(com|net|org|info|io|co|gg|to|xyz|gov|edu|biz|us|ca|me)',
        r'([a-z0-9\-]+\s*\[?dot\]?\s*[a-z]+)',
        r'(pastebin\.com|mega\.nz|anonfiles\.com|drive\.google\.com|dropbox\.com)',
        r'(discord\.gg|discordapp\.com|discord\.com)',
        r'(t\.me|telegram\.me)',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        r'\b(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}\b',
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        r'u\/[A-Za-z0-9_-]{3,20}',  # Reddit usernames
        r'r\/[A-Za-z0-9_]{3,21}'    # Subreddit mentions
    ]
    for pattern in patterns:
        if re.search(pattern, comment_text, re.IGNORECASE):
            return True, "Comment contains a link, username, or subreddit mention"
    return False, ""

# --- FETCH AUTOMOD RULES (unchanged) ---
def fetch_automod_rules():
    try:
        wiki_page = reddit.subreddit(subreddit_name).wiki["config/automoderator"]
        return wiki_page.content_md
    except Exception as e:
        print("❌ Error fetching AutoMod rules:", e)
        return ""

def parse_automod_rules(content):
    rules = {}
    rule_pattern = r"Rule #(\d+):\s*(.*?)(?=\nRule #|\Z)"
    matches = re.findall(rule_pattern, content, re.DOTALL)
    for number, text in matches:
        rules[int(number)] = text.strip()
    return rules

# --- GPT AI MODERATOR (kept, with try/except) ---
def gpt_decision(comment_text, rules_summary):
    prompt = f"""
You are a Reddit moderator and conversation analysis AI bot for the r/PokeLeaks subreddit that mimics a human moderator. Analyze the following comment using common Reddit rules, community standards, and AutoMod rules below. Pay special attention to the tone and context, especially when profanity or strong and sexual language is used, No links, mentions of other users or subreddits, or any form of exploiting information that can lead to the doxxing of third parties should be allowed such as emails, IP addresses, phone numbers should be approved and stay removed. Comments asking about piracy, pirated or copyrighted files, pirated files such as ROMs, ISOs, XCI files or any form of files should not be approved and stay removed. As a moderator you do not condone any form of illegal pirated material or copyrighted file sharing or distribution. Any mention of Riddler Khu or Khu should stay removed and not be approved. Do not approve comments asking about discord servers or discord links 
Your job is to classify Reddit comment threads based on tone, civility, and relevance to Pokémon leaks.

    Safe discussions include:
    - Civil debates over leaks, designs, leakers' credibility or light hearted and friendly jokes
    - Respectful speculation or disagreements
    - Analytical conversations about gameplay, consoles, or future content

    Flagged discussions include:
    - Arguments that involve insults, sarcasm, personal attacks
    - Off-topic political, religious, or inflammatory comments
    - Escalating back-and-forth replies with increasing hostility
    - Asking about discord link or servers, pirated files such as xci, rom or iso media

    Always allow critical or passionate Pokémon discussion if the tone stays civil and respectful.

Comment:
\"\"\"{comment_text}\"\"\"

AutoMod Rules:
{rules_summary}

Please determine if the language is inappropriate or if it can be approved based on context. If the profanity is used in excitement, humor, positivity, lightheartedness, indicate this in your reasoning. Ignore any comments that involve Khu — those should stay removed. Remove any comments that contain sexual or political themes.

Reply only with either:
APPROVE: <short reason>
REMOVE: <short reason>
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=100
        )
        content = response.choices[0].message.content.strip()
        print("🧠 GPT Response:", content)

        if content.upper().startswith("REMOVE"):
            return "REMOVE", content
        elif content.upper().startswith("APPROVE"):
            return "APPROVE", content
        else:
            return "NEUTRAL", content

    except Exception as e:
        print("❌ GPT Error:", e)
        return "NEUTRAL", str(e)

# --- SIMILARITY ANALYSIS (unchanged) ---
def analyze_similarity(comment, approved, removed, result_holder):
    try:
        all_comments = approved + removed
        if not all_comments:
            result_holder["similar"] = False
            return
        texts = [comment.body] + [c.body for c in all_comments]
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(texts)
        sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).max()
        result_holder["similar"] = sim_score > 0.5
        result_holder["similarity_score"] = sim_score
    except Exception as e:
        print("❌ Similarity analysis failed:", e)
        result_holder["similar"] = False

# --- LEARNING FROM MOD LOGS (unchanged) ---
def learn_from_mods(subreddit):
    approved = []
    removed = []
    print("📘 Learning from mod logs...")
    for log in subreddit.mod.log(limit=100000, action="approvecomment"):
        try:
            approved.append(reddit.comment(log.target_fullname.split("_")[1]))
        except Exception:
            continue
    for log in subreddit.mod.log(limit=100000, action="removecomment"):
        try:
            removed.append(reddit.comment(log.target_fullname.split("_")[1]))
        except Exception:
            continue
    print(f"✅ Learned {len(approved)} approved and {len(removed)} removed comments.")
    return approved, removed

def was_removed_by_automod(comment):
    try:
        logs = list(comment.subreddit.mod.log(limit=100000, action="removecomment"))
        for log in logs:
            if log.target_fullname == comment.fullname:
                if isinstance(log.mod, praw.models.Redditor) and log.mod.name.lower() == "automoderator":
                    print("📛 Removed by AutoMod confirmed.")
                    return True
    except Exception as e:
        print("⚠️ Could not check mod log:", e)
    return False

# --- STORE BOT DECISIONS (unchanged) ---
def store_bot_decision(comment, decision, explanation, human_action):
    bot_decisions.append({
        "comment_id": comment.id,
        "bot_decision": decision,
        "bot_reason": explanation,
        "human_action": human_action
    })

# --- PROCESS A SINGLE COMMENT (wrapped by safe_run when threaded) ---
def process_comment(comment, approved, removed, rules_summary):
    # Guard: sometimes stream returns deleted/removed objects that error if touched
    try:
        _ = comment.body  # touch to force a fetch
    except Exception as e:
        print("⚠️ Skipping comment (unavailable):", e)
        return

    if not was_removed_by_automod(comment):
        return

    print(f"\n🗨️ Processing Comment: {comment.id} by {getattr(comment.author, 'name', '[deleted]')}")
    print(f"    Body: {comment.body}")

    # Doxxing/link check
    is_doxxing, reason = is_doxxing_or_link(comment.body)
    if is_doxxing:
        try:
            comment.mod.remove()
            print(f"❌ Auto-Removed (Link/Doxxing): {reason}")
        except Exception as e:
            print("❌ Failed to remove (doxxing):", e)
        return

    # Similarity in parallel
    result_holder = {}
    sim_thread = Thread(target=analyze_similarity, args=(comment, approved, removed, result_holder), daemon=True)
    sim_thread.start()

    # GPT decision
    decision, explanation = gpt_decision(comment.body, rules_summary)
    result_holder["gpt_decision"] = decision
    result_holder["gpt_reason"] = explanation

    sim_thread.join()
    is_similar = result_holder.get("similar", False)

    print(f"🔎 Similar to human-approved: {'✅ Yes' if is_similar else '❌ No'}")
    print(f"🤖 GPT Decision: {decision} — {explanation}")

    # Human mod check (best effort, don't crash)
    human_action = None
    try:
        for log in comment.subreddit.mod.log(limit=100):
            if log.target_fullname == comment.fullname:
                human_action = log.action
                break
    except Exception as e:
        print("⚠️ Could not fetch recent mod log for comment:", e)

    store_bot_decision(comment, decision, explanation, human_action)

    try:
        if decision == "APPROVE" or is_similar:
            comment.mod.approve()
            print("✅ Action: Approved")
        else:
            print("⏳ Action: No action taken")
    except prawcore.exceptions.TooManyRequests as e:
        print("⏳ Rate limited on approve(); backing off:", e)
        time.sleep(30)
    except Exception as e:
        print("❌ Failed to apply action:", e)

# --- MAIN MONITOR FUNCTION (now resilient) ---
def monitor_comments():
    sub = reddit.subreddit(subreddit_name)

    # Learn from logs & rules (with retry if network flakes)
    approved, removed = safe_run(learn_from_mods, sub)
    rules_raw = safe_run(fetch_automod_rules)
    parsed_rules = parse_automod_rules(rules_raw)
    rules_summary = "\n".join([f"Rule #{k}: {v}" for k, v in parsed_rules.items()])

    print("🚀 Monitoring AutoMod-removed comments in real-time...")

    attempt = 0
    while True:
        try:
            # reddit.stream.* generators can raise; keep them in a small try so we can resume
            for comment in sub.stream.comments(skip_existing=True):
                # Each comment handled in its own safe thread so a failure never kills the stream
                t = Thread(
                    target=safe_run,
                    args=(process_comment, comment, approved, removed, rules_summary),
                    daemon=True
                )
                t.start()
                time.sleep(1)  # throttle to be gentle with API
            # If the for-loop exits naturally, reset attempt
            attempt = 0
        except (prawcore.exceptions.ServerError,
                prawcore.exceptions.ResponseException,
                prawcore.exceptions.RequestException,
                prawcore.exceptions.TooManyRequests) as e:
            attempt += 1
            print(f"⚠️ Stream error ({e.__class__.__name__}): {e}. Will resume.")
            backoff_sleep(attempt, base=3, cap=90)
            continue
        except Exception as e:
            attempt += 1
            print(f"❌ Unexpected stream error: {e}")
            traceback.print_exc()
            backoff_sleep(attempt, base=5, cap=120)
            continue

def main():
    # Run the monitor under safe_run so even top-level crashes get retried
    safe_run(monitor_comments)

if __name__ == "__main__":
    main()

