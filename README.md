# PokeLeaks mod tool (Devvit)

A Reddit Developer Platform (Devvit Web) mod tool that replaces the old
`hybridasync.py` bot. Devvit now owns everything Reddit-facing — event triggers,
settings UI, and all mod actions — while the heavy image ML stays in a small
Python service (`duplicate_service.py`) reached over HTTP.

```
Reddit event ──▶ Devvit trigger ──▶ (settings + Redis + Reddit API)
                        │
                        └── image work only ──▶ HTTPS ──▶ duplicate_service.py
                                                          (pHash + ORB + EfficientNet)
```

## Why the split

Devvit runs sandboxed TypeScript on Reddit's infrastructure. It cannot run
OpenCV, Torch, or download-and-decode images with native libs, and it has no
long-lived process for polling. So:

- **Everything event-driven and Reddit-facing moved into Devvit.** No more polling
  loops, no more accepting mod invites, no more hardcoded Reddit credentials — each
  subreddit just installs the app and Devvit authenticates automatically.
- **Only the image math stayed in Python**, behind a tiny HTTP API. That service no
  longer talks to Reddit at all, which is why it needs no credentials.

## What maps to what

| Old bot (`hybridasync.py`)                         | Now |
| -------------------------------------------------- | --- |
| stream / modqueue duplicate workers                | `onPostSubmit` → `POST /check` → remove + sticky comment |
| `shared_mod_log_monitor` (`removelink`)            | `onModAction` → `POST /mod-removed` |
| `shared_removal_checker` (user deletes)            | `onPostDelete` → `POST /delete` |
| `handle_report_thresholds`                         | `onPostReport` / `onCommentReport` |
| `monitor_reported_posts` (re-approve)              | `onPostReport` |
| `handle_modqueue_items` (1 report → 1h auto-approve) | `onPostReport` + a one-off Scheduler job |
| `handle_spoiler_status`                            | `onPostSpoilerUpdate` + mod-log check |
| `sync_subreddit_rules_and_config` (wiki table)     | Subreddit **settings** + a validated thresholds JSON |
| `sync_moderated_subreddits` (invite accept)        | Removed — per-subreddit install |
| initial per-subreddit scan                         | `onAppInstall` / `onAppUpgrade` backfill → `POST /index` |

## Setup

### 1. The Python service

```bash
pip install aiohttp numpy pillow imagehash opencv-python-headless torch torchvision
export AUTH_TOKEN="some-long-random-string"   # optional but recommended
python duplicate_service.py                     # serves :8080
```

Expose it over **HTTPS** on a stable hostname (reverse proxy or tunnel). Devvit
can only fetch allow-listed HTTPS domains.

### 2. The Devvit app

```bash
npm install -g devvit
devvit login
# Scaffold once with the Mod Tool template, then drop these files in:
#   devvit new --template mod-tool pokeleaks-mod
# (copy devvit.json, package.json, tsconfig.json, src/ over the scaffold)
npm install
```

Edit **`devvit.json`**:

- Put your service hostname (no scheme, no path — e.g. `dupes.example.com`) in
  `permissions.http.domains`. This is submitted for admin review on upload.

Then:

```bash
npm run typecheck
devvit playtest r/YourTestSub     # hot-reloads while you test
devvit settings set serviceAuthToken   # store the shared secret (matches AUTH_TOKEN)
```

### 3. Per-subreddit settings

Moderators configure these on the app's install settings page:

- **Duplicate-detection service base URL** — e.g. `https://dupes.example.com`
- **Report thresholds JSON** — keyed by report reason (`"*"` = any reason):
  ```json
  {
    "This is spam": { "postRemove": 3, "commentRemove": 3 },
    "*": { "postApprove": 5 }
  }
  ```
  `0` (or omitted) disables that action. Remove always wins over approve.
- Toggles for duplicate detection, report thresholds, 1-hour auto-approve,
  re-approving reported posts, spoiler enforcement, and install backfill.

## Notes / caveats

- Triggers can double-deliver; handlers de-dupe via short-lived Redis keys and
  always re-check live state before acting.
- Redis is siloed per subreddit installation, which is fine here — the duplicate
  index lives in the Python service, and Devvit only stores small bookkeeping keys.
- The service keeps its index in memory. Restarting it clears the index; the next
  install/upgrade backfill re-seeds recent posts. Add a persistent store if you
  need durability across restarts.
- If your scaffold resolves the Devvit packages through the `@devvit/web/server`
  barrel instead of the individual `@devvit/*` packages, switch the imports at the
  top of `src/server/index.ts` accordingly — the APIs are identical.
```
