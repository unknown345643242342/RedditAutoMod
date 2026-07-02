# MultiModKit — PokeLeaks mod tool (Devvit)

A Reddit Developer Platform (Devvit Web) app that replaces the old `hybridasync.py`
moderation bot. It runs on Reddit's own infrastructure and handles everything
Reddit-facing — event triggers, a settings UI, and all mod actions (remove,
approve, comment, spoiler). The one thing it does **not** do is the heavy image
math; that stays in a small Python service (`duplicate_service.py`) reached over
HTTP, intended to run on a Raspberry Pi.

```
Reddit event ──▶ Devvit app (this repo) ──▶ Reddit API + Redis + settings
                        │
                        └─ image duplicate check only ─▶ HTTPS ─▶ duplicate_service.py (Pi)
                                                                  pHash + ORB + EfficientNet
```

The duplicate-detection half is optional and only activates once the Pi service
is configured. Everything else (report thresholds, spoiler enforcement,
auto-approve, re-approve) works on its own with nothing else running.

---

## What it does

Each item maps to a piece of the original bot:

| Feature | Trigger | Old bot equivalent |
| --- | --- | --- |
| Remove duplicate image posts | `onPostSubmit` → Pi `/check` | stream / modqueue duplicate workers |
| Remove/approve on report thresholds | `onPostReport`, `onCommentReport` | `handle_report_thresholds` |
| Re-approve a mod-approved post that gets reported again | `onPostReport` | `monitor_reported_posts` |
| Auto-approve a single-report post after 1 hour | `onPostReport` + scheduler | `handle_modqueue_items` |
| Re-spoiler a post un-spoilered by a non-mod | `onPostSpoilerUpdate` | `handle_spoiler_status` |
| Flag a mod-removed image so reposts are caught | `onModAction` → Pi `/mod-removed` | `shared_mod_log_monitor` |
| Purge a user-deleted post from the index | `onPostDelete` → Pi `/delete` | `shared_removal_checker` |
| Index recent posts on install/upgrade | `onAppInstall`, `onAppUpgrade` → Pi `/index` | initial per-sub scan |

Multi-subreddit registry and mod-invite acceptance are gone by design: each
subreddit simply installs the app.

---

## Project layout

```
multimodkit/
├── devvit.json            # app manifest: triggers, scheduler, settings, permissions
├── package.json           # build/dev/typecheck scripts + pinned deps
├── tsconfig.json
├── src/
│   └── server/
│       └── index.ts       # the entire app (Hono server, all handlers)
├── dist/
│   └── server/
│       └── index.cjs       # built bundle (generated; do not edit)
└── duplicate_service.py    # the Pi-side ML service (runs separately)
```

Two config details that are easy to get wrong (they cost a lot of debugging):

- **`server.entry` in `devvit.json` must be just `"index.cjs"`** — NOT
  `"dist/server/index.cjs"`. Devvit automatically prepends `dist/server/`, so
  putting the full path produces `dist/server/dist/server/index.cjs` and fails.
- **`devvit.json` needs a `scripts` block** (`dev` and `build`). `devvit playtest`
  runs `scripts.dev` to build the bundle; without it, nothing is generated and you
  get "waiting for config.server.entry file".

---

## Everyday commands

Run these from the `multimodkit/` folder.

```bash
npm install            # first time only (installs pinned deps)
npm run build          # one-off build -> dist/server/index.cjs
npm run typecheck      # tsc --noEmit; should print nothing
npx devvit login       # connect your Reddit account (first time / new machine)
npx devvit playtest r/multimodkit_dev   # live, auto-rebuilding test install
npx devvit upload      # publish a version for real installs
```

`npm run dev` on its own only starts the esbuild watcher and never exits — you
don't run it directly; `devvit playtest` runs it for you.

**Always test on a private throwaway subreddit first (`r/multimodkit_dev`), never
on r/PokeLeaks.** A wrong threshold can remove real posts.

---

## Settings (per subreddit)

Set these on the app's install-settings page for each subreddit.

- **Duplicate-detection service base URL** — e.g. `https://your-pi-host`. Leave
  blank to disable duplicate detection entirely (the rest still works). The
  hostname must also be in `permissions.http.domains` in `devvit.json`.
- **Report thresholds JSON** — keyed by report reason, with `"*"` as a wildcard
  for any reason. `0` (or omitted) disables that action. Remove wins over approve.
  ```json
  {
    "This is spam": { "postRemove": 3, "commentRemove": 3 },
    "*": { "postApprove": 5 }
  }
  ```
  Valid keys per reason: `postRemove`, `postApprove`, `commentRemove`,
  `commentApprove`. The value is validated when you save.
- Toggles: remove duplicates, act on report thresholds, 1-hour auto-approve,
  re-approve reported posts, spoiler enforcement, backfill on install.

### The auth token

The Pi service can require a shared secret (`X-Auth-Token`). It is **not** declared
in `devvit.json` (the schema rejects a global secret string there — that's what the
`serviceAuthToken is not exactly one from ...` error was). Store it as a Devvit
secret from the CLI instead. Confirm the exact subcommand for your CLI version with
`npx devvit settings --help`; it is roughly:

```bash
npx devvit settings set serviceAuthToken
```

If you skip the token, also remove the `X-Auth-Token` check by leaving `AUTH_TOKEN`
unset on the Pi (see below). Then anyone who knows the URL can call the service, so
prefer a token for anything public.

---

## The Raspberry Pi service (duplicate detection — optional, set up last)

Only needed for duplicate-image removal. Heavy: it loads an EfficientNet model, so
use a Pi 4/5 with ≥4 GB RAM.

```bash
pip install aiohttp numpy pillow imagehash opencv-python-headless torch torchvision
export AUTH_TOKEN="a-long-random-string"   # optional; must match the Devvit secret
python duplicate_service.py                  # serves on :8080
```

It must be reachable over **HTTPS** at a stable hostname (a reverse proxy or tunnel),
because Devvit only calls allow-listed HTTPS domains. Once it's reachable:

1. Put the hostname (domain only, no `https://`, no path) in
   `permissions.http.domains` in `devvit.json`, then `npx devvit upload` (this goes
   for admin review).
2. Set the **Duplicate-detection service base URL** subreddit setting to the full
   `https://...` URL.

The service holds its index in memory; restarting it clears the index, and the next
install/upgrade backfill re-seeds recent posts. It never talks to Reddit and needs
no Reddit credentials.

---

## Gotchas we already hit (so future-you doesn't re-debug them)

- **`devvit` command not found** → use `npx devvit ...` from the project folder, or
  install globally with `sudo npm install -g devvit`.
- **`serviceAuthToken is not exactly one from ...`** → don't declare the token as a
  global setting; use a Devvit secret (above).
- **"waiting for config.server.entry file"** → `server.entry` must be `"index.cjs"`
  and `devvit.json` must have the `scripts` block.
- **esbuild "Could not resolve src/server/index.ts"** → the source file isn't where
  the build script expects; it must be at `src/server/index.ts`.
- **esbuild allow-scripts warning** → `npm rebuild esbuild` (or
  `npm approve-scripts esbuild`) so its binary is set up.
- **Wrong system clock** (Pi showing tomorrow's date) → uploads over HTTPS can fail
  with certificate errors; fix with `sudo timedatectl set-ntp true`.
```
