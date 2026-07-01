/**
 * PokeLeaks mod tool — Devvit Web server.
 *
 * This replaces the Reddit-facing half of the old asyncpraw bot. Every polling
 * loop from hybridasync.py is now an event trigger, and all Reddit mod actions
 * (remove / approve / comment / spoiler) happen here. The only thing that still
 * lives in Python is the heavy image ML, reached over HTTP (see duplicate_service.py).
 *
 * Mapping from the old bot:
 *   run_pokemon_duplicate_bot (stream/modqueue) -> onPostSubmit  -> POST /check
 *   shared_mod_log_monitor (removelink)         -> onModAction   -> POST /mod-removed
 *   shared_removal_checker (user deletes)        -> onPostDelete  -> POST /delete
 *   handle_report_thresholds                     -> onPostReport / onCommentReport
 *   monitor_reported_posts (re-approve)          -> onPostReport
 *   handle_modqueue_items (1-report 1h timer)    -> onPostReport + scheduler one-off
 *   handle_spoiler_status                        -> onPostSpoilerUpdate
 *   sync_subreddit_rules_and_config (wiki table) -> subreddit settings + JSON config
 *   sync_moderated_subreddits (invite accept)    -> gone; each sub installs the app
 */

import express from 'express';
import { createServer, getServerPort, context } from '@devvit/server';
import { reddit } from '@devvit/reddit';
import { redis } from '@devvit/redis';
import { scheduler } from '@devvit/scheduler';
import { settings } from '@devvit/settings';

// ---------------------------------------------------------------------------
// Types (trigger payloads parsed permissively; we re-fetch rich models below)
// ---------------------------------------------------------------------------
type ThingV2 = { id: string; url?: string; permalink?: string; title?: string; authorName?: string };
type PostSubmitBody = { post?: ThingV2; author?: { name?: string } };
type ReportBody = { post?: ThingV2; comment?: ThingV2; reason?: string };
type SpoilerBody = { post?: ThingV2 };
type DeleteBody = { postId?: string; post?: ThingV2 };
type ModActionBody = {
  action?: string;
  moderator?: { name?: string };
  moderatorName?: string;
  targetPost?: ThingV2;
};
type TaskBody = { data?: { postId?: string } };
type ValidateBody<T> = { value?: T };

type RuleThresholds = {
  postRemove?: number;
  postApprove?: number;
  commentRemove?: number;
  commentApprove?: number;
};
type ThresholdConfig = Record<string, RuleThresholds>;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
const bareId = (id: string): string => id.replace(/^t[0-9]_/, '');
const t3 = (id: string): string => (id.startsWith('t3_') ? id : `t3_${bareId(id)}`);
const t1 = (id: string): string => (id.startsWith('t1_') ? id : `t1_${bareId(id)}`);

const IMAGE_RE = /\.(jpg|jpeg|png|gif)$/i;
const isImageUrl = (url?: string): boolean => !!url && IMAGE_RE.test(url.split('?')[0]);

async function getBool(key: string, fallback = false): Promise<boolean> {
  try {
    const v = await settings.get(key);
    return v === undefined || v === null ? fallback : Boolean(v);
  } catch {
    return fallback;
  }
}

async function getStr(key: string): Promise<string> {
  try {
    return ((await settings.get(key)) as string) ?? '';
  } catch {
    return '';
  }
}

async function getThresholds(): Promise<ThresholdConfig> {
  const raw = await getStr('reportThresholds');
  if (!raw.trim()) return {};
  try {
    const parsed = JSON.parse(raw);
    return typeof parsed === 'object' && parsed !== null ? (parsed as ThresholdConfig) : {};
  } catch {
    console.error('[config] reportThresholds is not valid JSON; ignoring');
    return {};
  }
}

/** Threshold for a specific reason + action, falling back to the "*" wildcard. */
function thresholdFor(cfg: ThresholdConfig, reason: string, action: keyof RuleThresholds): number {
  const safe = reason.replace(/\|/g, '-').trim();
  const specific = cfg[safe]?.[action];
  if (typeof specific === 'number') return specific;
  const wildcard = cfg['*']?.[action];
  return typeof wildcard === 'number' ? wildcard : 0;
}

/** Count how many reports carry each reason string. */
function countReasons(reasons: string[] | undefined): Map<string, number> {
  const counts = new Map<string, number>();
  for (const r of reasons ?? []) counts.set(r, (counts.get(r) ?? 0) + 1);
  return counts;
}

/** POST JSON to the Python duplicate-detection service. Returns null on any failure. */
async function callService(path: string, body: Record<string, unknown>): Promise<any | null> {
  const base = (await getStr('duplicateServiceUrl')).replace(/\/+$/, '');
  if (!base) {
    console.error('[service] duplicateServiceUrl is not set');
    return null;
  }
  const token = await getStr('serviceAuthToken');
  try {
    const res = await fetch(`${base}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { 'X-Auth-Token': token } : {}),
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      console.error(`[service] ${path} -> HTTP ${res.status}`);
      return null;
    }
    return await res.json();
  } catch (err) {
    console.error(`[service] ${path} failed:`, err);
    return null;
  }
}

function buildDuplicateComment(o: any): string {
  const author = o?.author || '[Deleted]';
  const title = o?.title || 'original post';
  const permalink = o?.permalink || '';
  const date = o?.date || '';
  const status = o?.status || 'Active';
  const titleCell = permalink ? `[${title}](${permalink})` : title;
  return [
    '> **Duplicate detected**',
    '',
    '| Original Author | Title | Date | Status |',
    '|:---------------:|:-----:|:----:|:------:|',
    `| ${author} | ${titleCell} | ${date} | ${status} |`,
  ].join('\n');
}

/** Simple TTL de-dupe so double-delivered triggers don't double-act. */
async function firstTimeSeen(key: string, ttlSeconds: number): Promise<boolean> {
  try {
    if (await redis.get(key)) return false;
    await redis.set(key, '1');
    await redis.expire(key, ttlSeconds);
    return true;
  } catch {
    return true; // never let de-dupe bookkeeping block real work
  }
}

// ---------------------------------------------------------------------------
// Express app
// ---------------------------------------------------------------------------
const app = express();
app.use(express.json());

const ok = (res: express.Response) => res.json({ status: 'ok' });
const wrap =
  (fn: (req: express.Request, res: express.Response) => Promise<unknown>) =>
  async (req: express.Request, res: express.Response) => {
    try {
      await fn(req, res);
    } catch (err) {
      console.error(`[handler] ${req.path} error:`, err);
      if (!res.headersSent) res.json({ status: 'ok' }); // ack so Reddit doesn't spam retries
    }
  };

// ===========================================================================
// Duplicate detection: new post submitted  (old stream/modqueue workers)
// ===========================================================================
app.post(
  '/internal/triggers/post-submit',
  wrap(async (req, res) => {
    const body = req.body as PostSubmitBody;
    const rawId = body.post?.id;
    if (!rawId) return ok(res);
    const postId = t3(rawId);

    if (!(await getBool('enableDuplicateDetection', true))) return ok(res);
    if (!(await firstTimeSeen(`dup:seen:${postId}`, 3600))) return ok(res);

    const post = await reddit.getPostById(postId);
    if (!isImageUrl(post.url)) return ok(res);

    const verdict = await callService('/check', {
      subreddit: context.subredditName,
      submission_id: bareId(postId),
      url: post.url,
      created_utc: Math.floor(post.createdAt.getTime() / 1000),
      author: post.authorName,
      title: post.title,
      permalink: post.permalink,
    });

    if (verdict?.duplicate && !post.approved) {
      await post.remove(false);
      const comment = await post.addComment({ text: buildDuplicateComment(verdict.original) });
      await comment.distinguish(true);
      console.log(`[r/${context.subredditName}] removed duplicate (${verdict.method}): ${postId}`);
    }
    return ok(res);
  })
);

// ===========================================================================
// Report thresholds + re-approve + 1-report auto-approve timer (posts)
// ===========================================================================
app.post(
  '/internal/triggers/post-report',
  wrap(async (req, res) => {
    const body = req.body as ReportBody;
    const rawId = body.post?.id;
    if (!rawId) return ok(res);
    const postId = t3(rawId);
    const post = await reddit.getPostById(postId);

    // monitor_reported_posts: keep mod-approved posts approved.
    if ((await getBool('enableReApproveReported', true)) && post.approved) {
      await post.approve();
      console.log(`[r/${context.subredditName}] re-approved reported post ${postId}`);
      return ok(res);
    }

    // handle_report_thresholds: remove takes priority over approve.
    if (await getBool('enableReportThresholds', true)) {
      const cfg = await getThresholds();
      const counts = countReasons(post.userReportReasons);

      for (const [reason, count] of counts) {
        if (count >= thresholdFor(cfg, reason, 'postRemove') && thresholdFor(cfg, reason, 'postRemove') > 0) {
          await post.remove(false);
          await clearAutoApprove(postId);
          console.log(`[r/${context.subredditName}] removed post ${postId} (${count}x "${reason}")`);
          return ok(res);
        }
      }
      for (const [reason, count] of counts) {
        if (count >= thresholdFor(cfg, reason, 'postApprove') && thresholdFor(cfg, reason, 'postApprove') > 0) {
          await post.approve();
          await clearAutoApprove(postId);
          console.log(`[r/${context.subredditName}] approved post ${postId} (${count}x "${reason}")`);
          return ok(res);
        }
      }
    }

    // handle_modqueue_items: single-report posts auto-approve after 1h unless
    // more reports arrive in the meantime.
    if (await getBool('enableAutoApproveSingleReport', true)) {
      const total = post.numberOfReports ?? 0;
      const key = `autoapprove:job:${postId}`;
      const existing = await redis.get(key);
      if (total === 1 && !existing) {
        const jobId = await scheduler.runJob({
          name: 'auto-approve-single-report',
          data: { postId },
          runAt: new Date(Date.now() + 60 * 60 * 1000),
        });
        await redis.set(key, jobId);
        await redis.expire(key, 2 * 60 * 60);
        console.log(`[r/${context.subredditName}] armed 1h auto-approve for ${postId}`);
      } else if (total > 1 && existing) {
        await clearAutoApprove(postId);
      }
    }
    return ok(res);
  })
);

async function clearAutoApprove(postId: string): Promise<void> {
  const key = `autoapprove:job:${postId}`;
  const jobId = await redis.get(key);
  if (jobId) {
    try {
      await scheduler.cancelJob(jobId);
    } catch {
      /* already fired or gone */
    }
    await redis.del(key);
  }
}

app.post(
  '/internal/scheduler/auto-approve',
  wrap(async (req, res) => {
    const { data } = (req.body as TaskBody) ?? {};
    const postId = data?.postId;
    if (!postId) return ok(res);
    const post = await reddit.getPostById(t3(postId));
    if (!post.removed && !post.approved && (post.numberOfReports ?? 0) === 1) {
      await post.approve();
      console.log(`[r/${context.subredditName}] auto-approved single-report post ${postId}`);
    }
    await redis.del(`autoapprove:job:${t3(postId)}`);
    return ok(res);
  })
);

// ===========================================================================
// Report thresholds (comments)
// ===========================================================================
app.post(
  '/internal/triggers/comment-report',
  wrap(async (req, res) => {
    if (!(await getBool('enableReportThresholds', true))) return ok(res);
    const body = req.body as ReportBody;
    const rawId = body.comment?.id;
    if (!rawId) return ok(res);
    const comment = await reddit.getCommentById(t1(rawId));

    const cfg = await getThresholds();
    const counts = countReasons(comment.userReportReasons);

    for (const [reason, count] of counts) {
      if (count >= thresholdFor(cfg, reason, 'commentRemove') && thresholdFor(cfg, reason, 'commentRemove') > 0) {
        await comment.remove(false);
        console.log(`[r/${context.subredditName}] removed comment ${rawId} (${count}x "${reason}")`);
        return ok(res);
      }
    }
    for (const [reason, count] of counts) {
      if (count >= thresholdFor(cfg, reason, 'commentApprove') && thresholdFor(cfg, reason, 'commentApprove') > 0) {
        await comment.approve();
        console.log(`[r/${context.subredditName}] approved comment ${rawId} (${count}x "${reason}")`);
        return ok(res);
      }
    }
    return ok(res);
  })
);

// ===========================================================================
// Spoiler enforcement  (old handle_spoiler_status)
// ===========================================================================
app.post(
  '/internal/triggers/post-spoiler-update',
  wrap(async (req, res) => {
    if (!(await getBool('enableSpoilerEnforcement', true))) return ok(res);
    const body = req.body as SpoilerBody;
    const rawId = body.post?.id;
    if (!rawId) return ok(res);
    const postId = t3(rawId);

    // Our own markAsSpoiler() re-fires this trigger; short de-dupe stops a loop.
    if (!(await firstTimeSeen(`spoiler:seen:${postId}:${Date.now() >> 13}`, 30))) return ok(res);

    const post = await reddit.getPostById(postId);
    if (post.spoiler) return ok(res); // spoiler is set — nothing to enforce

    // It was un-spoilered. Allow it only if a human moderator did it.
    let byMod = false;
    try {
      const log = await reddit
        .getModerationLog({ subredditName: context.subredditName!, type: 'unspoiler', limit: 50 })
        .all();
      byMod = log.some((a) => a.target?.id === postId && a.moderatorName !== context.appSlug);
    } catch (err) {
      console.error('[spoiler] mod-log lookup failed:', err);
    }

    if (!byMod) {
      await post.markAsSpoiler();
      console.log(`[r/${context.subredditName}] re-spoilered ${postId} (no mod log entry)`);
    }
    return ok(res);
  })
);

// ===========================================================================
// Moderator removed an original  ->  flag its hash (old shared_mod_log_monitor)
// ===========================================================================
app.post(
  '/internal/triggers/mod-action',
  wrap(async (req, res) => {
    const body = req.body as ModActionBody;
    const modName = body.moderator?.name ?? body.moderatorName;
    const targetId = body.targetPost?.id;
    if (body.action === 'removelink' && targetId && modName && modName !== context.appSlug) {
      await callService('/mod-removed', {
        subreddit: context.subredditName,
        submission_id: bareId(targetId),
      });
    }
    return ok(res);
  })
);

// ===========================================================================
// Post deleted by its author  ->  purge from index (old shared_removal_checker)
// ===========================================================================
app.post(
  '/internal/triggers/post-delete',
  wrap(async (req, res) => {
    const body = req.body as DeleteBody;
    const rawId = body.postId ?? body.post?.id;
    if (!rawId) return ok(res);
    await callService('/delete', {
      subreddit: context.subredditName,
      submission_id: bareId(rawId),
    });
    return ok(res);
  })
);

// ===========================================================================
// Backfill recent posts on install/upgrade  (old per-subreddit initial scan)
// ===========================================================================
async function backfill(): Promise<void> {
  if (!(await getBool('backfillOnInstall', true))) return;
  if (!(await getBool('enableDuplicateDetection', true))) return;
  try {
    const posts = await reddit.getNewPosts({ subredditName: context.subredditName!, limit: 25 }).all();
    let indexed = 0;
    for (const post of posts) {
      if (!isImageUrl(post.url)) continue;
      const r = await callService('/index', {
        subreddit: context.subredditName,
        submission_id: bareId(post.id),
        url: post.url,
        created_utc: Math.floor(post.createdAt.getTime() / 1000),
        author: post.authorName,
        title: post.title,
        permalink: post.permalink,
      });
      if (r?.indexed) indexed++;
    }
    console.log(`[r/${context.subredditName}] backfill indexed ${indexed} images`);
  } catch (err) {
    console.error('[backfill] failed:', err);
  }
}

app.post('/internal/triggers/app-install', wrap(async (_req, res) => { await backfill(); ok(res); }));
app.post('/internal/triggers/app-upgrade', wrap(async (_req, res) => { await backfill(); ok(res); }));

// ===========================================================================
// Settings validation for the thresholds JSON
// ===========================================================================
app.post('/internal/settings/validate-thresholds', (req, res) => {
  const { value } = (req.body as ValidateBody<string>) ?? {};
  if (!value || !value.trim()) return res.json({ success: true }); // empty = disabled
  try {
    const parsed = JSON.parse(value);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return res.json({ success: false, error: 'Must be a JSON object keyed by report reason.' });
    }
    for (const [reason, rules] of Object.entries(parsed)) {
      if (typeof rules !== 'object' || rules === null) {
        return res.json({ success: false, error: `"${reason}" must map to an object of thresholds.` });
      }
      for (const [k, v] of Object.entries(rules as Record<string, unknown>)) {
        if (!['postRemove', 'postApprove', 'commentRemove', 'commentApprove'].includes(k)) {
          return res.json({ success: false, error: `Unknown key "${k}" under "${reason}".` });
        }
        if (typeof v !== 'number' || v < 0 || !Number.isInteger(v)) {
          return res.json({ success: false, error: `"${reason}.${k}" must be a non-negative integer.` });
        }
      }
    }
    return res.json({ success: true });
  } catch {
    return res.json({ success: false, error: 'Invalid JSON.' });
  }
});

// ---------------------------------------------------------------------------
const server = createServer(app);
server.listen(getServerPort());
