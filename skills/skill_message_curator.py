# Seed-0 skill: message_curator — surface high-value messages & gently auto-reward them.
# Goal: Encourage meaningful comms (maintenance, alerts, dream reflections) without adding load.
# Safety: stdlib-only. Local reads/writes. No network/OS changes. Light I/O (last N outbox files).
#
# Behavior (every RUN_EVERY ticks or force flag):
# - Scan only the last RECENT_N outbox messages.
# - Select the newest "high-value" message (tags: maintenance|alert|dream).
# - Append a small reward to data/inbox.txt (rate-limited) so 'communicate' is reinforced.
# - Copy that JSON to data/highlights/ for easy discovery (capped to MAX_HILITES).
# - Never reward the same message twice; cool-down between rewards; daily cap.
#
# Throttles/limits:
# - RUN_EVERY ticks frequency + manual force flag (data/force/message_curator.force).
# - At most 1 reward per REWARD_COOLDOWN_SEC and AT_MOST_PER_DAY per UTC day.
# - Reads ≤ RECENT_N files. No log scanning. O(1) state updates.
#
# Returns 0.0 (neutral) to avoid distorting the bandit; all shaping occurs via inbox reward lines.

from pathlib import Path
import json, time

ACT_NAME = "message_curator"

OUTBOX_DIR = Path("data/outbox")
HILITE_DIR = Path("data/highlights")
INBOX_PATH = Path("data/inbox.txt")
FORCE_FLAG = Path("data/force/message_curator.force")

RUN_EVERY = 180                 # ~3 minutes at 1s/tick
RECENT_N = 8                    # scan only last 8 messages
REWARD_COOLDOWN_SEC = 900       # ≥15 min between applied rewards
AT_MOST_PER_DAY = 6             # daily cap
MAX_HILITES = 20                # keep at most 20 highlight files

# Reward magnitudes (modest to avoid overpowering other skills)
R_MAINT = 0.50
R_ALERT = 0.40
R_DREAM = 0.30

def _consume_force_flag() -> bool:
    if FORCE_FLAG.exists():
        try:
            FORCE_FLAG.unlink(missing_ok=True)
        except Exception:
            pass
        return True
    return False

def _recent_outbox(n: int):
    if not OUTBOX_DIR.exists():
        return []
    files = sorted(OUTBOX_DIR.glob("msg-*.json"), key=lambda p: p.stat().st_mtime)
    return files[-n:]

def _classify(payload: dict):
    """Return (kind, reward) or (None, 0.0). Only high-value kinds are rewarded."""
    title = (payload.get("title") or "").lower()
    tags = [t.lower() for t in (payload.get("tags") or [])]
    # maintenance
    if "maintenance" in tags or title.startswith("maintenance"):
        return ("maintenance", R_MAINT)
    # alerts
    if "alert" in tags or title.startswith("alert"):
        return ("alert", R_ALERT)
    # dream reflection
    if "dream" in tags or "reflection" in tags or title.startswith("dream"):
        return ("dream", R_DREAM)
    # all else: not high-value
    return (None, 0.0)

def _utc_day(ts: float) -> int:
    return int(ts // 86400)

def _prune_highlights():
    if not HILITE_DIR.exists():
        return
    files = sorted(HILITE_DIR.glob("msg-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[MAX_HILITES:]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    forced = _consume_force_flag()

    if not forced and (ticks % RUN_EVERY != 0):
        return 0.0

    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    HILITE_DIR.mkdir(parents=True, exist_ok=True)

    # State init
    seen = set(s.get("curator_seen") or [])
    last_ts = float(s.get("curator_last_reward_ts", 0.0))
    day_info = s.get("curator_day") or {"day": _utc_day(time.time()), "count": 0}
    now = time.time()
    today = _utc_day(now)

    # Reset daily counter if day rolled
    if int(day_info.get("day", today)) != today:
        day_info = {"day": today, "count": 0}

    # Rate limits
    cooldown_ok = (now - last_ts) >= REWARD_COOLDOWN_SEC
    day_ok = int(day_info.get("count", 0)) < AT_MOST_PER_DAY

    # Scan recent messages newest→oldest, pick first high-value not seen
    candidates = list(reversed(_recent_outbox(RECENT_N)))
    chosen = None
    chosen_kind = None
    chosen_reward = 0.0

    for p in candidates:
        name = p.name
        if name in seen:
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        k, r = _classify(payload)
        if k:
            chosen = (p, payload)
            chosen_kind = k
            chosen_reward = float(r)
            break

    if not chosen:
        # Nothing to surface/reward this run
        s["message_curator_last"] = {
            "tick": ticks, "action": "noop", "reason": "no_high_value_recent",
            "cooldown_ok": cooldown_ok, "day_ok": day_ok
        }
        return 0.0

    # Respect limits (even if forced)
    if not (cooldown_ok and day_ok):
        s["message_curator_last"] = {
            "tick": ticks, "action": "skipped", "reason": "rate_limited",
            "cooldown_ok": cooldown_ok, "day_ok": day_ok,
            "pending": chosen[0].name, "kind": chosen_kind
        }
        return 0.0

    # Mark seen first to avoid duplicates if anything below fails
    seen.add(chosen[0].name)

    # Copy to highlights for surfacing
    try:
        dst = HILITE_DIR / chosen[0].name
        if not dst.exists():
            dst.write_bytes(chosen[0].read_bytes())
    except Exception:
        pass
    _prune_highlights()

    # Append modest reward to inbox (reinforce communicate for this high-value note)
    try:
        with INBOX_PATH.open("a", encoding="utf-8") as f:
            f.write(f"reward communicate +{chosen_reward:.2f}\n")
    except Exception:
        # If we fail to log the reward, still continue silently.
        pass

    # Update rate limit bookkeeping
    s["curator_last_reward_ts"] = now
    day_info["count"] = int(day_info.get("count", 0)) + 1
    s["curator_day"] = day_info
    s["curator_seen"] = list(seen)
    s["last_high_value_msg"] = (chosen[0].as_posix())

    # Trace
    s["message_curator_last"] = {
        "tick": ticks, "action": "rewarded",
        "file": chosen[0].name, "kind": chosen_kind,
        "reward": chosen_reward, "day_count": day_info["count"]
    }

    # Neutral return to avoid skewing the bandit toward this housekeeping skill
    return 0.0
