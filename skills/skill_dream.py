# Seed-0 skill: dream (Reflect → Learn → Evolve)
# - Reflect: read last ~2k events, compute action counts/avg rewards, errors
# - Learn: small off-policy nudges to Q-values using those averages; penalize error-heavy actions
# - Evolve: gently anneal epsilon toward stability (floor), keep 24h throttle + manual force
# - Anti-spam: small negative reward when not due, to avoid "dream spamming"

from pathlib import Path
from collections import deque, defaultdict
import json, time

ACT_NAME = "dream"

FORCE_FLAG = Path("data/force/dream.force")
THROTTLE_SECS = 3600  # 1 hour
LOG_PATH = Path("data/events.log")
WINDOW = 2000          # number of recent events to reflect over
DREAM_ALPHA = 0.05     # learning rate for dream-time Q updates
EPS_FLOOR = 0.10
EPS_DECAY = 0.995
ERROR_PENALTY_Q = -0.15
ERROR_THRESHOLD = 3

def consume_force_flag() -> bool:
    if FORCE_FLAG.exists():
        try:
            FORCE_FLAG.unlink(missing_ok=True)
        except Exception:
            pass
        return True
    return False

def _tail_events(path: Path, max_lines: int) -> list[dict]:
    evts = deque(maxlen=max_lines)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evts.append(json.loads(line))
            except Exception:
                # Skip malformed lines
                continue
    return list(evts)

def act(ctx):
    s = ctx["state"]
    now = time.time()
    last = float(s.get("last_dream_ts", 0.0))
    forced = consume_force_flag()

    due = (now - last) >= THROTTLE_SECS
    if not (forced or due):
        return 0.0  # neutral when not due, no penalty selecting dream when not due

    # ----------------------------
    # REFLECT
    # ----------------------------
    events = _tail_events(LOG_PATH, WINDOW)
    counts = defaultdict(int)
    reward_sum = defaultdict(float)
    errors = defaultdict(int)
    notes_count = 0

    for e in events:
        kind = e.get("kind")
        if kind == "tick":
            a = e.get("action")
            if a is not None:
                counts[a] += 1
                try:
                    reward_sum[a] += float(e.get("reward", 0.0))
                except Exception:
                    pass
        elif kind == "skill_error":
            a = e.get("action")
            if a is not None:
                errors[a] += 1
        elif kind == "note":
            notes_count += 1

    avgs = {}
    for a, c in counts.items():
        if c > 0:
            avgs[a] = reward_sum[a] / c

    # ----------------------------
    # LEARN (off-policy nudges)
    # ----------------------------
    q = s.setdefault("q", {})
    n = s.setdefault("n", {})

    # Nudge Q toward observed averages
    for a, avg in avgs.items():
        old = float(q.get(a, 0.0))
        q[a] = (1.0 - DREAM_ALPHA) * old + DREAM_ALPHA * float(avg)

    # Penalize actions that error'd repeatedly
    for a, err_cnt in errors.items():
        if err_cnt >= ERROR_THRESHOLD:
            q[a] = min(float(q.get(a, 0.0)), ERROR_PENALTY_Q)

    # ----------------------------
    # EVOLVE (epsilon anneal)
    # ----------------------------
    eps = float(s.get("epsilon", 0.2))
    eps = max(EPS_FLOOR, eps * EPS_DECAY)
    s["epsilon"] = eps

    # ----------------------------
    # RECORD MEMORY
    # ----------------------------
    # build a compact summary for this dream
    top_by_avg = sorted(avgs.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_by_count = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    dream_rec = {
        "ts": now,
        "tick": int(s.get("ticks", 0)),
        "forced": bool(forced),
        "summary": {
            "notes_seen": int(notes_count),
            "top_avg_reward": [(a, round(v, 4)) for a, v in top_by_avg],
            "top_counts": top_by_count,
            "errors": dict(errors),
            "epsilon_after": round(eps, 4)
        }
    }

    dreams = s.setdefault("dreams", [])
    dreams.append(dream_rec)
    if len(dreams) > 100:
        del dreams[:len(dreams)-100]

    s["last_dream_ts"] = now
    s["dream_count"] = int(s.get("dream_count", 0)) + 1

    # Modest positive reward for completing a dream successfully
    return 0.3
