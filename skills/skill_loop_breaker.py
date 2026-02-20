# Seed-0 skill: loop_breaker — disrupts repetitive action loops and promotes exploration.
# Safe: stdlib-only, local state edits (epsilon/Q). No network/OS changes.
#
# Mechanism:
# - Reads recent tick actions from data/events.log.
# - Detects dominance (top action share) and long same-action streaks.
# - When stuck: bumps epsilon (capped) and softly damps the dominant action's Q,
#   while nudging underused actions up to a soft floor.
# - When healthy: gently cools epsilon toward a floor (never hits 0).
# - Neutral reward unless corrective action taken (+0.05).
#
# Drop-in: put in skills/, the core hot-reloads it. No main.py changes required.

from pathlib import Path
from collections import deque, Counter
import json

ACT_NAME = "loop_breaker"

LOG_PATH    = Path("data/events.log")
WINDOW      = 600        # last ~10 minutes at 1s/tick
RUN_EVERY   = 45         # evaluate every 45 ticks
MIN_SAMPLE  = 80         # need at least this many ticks to act
DOM_FRAC    = 0.78       # treat as stuck if top action >= 78% of window
STREAK_MAX  = 20         # or if any same-action streak >= 20
EPS_BUMP    = 0.06       # exploration bump when stuck
EPS_CAP     = 0.35       # exploration cap
EPS_FLOOR   = 0.08       # exploration floor
EPS_COOL    = 0.997      # gentle cooling when healthy
Q_DAMP      = 0.88       # multiply dominant Q by this when correcting
Q_NUDGE     = 0.02       # soft floor for underused actions' Q
Q_MAX_ABS   = 1.0        # bound Q in [-1, +1]

def _tail_tick_actions(path: Path, want: int):
    evts = deque(maxlen=want*2)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("kind") == "tick":
                a = e.get("action")
                if a:
                    evts.append(a)
    # return only the last `want` actions in chronological order
    return list(evts)[-want:]

def _longest_streak(seq):
    if not seq:
        return 0
    best = 1
    cur = 1
    prev = seq[0]
    for a in seq[1:]:
        if a == prev:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
            prev = a
    return best

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0  # neutral most of the time

    actions = _tail_tick_actions(LOG_PATH, WINDOW)
    total = len(actions)
    eps = float(s.get("epsilon", 0.2))
    q = s.setdefault("q", {})

    # keep exploration above floor
    if eps < EPS_FLOOR:
        s["epsilon"] = EPS_FLOOR

    if total < MIN_SAMPLE:
        # not enough evidence yet; gently cool toward floor
        s["epsilon"] = max(EPS_FLOOR, float(s.get("epsilon", EPS_FLOOR)) * EPS_COOL)
        s["loop_breaker_last"] = {"tick": ticks, "sample": total, "note": "insufficient_sample"}
        return 0.0

    cnt = Counter(actions)
    (top, topc) = cnt.most_common(1)[0]
    frac = topc / total
    streak = _longest_streak(actions)

    corrected = False

    # condition: dominance or long streak → intervene
    if frac >= DOM_FRAC or streak >= STREAK_MAX:
        # bump exploration (cap)
        s["epsilon"] = min(float(s.get("epsilon", eps)) + EPS_BUMP, EPS_CAP)
        # damp dominant Q so others get a chance
        if top in q:
            q[top] = float(q.get(top, 0.0)) * Q_DAMP
        # nudge underused actions to a soft floor
        for a in cnt:
            if a != top and float(q.get(a, 0.0)) < Q_NUDGE:
                q[a] = Q_NUDGE
        corrected = True
    else:
        # healthy → cool epsilon but not below floor
        s["epsilon"] = max(EPS_FLOOR, float(s.get("epsilon", eps)) * EPS_COOL)

    # bound Q-values
    for k, v in list(q.items()):
        if v > Q_MAX_ABS:
            q[k] = Q_MAX_ABS
        elif v < -Q_MAX_ABS:
            q[k] = -Q_MAX_ABS

    # trace
    s["loop_breaker_last"] = {
        "tick": ticks,
        "sample": total,
        "dominant": top,
        "dominant_frac": round(frac, 3),
        "longest_streak": streak,
        "epsilon_after": round(float(s.get("epsilon", 0.0)), 4),
        "corrected": corrected
    }

    return 0.05 if corrected else 0.0
