# Seed-0 skill: stability_pilot — elevates exploration and rebalances preferences when stuck.
# Goal: Stabilise learning by disrupting repetitive loops and letting dream-driven variance matter.
# Safe: stdlib-only, local state edits (epsilon/Q). No network/OS changes.
#
# Mechanism:
# - Inspect recent actions -> entropy, dominance, and longest streak.
# - If trapped: bump epsilon (capped), damp the dominant action's Q, raise a soft floor for underused actions,
#   and create a short "burst window" where alternatives are favored.
# - Dream synergy: if dream Q<0 yet the 24h window is almost due, temporarily lift Q(dream) so it can fire once.
# - If healthy: gently cool epsilon toward a floor (never below).
# - Neutral reward unless corrective action taken (+0.05).
#
# Drop-in: save to skills/, core will hot-reload automatically.

from pathlib import Path
from collections import deque, Counter
import json, time, math

ACT_NAME = "stability_pilot"

LOG_PATH       = Path("data/events.log")
WINDOW_TICKS   = 800     # ~13 min @1s
RUN_EVERY      = 30      # evaluate every 30 ticks
MIN_SAMPLE     = 120     # need enough ticks to judge
DOM_FRAC       = 0.80    # dominance threshold
STREAK_MAX     = 25      # or long same-action streak
ENTROPY_LOW    = 0.65    # below -> considered stuck
EPS_FLOOR      = 0.10    # exploration never below this
EPS_CAP        = 0.35    # exploration never above this
EPS_STEP_UP    = 0.05    # bump when stuck
EPS_COOL       = 0.996   # gentle cooling when healthy
Q_DAMP         = 0.88    # damp dominant Q when correcting
Q_SOFT_FLOOR   = 0.02    # raise underused actions to at least this
Q_MAX_ABS      = 1.0     # clamp Q into [-1, 1]
BURST_TICKS    = 90      # duration to favor variety after correction
DREAM_BIAS_WIN = 180     # seconds to next dream when we lift its Q a bit
DREAM_Q_LIFT   = 0.05    # temporary lift target for dream

def _tail_actions(path: Path, want: int):
    evts = deque(maxlen=want * 2)
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

def _normalized_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    k = len(counter)
    if total <= 0 or k <= 1:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p + 1e-12)
    hmax = math.log(k + 1e-12)
    return float(h / hmax) if hmax > 0 else 0.0

def _secs_to_next_dream(state) -> float:
    last = float(state.get("last_dream_ts", 0.0))
    if last <= 0:
        return 0.0  # allow first dream
    rem = 86400.0 - (time.time() - last)
    return rem

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0

    actions = _tail_actions(LOG_PATH, WINDOW_TICKS)
    total = len(actions)

    # Keep epsilon in a safe band
    eps = float(s.get("epsilon", 0.2))
    if eps < EPS_FLOOR:
        s["epsilon"] = EPS_FLOOR

    q = s.setdefault("q", {})
    corrected = False

    if total >= MIN_SAMPLE:
        cnt = Counter(actions)
        (top, topc) = cnt.most_common(1)[0]
        frac = topc / total
        streak = _longest_streak(actions)
        h_norm = _normalized_entropy(cnt)

        stuck = (frac >= DOM_FRAC) or (streak >= STREAK_MAX) or (h_norm < ENTROPY_LOW)

        if stuck:
            # Exploration burst
            s["epsilon"] = min(float(s.get("epsilon", EPS_FLOOR)) + EPS_STEP_UP, EPS_CAP)
            # Damp dominant action Q so others get a chance
            if top in q:
                q[top] = float(q.get(top, 0.0)) * Q_DAMP
            # Raise underused actions to a soft floor
            for a in cnt:
                if a != top and float(q.get(a, 0.0)) < Q_SOFT_FLOOR:
                    q[a] = Q_SOFT_FLOOR
            # Begin short burst window
            s["stability_burst_until"] = ticks + BURST_TICKS
            corrected = True
        else:
            # Healthy → cool epsilon gently but never below floor
            s["epsilon"] = max(EPS_FLOOR, float(s.get("epsilon", EPS_FLOOR)) * EPS_COOL)

        # Dream synergy: if dream Q is negative and due soon, allow it to trigger once
        rem = _secs_to_next_dream(s)
        if 0.0 <= rem <= DREAM_BIAS_WIN:
            dq = float(q.get("dream", 0.0))
            if dq < DREAM_Q_LIFT:
                q["dream"] = DREAM_Q_LIFT

        # During burst window, keep a mild preference for a couple of alternatives
        burst_until = int(s.get("stability_burst_until", 0))
        if ticks < burst_until:
            # Favor non-dominant, non-heartbeat actions mildly
            for a in list(q.keys()):
                if a not in ("heartbeat", top):
                    q[a] = max(float(q.get(a, 0.0)), Q_SOFT_FLOOR)

        # Clamp Q-values
        for k, v in list(q.items()):
            if v > Q_MAX_ABS:
                q[k] = Q_MAX_ABS
            elif v < -Q_MAX_ABS:
                q[k] = -Q_MAX_ABS

        # Trace for inspection
        s["stability_pilot_last"] = {
            "tick": ticks,
            "sample": total,
            "dominant": top,
            "dominant_frac": round(frac, 3),
            "longest_streak": streak,
            "entropy": round(h_norm, 3),
            "epsilon_after": round(float(s.get("epsilon", 0.0)), 4),
            "dream_secs_remaining": round(rem, 1),
            "q_dream": round(float(q.get("dream", 0.0)), 4),
            "corrected": corrected
        }
    else:
        # Not enough evidence: just cool epsilon slightly while keeping the floor
        s["epsilon"] = max(EPS_FLOOR, float(s.get("epsilon", EPS_FLOOR)) * EPS_COOL)
        s["stability_pilot_last"] = {"tick": ticks, "sample": total, "note": "insufficient_sample"}

    return 0.05 if corrected else 0.0
