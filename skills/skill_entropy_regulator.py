# Seed-0 skill: entropy_regulator — disrupts repetitive loops and promotes diverse exploration.
# Safe: stdlib-only, no network/OS changes. Runs periodically and adjusts epsilon + Q-values.
#
# Mechanism:
# - Reads a sliding window of recent tick actions from data/events.log.
# - Computes normalized Shannon entropy of the action distribution (0..1).
# - If entropy is LOW (or one action dominates), it bumps exploration (epsilon)
#   and softly damps the dominant action's Q while nudging underused actions.
# - If entropy is HIGH (very diverse), it gently cools epsilon toward a floor.
# - Neutral reward when no intervention; small positive reward when corrective action taken.
#
# Throttle: every RUN_EVERY ticks.

from pathlib import Path
from collections import deque, Counter
import json, math

ACT_NAME = "entropy_regulator"

LOG_PATH    = Path("data/events.log")
WINDOW      = 600       # number of recent ticks to consider (~10 min at 1s)
RUN_EVERY   = 60        # run every 60 ticks (~1 min)
ENTROPY_LOW = 0.62      # below this, treat as "stuck"
ENTROPY_HIGH= 0.92      # above this, cool exploration slightly
DOM_FRAC    = 0.80      # hard dominance threshold for top action
EPS_BUMP    = 0.07      # exploration bump on correction
EPS_CAP     = 0.35      # do not exceed this epsilon
EPS_FLOOR   = 0.08      # never go below this epsilon
EPS_COOL    = 0.995     # gentle cooling multiplier when healthy
Q_DAMP      = 0.90      # multiply dominant Q by this on correction
Q_NUDGE     = 0.02      # minimum Q for underused actions (soft floor)
MIN_SAMPLE  = 80        # need at least this many ticks to act

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
    # Return only the last `want` actions
    return list(evts)[-want:]

def _normalized_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    k = len(counts)
    if total == 0 or k <= 1:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p + 1e-12)
    hmax = math.log(k + 1e-12)
    return float(h / hmax) if hmax > 0 else 0.0

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0  # neutral most of the time

    actions = _tail_tick_actions(LOG_PATH, WINDOW)
    total = len(actions)
    eps = float(s.get("epsilon", 0.2))
    q = s.setdefault("q", {})

    # Always keep a floor on exploration
    if eps < EPS_FLOOR:
        s["epsilon"] = EPS_FLOOR

    if total < MIN_SAMPLE:
        # Not enough evidence yet; gently cool toward floor
        s["epsilon"] = max(EPS_FLOOR, float(s.get("epsilon", EPS_FLOOR)) * EPS_COOL)
        s["entropy_regulator_last"] = {"tick": ticks, "sample": total, "note": "insufficient_sample"}
        return 0.0

    c = Counter(actions)
    h_norm = _normalized_entropy(c)
    (top, topc) = c.most_common(1)[0]
    frac = topc / total

    reward = 0.0
    corrected = False

    # Condition 1: severe dominance or low entropy → intervene
    if frac >= DOM_FRAC or h_norm < ENTROPY_LOW:
        # bump exploration
        s["epsilon"] = min(float(s.get("epsilon", 0.2)) + EPS_BUMP, EPS_CAP)
        # damp dominant action so others get a chance
        if top in q:
            q[top] = float(q.get(top, 0.0)) * Q_DAMP
        # nudge underused actions up to a soft floor
        for a in c:
            if a != top:
                if float(q.get(a, 0.0)) < Q_NUDGE:
                    q[a] = Q_NUDGE
        corrected = True
        reward = +0.05

    # Condition 2: very high entropy → gently cool (but not below floor)
    elif h_norm > ENTROPY_HIGH:
        s["epsilon"] = max(EPS_FLOOR, float(s.get("epsilon", 0.2)) * EPS_COOL)

    # Bound Q-values
    for k, v in list(q.items()):
        if v > 1.0:
            q[k] = 1.0
        elif v < -1.0:
            q[k] = -1.0

    # Trace for inspection
    s["entropy_regulator_last"] = {
        "tick": ticks,
        "sample": total,
        "unique_actions": len(c),
        "dominant": top,
        "dominant_frac": round(frac, 3),
        "h_norm": round(h_norm, 3),
        "epsilon_after": round(float(s.get("epsilon", 0.0)), 4),
        "corrected": corrected
    }

    return reward if corrected else 0.0
