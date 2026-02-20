# Seed-0 skill: policy_optimizer — adaptive exploration & learning-rate tuning.
# Goal: Improve adaptive potential by addressing policy optimization traps.
# Safe: stdlib-only, local edits to state (epsilon/alpha/q). No network/OS changes.
#
# What it does (every RUN_EVERY ticks):
# 1) Reads a recent window of tick events to estimate dominance, entropy, and non-stationarity.
# 2) If stuck: gently raise epsilon (≤ EPS_CAP), shrink dominant Q, lift underused actions to a soft floor.
# 3) Adaptive alpha: increase when behavior/rewards look non-stationary, decrease when stable (keeps learning responsive).
# 4) UCB-style optimism: blend a small exploration bonus into Q for low-count actions (encourages trying them).
# 5) Soft temperature mix: when highly peaked, blend Qs slightly toward the mean to reduce overconfidence.
# Returns +0.03 only when corrective action applied; otherwise 0.0.

from pathlib import Path
from collections import deque, Counter
import json, math

ACT_NAME = "policy_optimizer"

LOG_PATH      = Path("data/events.log")
WINDOW_TICKS  = 800      # ~13 min at 1s/tick
RUN_EVERY     = 60       # run once per minute
MIN_SAMPLE    = 120
DOM_FRAC      = 0.80
ENTROPY_LOW   = 0.65
STREAK_MAX    = 25

# Epsilon controls
EPS_FLOOR     = 0.10
EPS_CAP       = 0.35
EPS_STEP_UP   = 0.05
EPS_COOL      = 0.996

# Alpha controls
ALPHA_MIN     = 0.10
ALPHA_MAX     = 0.50
ALPHA_DECAY   = 0.995  # when stable, drift alpha down a bit (but not below min)

# Q adjustments
Q_DAMP        = 0.90   # shrink dominant Q by this when stuck
Q_SOFT_FLOOR  = 0.02   # raise underused actions to at least this
Q_TEMP_MIX    = 0.05   # blend toward mean Q when distribution is very peaked
Q_UCB_COEF    = 0.10   # strength of optimism bonus
Q_UCB_BLEND   = 0.10   # how much of the bonus to blend into Q
Q_MAX_ABS     = 1.0

def _tail_ticks(path: Path, want: int):
    evts = deque(maxlen=want*2)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("kind") == "tick":
                    evts.append(e)
            except Exception:
                pass
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
        if p > 0:
            h -= p * math.log(p + 1e-12)
    hmax = math.log(k + 1e-12)
    return float(h / hmax) if hmax > 0 else 0.0

def _nonstationarity(evts):
    # compare avg reward of first vs second half of window
    if not evts:
        return 0.0
    m = len(evts) // 2
    def avg(arr):
        s = 0.0; n = 0
        for e in arr:
            try:
                s += float(e.get("reward", 0.0)); n += 1
            except Exception:
                pass
        return (s / n) if n else 0.0
    a1 = avg(evts[:m]); a2 = avg(evts[m:])
    return abs(a2 - a1)

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0

    # Ensure base fields
    q = s.setdefault("q", {})
    n = s.setdefault("n", {})
    eps = float(s.get("epsilon", 0.2))
    alpha = float(s.get("alpha", 0.3))

    if eps < EPS_FLOOR:
        s["epsilon"] = EPS_FLOOR

    evts = _tail_ticks(LOG_PATH, WINDOW_TICKS)
    total = len(evts)
    if total < MIN_SAMPLE:
        # Not enough data; gently cool toward floors
        s["epsilon"] = max(EPS_FLOOR, float(s.get("epsilon", EPS_FLOOR)) * EPS_COOL)
        s["alpha"] = max(ALPHA_MIN, min(ALPHA_MAX, alpha * ALPHA_DECAY))
        s["policy_optimizer_last"] = {"tick": ticks, "sample": total, "note": "insufficient_sample"}
        return 0.0

    actions = [e.get("action") for e in evts if e.get("action")]
    cnt = Counter(actions)
    (top, topc) = cnt.most_common(1)[0]
    frac = topc / len(actions)
    streak = _longest_streak(actions)
    ent = _normalized_entropy(cnt)
    ns = _nonstationarity(evts)  # absolute change in avg reward between halves

    corrected = False

    # 1) If stuck: intervene (epsilon up, damp dominant, lift underused)
    if frac >= DOM_FRAC or streak >= STREAK_MAX or ent < ENTROPY_LOW:
        s["epsilon"] = min(float(s.get("epsilon", eps)) + EPS_STEP_UP, EPS_CAP)
        if top in q:
            q[top] = float(q.get(top, 0.0)) * Q_DAMP
        for a in cnt:
            if a != top and float(q.get(a, 0.0)) < Q_SOFT_FLOOR:
                q[a] = Q_SOFT_FLOOR
        corrected = True
    else:
        # 2) Healthy: cool exploration slightly (but keep floor)
        s["epsilon"] = max(EPS_FLOOR, float(s.get("epsilon", eps)) * EPS_COOL)

    # 3) Adaptive alpha (learning rate): raise when non-stationary, lower when stable
    # Map ns (0..~0.3) to [ALPHA_MIN, ALPHA_MAX]
    ns_clamped = max(0.0, min(0.3, ns))
    target_alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (ns_clamped / 0.3)
    # Smoothly move alpha 10% toward target
    s["alpha"] = max(ALPHA_MIN, min(ALPHA_MAX, alpha * 0.9 + target_alpha * 0.1))

    # 4) UCB-style optimism: add small bonus to rarely tried actions (based on n)
    # We blend a fraction of (q + bonus) back into q to avoid runaway changes.
    N = sum(int(n.get(a, 0)) for a in cnt) + 1
    if N > 1:
        for a in cnt:
            na = max(1, int(n.get(a, 0)))
            bonus = Q_UCB_COEF * math.sqrt(math.log(N + 1.0) / na)
            q[a] = float(q.get(a, 0.0)) * (1.0 - Q_UCB_BLEND) + (float(q.get(a, 0.0)) + bonus) * Q_UCB_BLEND

    # 5) Soft temperature mix when highly peaked: bring Qs slightly toward their mean
    if frac >= 0.85 or ent < 0.55:
        if q:
            mean_q = sum(float(v) for v in q.values()) / max(1, len(q))
            for a in list(q.keys()):
                q[a] = float(q[a]) * (1.0 - Q_TEMP_MIX) + mean_q * Q_TEMP_MIX
        corrected = True or corrected

    # Clamp Q-values
    for k, v in list(q.items()):
        if v > Q_MAX_ABS:
            q[k] = Q_MAX_ABS
        elif v < -Q_MAX_ABS:
            q[k] = -Q_MAX_ABS

    s["policy_optimizer_last"] = {
        "tick": ticks,
        "sample": total,
        "dominant": top,
        "dominant_frac": round(frac, 3),
        "longest_streak": streak,
        "entropy": round(ent, 3),
        "nonstationarity": round(ns, 4),
        "epsilon_after": round(float(s.get("epsilon", 0.0)), 4),
        "alpha_after": round(float(s.get("alpha", 0.0)), 4),
        "corrected": bool(corrected)
    }

    return 0.03 if corrected else 0.0
