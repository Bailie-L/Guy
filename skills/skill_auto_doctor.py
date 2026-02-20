# Seed-0 skill: auto_doctor — self-corrects policy collapse without touching network/OS.
# Effects (safe, local, reversible via state.json):
# - Detects dominance (≥80% one action in last WINDOW ticks) and bumps exploration (epsilon)
#   while softly damping the dominant action's Q so others get a chance.
# - Adds tiny schedule nudges so maintenance skills fire near their windows.
# - Gently anneals epsilon back toward a floor when system looks healthy.
#
# Throttle: runs every RUN_EVERY ticks. Neutral reward unless corrective action taken.

from pathlib import Path
from collections import deque, Counter
import json

ACT_NAME = "auto_doctor"

LOG_PATH   = Path("data/events.log")
WINDOW     = 600      # ticks to analyze
RUN_EVERY  = 60       # run every 60 ticks (~1 min)
DOMINANCE  = 0.80     # if top action ≥ 80% of window → corrective action
EPS_BUMP   = 0.05     # added to epsilon when stuck
EPS_CAP    = 0.35     # never exceed this
EPS_FLOOR  = 0.10     # never go below this (keeps exploration alive)
EPS_COOLDN = 0.997    # gentle decay toward floor when healthy
Q_DAMP     = 0.90     # multiply dominant Q by this when correcting
HEARTBEAT  = "heartbeat"

def _tail_ticks(path: Path, want: int):
    evts = deque(maxlen=want*2)
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("kind") == "tick":
                    evts.append(e)
            except Exception:
                pass
    return list(evts)[-want:]

def _next_due(ticks: int, mod: int) -> int:
    return (-ticks) % mod

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0  # neutral most of the time

    q = s.setdefault("q", {})
    eps = float(s.get("epsilon", 0.2))

    recent = _tail_ticks(LOG_PATH, WINDOW)
    actions = [e.get("action") for e in recent if e.get("action")]
    total = len(actions)
    if total < 50:
        # Not enough evidence; gently keep epsilon from dying
        s["epsilon"] = max(EPS_FLOOR, eps * EPS_COOLDN)
        return 0.0

    c = Counter(actions)
    (top, topc) = c.most_common(1)[0]
    frac = topc / total

    reward = 0.0
    corrected = False

    # Correct dominance
    if frac >= DOMINANCE:
        # bump exploration (cap), damp dominant Q
        s["epsilon"] = min(eps + EPS_BUMP, EPS_CAP)
        if top in q:
            q[top] = float(q.get(top, 0.0)) * Q_DAMP
        # If heartbeat is the culprit, give small lifts to alternatives
        if top == HEARTBEAT:
            q["reflect"] = max(float(q.get("reflect", 0.0)), 0.02)
            q["communicate"] = max(float(q.get("communicate", 0.0)), 0.03)
        corrected = True
        reward = +0.05  # mark corrective action

    else:
        # healthy → slowly cool epsilon but never below floor
        s["epsilon"] = max(EPS_FLOOR, eps * EPS_COOLDN)

    # Schedule nudges: help the agent try maintenance right near their windows
    nb = _next_due(ticks, 120)
    nv = _next_due(ticks, 180)
    if nb <= 2:
        q["self_bundle"] = max(float(q.get("self_bundle", 0.0)), 0.10)
    if nv <= 2:
        q["self_verify"] = max(float(q.get("self_verify", 0.0)), 0.10)

    # Keep Q-values bounded
    for k, v in list(q.items()):
        if v > 1.0: q[k] = 1.0
        if v < -1.0: q[k] = -1.0

    # Trace
    s["auto_doctor_last"] = {
        "tick": ticks,
        "dominant": top,
        "frac": round(frac, 3),
        "epsilon_after": s["epsilon"]
    }

    return reward if corrected else 0.0
