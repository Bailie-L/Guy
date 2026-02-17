# Seed-0 skill: entropy_guard
# Unsticks policy if one action dominates: bumps epsilon and dampens the top action's Q.
from pathlib import Path
from collections import Counter, deque
import json

ACT_NAME = "entropy_guard"

LOG_PATH = Path("data/events.log")
TAIL = 500                # reflect window
RUN_EVERY = 120           # ticks
DOMINANCE = 0.80          # if top action > 80% of last TAIL ticks → unstick
EPS_BUMP = 0.05           # how much to add (capped later)
EPS_CAP  = 0.35           # never exceed this
Q_DAMP   = 0.97           # multiply dominant Q by this (soft push)

def _tail_actions(path: Path, max_lines: int):
    evts = deque(maxlen=max_lines)
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("kind") == "tick":
                    evts.append(e.get("action"))
            except Exception:
                pass
    return list(evts)

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0  # neutral most of the time

    acts = _tail_actions(LOG_PATH, TAIL)
    if not acts:
        return 0.0

    c = Counter(a for a in acts if a)
    total = sum(c.values()) or 1
    top, topc = None, 0
    for a, cnt in c.most_common(1):
        top, topc = a, cnt

    if top and (topc / total) >= DOMINANCE and total >= 50:
        # bump epsilon
        eps = float(s.get("epsilon", 0.2))
        s["epsilon"] = min(eps + EPS_BUMP, EPS_CAP)
        # softly dampen dominant Q so others get a look-in
        q = s.setdefault("q", {})
        if top in q:
            q[top] = float(q[top]) * Q_DAMP
        # tiny positive to mark that we took corrective action
        return +0.05

    return 0.0
