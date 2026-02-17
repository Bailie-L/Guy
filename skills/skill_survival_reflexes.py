# Survival reflexes: reduce suffocation risk & water pinning by preferring 'wolf_unstuck'
# Safe: local state only. No OS/network. Runs periodically and adjusts Q-values + epsilon.
from collections import Counter, deque
from pathlib import Path
import json

ACT_NAME = "survival_reflexes"
RUN_EVERY = 50          # check every 50 ticks
SEQ_WINDOW = 60         # last N actions to inspect
MIN_REPEAT = 12         # if wolf_actions seen >= this in window, treat as risky loop
LOW_Q = 0.002           # treat wolf_actions as weak if below this
EPS_BUMP = 0.04         # exploration bump on risk
EPS_CAP = 0.35
DAMP = 0.90             # multiply risky action Q by this

def _recent_actions():
    # Prefer Guy's JSON event stream if present; fall back to state history
    p = Path("data/events.log")
    acts = deque(maxlen=SEQ_WINDOW)
    if p.exists():
        for line in p.open("r", encoding="utf-8"):
            try:
                e = json.loads(line)
                if e.get("kind") == "tick" and "action" in e:
                    acts.append(e["action"])
            except Exception:
                pass
    return list(acts)

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0

    q = s.setdefault("q", {})
    eps = float(s.get("epsilon", 0.2))
    recent = _recent_actions() or list(s.get("last_actions", []))[-SEQ_WINDOW:]
    c = Counter(recent)
    wolf_repeats = c.get("wolf_actions", 0)

    # Heuristic: risky if wolf_actions repeats a lot OR its Q has gone low/negative
    risky = (wolf_repeats >= MIN_REPEAT) or (q.get("wolf_actions", 0.0) <= LOW_Q)

    # Heuristic: bad-reward phase (e.g., water pin / suffocation precursor)
    # Use last_rewards if available; else ignore silently
    bad_phase = False
    for k in ("last_rewards", "recent_rewards"):
        if isinstance(s.get(k), list) and len(s[k]) >= 10:
            # many small/negative rewards → likely struggling
            tail = s[k][-10:]
            bad_phase = sum(1 for r in tail if r is not None and r <= 0.0) >= 7
            if bad_phase: break

    reward = 0.0
    if risky or bad_phase:
        # Prefer escape mechanics
        q["wolf_unstuck"] = max(float(q.get("wolf_unstuck", 0.0)), 0.08)

        # Soften the risky action so it’s selected less
        if "wolf_actions" in q:
            q["wolf_actions"] = float(q["wolf_actions"]) * DAMP

        # Briefly explore alternatives harder
        s["epsilon"] = min(EPS_CAP, eps + EPS_BUMP)

        # Trace
        notes = s.setdefault("notes", [])
        notes.append(f"SurvivalReflex: risk={'risky' if risky else ''}{' bad' if bad_phase else ''} repeats={wolf_repeats} q_wolf={q.get('wolf_actions', 0.0):.4f} -> boost unstuck, bump epsilon at tick {ticks}")

        reward = 0.02  # reinforce safety behavior

    # Bound Q-values
    for k,v in list(q.items()):
        if v > 1.0: q[k] = 1.0
        if v < -1.0: q[k] = -1.0

    return reward
