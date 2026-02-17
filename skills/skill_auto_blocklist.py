# Hard block auto_skill actions whenever the gate is closed.
# - Pushes all auto_skill_* Q-values to a tiny floor so the picker ignores them.
# - NO exploration (epsilon) changes.
from pathlib import Path
import json

ACT_NAME = "auto_blocklist"
REGISTRY_FILE = Path("data/self_coder_registry.json")

def _load_registry():
    try:
        return json.loads(REGISTRY_FILE.read_text())
    except Exception:
        return {}

def act(ctx):
    s = ctx.get("state", {})
    reg = _load_registry()
    budget = reg.get("budget", {})
    enabled = bool(budget.get("enabled", True))

    # Only act while the gate is CLOSED (over the 5% cap)
    if enabled:
        return 0.0

    q = s.get("q", {}) or {}
    changed = 0

    # Demote every auto_skill_* so the selector stops choosing them
    for a in list(q.keys()):
        if isinstance(a, str) and a.startswith("auto_skill_"):
            if q.get(a, 0.0) != 1e-12:
                q[a] = 1e-12
                changed += 1

    if changed:
        s.setdefault("notes", []).append("auto_blocklist: demoted auto_skill_* while over-cap (no epsilon change)")
        return 0.002

    return 0.0
