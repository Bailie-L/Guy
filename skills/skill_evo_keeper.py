# skills/skill_evo_keeper.py
# Purpose: Keep core self-evolution actions (codegen/bundle/verify) from starving.
# Strategy:
#   - Inspect the last WINDOW ticks (local, stdlib only).
#   - Enforce per-action floors and an overall target share.
#   - If below thresholds, append a few small reward nudges to data/inbox.txt.
#   - Writes are rate-limited and capped per run. Safe & reversible.
#
# Notes:
#   - Excludes cleanup_orphans from EVOL_ACTIONS.
#   - Floors are per-window minimum counts for each action.
#   - REWARD_BONUS makes bundle/verify nudges slightly stronger than coder.

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json, time, math

ACT_NAME = "evo_keeper"

ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / "data"
INBOX = DATA / "inbox.txt"
STORE = DATA / "evo_keeper.json"

# --- Tunables ---------------------------------------------------------------

EVOL_ACTIONS = ["self_coder", "self_bundle", "self_verify"]   # No cleanup_orphans
WINDOW       = 25                                              # lookback window (ticks)
MIN_WINDOW   = 10                                              # need at least this many ticks to act

# Overall share target for EVOL actions across the window (e.g., ~3 of 25 = 12%)
TARGET_RATIO       = 0.12

# Gentle base nudge size; per-action bonuses are applied below.
DELTA_REWARD       = 0.015

# Slightly prioritize bundling & verification over coder.
REWARD_BONUS       = {"self_bundle": 1.5, "self_verify": 1.2, "self_coder": 1.0}

# Per-window floors: ensure at least these many occurrences (if the window is large enough).
FLOOR_COUNTS       = {"self_bundle": 3, "self_verify": 2, "self_coder": 1}

# Safety rails
MAX_WRITES_PER_RUN = 3          # max reward lines we can append per act()
WRITE_COOLDOWN_S   = 30.0       # minimal time between writes

# --- Helpers ----------------------------------------------------------------

def _load_json(p: Path, default: Any):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def _save_json_atomic(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)

def _append_inbox(lines: List[str]) -> None:
    if not lines:
        return
    INBOX.parent.mkdir(parents=True, exist_ok=True)
    with INBOX.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")

# --- Core -------------------------------------------------------------------

def act(ctx) -> Optional[float]:
    s: Dict[str, Any] = ctx.get("state", {})
    raw_last: List[Dict[str, Any]] = s.get("last_actions", []) or []
    last = raw_last[-WINDOW:] if raw_last else []
    w = len(last)

    store = _load_json(STORE, {"last_write": 0.0})
    now = time.time()

    # Not enough evidence yet, or still in cooldown
    if w < MIN_WINDOW:
        s.setdefault("notes", []).append(f"evo_keeper: window={w} (<MIN_WINDOW), no-op")
        return 0.003

    last_write = float(store.get("last_write", 0.0))
    cooling = (now - last_write) <= WRITE_COOLDOWN_S

    # Count EVOL actions in the window
    counts: Dict[str, int] = {a: 0 for a in EVOL_ACTIONS}
    for x in last:
        a = x.get("action")
        if a in counts:
            counts[a] += 1

    evol_total = sum(counts.values())
    target_total = max(1, int(math.ceil(TARGET_RATIO * w)))

    # Compute per-action floor deficits (only if window is large enough to matter)
    floor_deficits: List[Tuple[str, int]] = []
    for a in EVOL_ACTIONS:
        floor = int(FLOOR_COUNTS.get(a, 0))
        deficit = max(0, floor - counts.get(a, 0))
        if deficit > 0:
            floor_deficits.append((a, deficit))

    # Do we need to nudge?
    need_ratio = evol_total < target_total
    need_floors = len(floor_deficits) > 0

    wrote = False
    lines: List[str] = []
    q: Dict[str, float] = s.get("q", {})

    if not cooling and (need_ratio or need_floors):
        # 1) Satisfy floors first (lowest-Q first helps the neglected ones)
        floor_deficits.sort(key=lambda t: float(q.get(t[0], 0.0)))
        writes = 0
        for a, deficit in floor_deficits:
            if writes >= MAX_WRITES_PER_RUN:
                break
            amt = DELTA_REWARD * float(REWARD_BONUS.get(a, 1.0))
            lines.append(f"reward {a} {amt:.4f}")
            writes += 1

        # 2) If still under overall EVOL share target, top up with lowest-Q actions
        if writes < MAX_WRITES_PER_RUN and evol_total + writes < target_total:
            to_boost = sorted(EVOL_ACTIONS, key=lambda a: float(q.get(a, 0.0)))
            for a in to_boost:
                if writes >= MAX_WRITES_PER_RUN or evol_total + writes >= target_total:
                    break
                amt = DELTA_REWARD * float(REWARD_BONUS.get(a, 1.0))
                lines.append(f"reward {a} {amt:.4f}")
                writes += 1

        if lines:
            _append_inbox(lines)
            store["last_write"] = now
            _save_json_atomic(STORE, store)
            wrote = True

    # Observability note
    counts_str = ", ".join(f"{k}:{counts[k]}" for k in EVOL_ACTIONS)
    floors_str = ", ".join(f"{a}>{FLOOR_COUNTS.get(a,0)}" for a in EVOL_ACTIONS)
    nudges_str = "; ".join(lines) if lines else "none"
    s.setdefault("notes", []).append(
        f"evo_keeper: window={w} counts[{counts_str}] floors[{floors_str}] "
        f"target_total={target_total} evol_total={evol_total} "
        f"cooldown={'yes' if cooling else 'no'} nudges={nudges_str}"
    )

    # Small positive reward so the bandit keeps this maintenance skill alive
    return 0.003
