# skills/skill_greenlight.py
# Purpose: Automatically "greenlight" a single promising self-coded skill for fair testing,
#          and bench (retire) obvious duds — keeping exploration sharp and stability intact.
#
# What it does each run (cheap, stdlib-only):
# 1) Reads last N events to compute a non-auto baseline avg reward.
# 2) Among probationary auto_skills: pick one with >= MIN_USES_CANDIDATE and avg > baseline.
#    -> sets it as registry["designated_candidate"] (only if budget is enabled).
# 3) Retires probationary auto_skills with >= RETIRE_ZERO_USES and all-zero rewards.
# 4) Leaves everything else alone. Writes results to data/self_coder_registry.json.
#
# Safety: touches only data/*.json; never edits skills/*.py or state.q values.

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

ACT_NAME = "greenlight"

# --- Files ---
DATA_DIR = Path("data")
EVENTS_LOG = DATA_DIR / "events.log"
REGISTRY_FILE = DATA_DIR / "self_coder_registry.json"

# --- Tunables ---
WINDOW_EVENTS = 1500            # how many recent events to evaluate
MIN_USES_CANDIDATE = 3          # minimum uses to consider a skill for candidate
CANDIDATE_MIN_DELTA = 0.0       # require avg > baseline (set >0.0 to be stricter)
RETIRE_ZERO_USES = 3            # retire if >= this uses AND all rewards are zero
TICK_COOLDOWN = 50              # light throttle (ticks) between heavier passes
SMALL_REWARD_ON_CHANGE = 0.005  # reward when we actually make a change

# --- Helpers ---

def _jload(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def _jsave(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def _iter_recent_events(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", errors="ignore") as f:
        for line in f.readlines()[-limit:]:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def _is_auto(action: Any) -> bool:
    return isinstance(action, str) and action.startswith("auto_skill_")

def _mean(vals: List[float]) -> float:
    vs = [v for v in vals if isinstance(v, (int, float))]
    return (sum(vs) / len(vs)) if vs else 0.0

# --- Core logic ---

def _compute_baseline_and_auto_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    non_auto_rewards: List[float] = []
    per_auto: Dict[str, List[float]] = {}
    for d in events:
        a = d.get("action")
        r = d.get("reward")
        rv = None
        try:
            rv = float(r) if r is not None else None
        except Exception:
            rv = None

        if _is_auto(a):
            if isinstance(rv, (int, float)):
                per_auto.setdefault(str(a), []).append(rv)
            else:
                per_auto.setdefault(str(a), []).append(0.0)
        else:
            if isinstance(rv, (int, float)):
                non_auto_rewards.append(rv)

    baseline = _mean(non_auto_rewards)
    auto_stats: Dict[str, Dict[str, Any]] = {}
    for name, rs in per_auto.items():
        uses = len(rs)
        avg = _mean(rs)
        nonzero_rate = (100.0 * sum(1 for x in rs if x != 0.0) / uses) if uses else 0.0
        auto_stats[name] = {"uses": uses, "avg": avg, "nonzero_rate": nonzero_rate}
    return {"baseline": baseline, "auto_stats": auto_stats}

def _choose_candidate(auto_stats: Dict[str, Dict[str, Any]],
                      registry: Dict[str, Any],
                      baseline: float) -> Optional[str]:
    # Only choose a candidate if the budget gate is enabled (under cap).
    budget = registry.get("budget", {})
    if not budget.get("enabled", True):
        return None

    skills = registry.get("skills", {})
    best_name: Optional[str] = None
    best_avg = None

    for name, st in auto_stats.items():
        meta = skills.get(name, {"status": "probation"})
        if meta.get("status", "probation") != "probation":
            continue
        uses = int(st.get("uses", 0))
        avg = float(st.get("avg", 0.0))
        if uses >= MIN_USES_CANDIDATE and (avg - baseline) > CANDIDATE_MIN_DELTA:
            if best_avg is None or avg > best_avg:
                best_name, best_avg = name, avg

    return best_name

def _retire_duds(auto_stats: Dict[str, Dict[str, Any]],
                 registry: Dict[str, Any]) -> List[str]:
    skills = registry.setdefault("skills", {})
    retired: List[str] = []
    for name, st in auto_stats.items():
        meta = skills.get(name, {"status": "probation"})
        if meta.get("status", "probation") != "probation":
            continue
        uses = int(st.get("uses", 0))
        avg = float(st.get("avg", 0.0))
        nonzero = float(st.get("nonzero_rate", 0.0))
        # retire if enough uses and clearly dead (all zero implies avg==0 and nonzero==0)
        if uses >= RETIRE_ZERO_USES and avg == 0.0 and nonzero == 0.0:
            meta["status"] = "retired"
            skills[name] = meta
            retired.append(name)
    return retired

# --- Public skill API ---

def act(ctx: Dict[str, Any]) -> float:
    """
    Main entrypoint: evaluates recent performance and updates the self-coder registry
    to (a) retire obvious duds and (b) select one promising probationer as candidate.
    """
    state = ctx.get("state", {})
    ticks = int(state.get("ticks", 0))

    # Throttle: avoid heavy work too often
    reg = _jload(REGISTRY_FILE, {})
    last_tick = int(reg.get("greenlight_last_tick", -10**9))
    if ticks - last_tick < TICK_COOLDOWN:
        return 0.0

    # Load recent events and compute metrics
    events = _iter_recent_events(EVENTS_LOG, WINDOW_EVENTS)
    metrics = _compute_baseline_and_auto_stats(events)
    baseline = float(metrics.get("baseline", 0.0))
    auto_stats = metrics.get("auto_stats", {})

    changed = False

    # Bench obvious duds
    retired = _retire_duds(auto_stats, reg)
    if retired:
        changed = True

    # Pick a single candidate (only when budget enabled)
    new_cand = _choose_candidate(auto_stats, reg, baseline)
    old_cand = reg.get("designated_candidate")
    if new_cand != old_cand:
        reg["designated_candidate"] = new_cand
        changed = True

    # Stamp and save
    reg["greenlight_last_tick"] = ticks
    _jsave(REGISTRY_FILE, reg)

    # Optional transparency note
    notes = state.setdefault("notes", [])
    if retired:
        notes.append(f"greenlight: retired {len(retired)} duds")
    if new_cand and new_cand != old_cand:
        notes.append(f"greenlight: candidate -> {new_cand}")
    if not new_cand and old_cand:
        notes.append("greenlight: candidate cleared")

    return SMALL_REWARD_ON_CHANGE if changed else 0.0
