# skills/skill_self_coder.py
# Purpose: Self-coding with a safety/quality gate so new skills help (not hurt).
# - Generates new auto_skills sparingly and only when budget allows.
# - Tracks per-skill performance from data/events.log and promotes/retires accordingly.
# - Enforces a global auto_skill budget target (<=5%) with short-lived bursts (<=10%)
#   for a single probation "candidate". Others self-throttle via a Q-fence.
# Safety: stdlib-only. Touches only data/*.json, data/inbox.txt, and skills/*.py

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json
import math
import time

ACT_NAME = "self_coder"

# ---- Paths ----
ROOT = Path(".")
DATA = ROOT / "data"
SKILLS_DIR = ROOT / "skills"
EVENTS_LOG = DATA / "events.log"
PATTERNS_FILE = DATA / "patterns.json"
GOALS_FILE = DATA / "goals.json"
GENERATED_LOG = DATA / "generated_skills.json"
REGISTRY_FILE = DATA / "self_coder_registry.json"
INBOX = DATA / "inbox.txt"

# ---- Tunables (quality gate) ----
WINDOW_EVENTS = 2000           # sample size for metrics
BUDGET_TARGET_PCT = 5.0        # steady-state auto_skill % cap
BUDGET_BURST_PCT = 10.0        # one candidate may spike up to this
PROMOTE_DELTA = 0.003          # must beat baseline by this margin
PROMOTE_MIN_USES = 300         # min uses to be eligible for promotion
RETIRE_AVG_LT = 0.001          # retire if avg below this after min uses
RETIRE_MIN_USES = 100
UNDERPERF_STRIKES_MAX = 3      # consecutive underperformance gates retire
MAX_SKILLS_PER_DAY = 3         # generation rate limit
MAX_CONCURRENT_PROB = 2        # at most N active probation skills at once
Q_FENCE_DISABLED = 1e-6        # Q to pin disabled skills to (self-throttle)
NEW_SKILL_REWARD = 0.2         # reward when we successfully generate
TICK_COOLDOWN = 80             # min ticks between heavy analyses (perf)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def _now() -> float:
    return time.time()

def _is_auto(action: str) -> bool:
    return isinstance(action, str) and action.startswith("auto_skill_")

def _iter_events(path: Path, limit: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            out.append(d)
    return out[-limit:] if limit else out

def _mean(xs: List[float]) -> float:
    vals = [x for x in xs if isinstance(x, (int, float))]
    return sum(vals) / len(vals) if vals else 0.0

# -----------------------------------------------------------------------------
# Metrics & Registry maintenance
# -----------------------------------------------------------------------------

def compute_recent_metrics() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      recent: dict with auto_skill_pct, baseline_non_auto_avg, non_auto_nonzero_rate, sample_size
      per_auto: dict[name] = {uses, avg_reward, nonzero_rate, last_tick}
    """
    ev = _iter_events(EVENTS_LOG, WINDOW_EVENTS)
    if not ev:
        return (
            {
                "auto_skill_pct": 0.0,
                "baseline_non_auto_avg": 0.0,
                "non_auto_nonzero_rate": 0.0,
                "sample_size": 0,
            },
            {},
        )

    non_auto_rewards: List[float] = []
    non_auto_count = 0
    non_auto_nonzero = 0
    auto_rewards_by_name: Dict[str, List[float]] = {}
    last_tick_by_name: Dict[str, int] = {}
    auto_count = 0
    total_count = 0

    for d in ev:
        a = d.get("action", "")
        r = d.get("reward", None)
        try:
            r = float(r) if r is not None else None
        except Exception:
            r = None

        total_count += 1
        if _is_auto(a):
            auto_count += 1
            auto_rewards_by_name.setdefault(a, []).append(r if isinstance(r, (int, float)) else 0.0)
            t = d.get("tick")
            if isinstance(t, int):
                last_tick_by_name[a] = t
        else:
            if isinstance(r, (int, float)):
                non_auto_rewards.append(r)
                if r != 0.0:
                    non_auto_nonzero += 1
            non_auto_count += 1

    baseline = _mean(non_auto_rewards)
    nonzero_rate = (100.0 * non_auto_nonzero / non_auto_count) if non_auto_count else 0.0
    auto_pct = (100.0 * auto_count / total_count) if total_count else 0.0

    per_auto: Dict[str, Any] = {}
    for name, rs in auto_rewards_by_name.items():
        uses = len(rs)
        avg = _mean(rs)
        nonzero = (100.0 * sum(1 for x in rs if isinstance(x, (int, float)) and x != 0.0) / uses) if uses else 0.0
        per_auto[name] = {
            "uses": uses,
            "avg_reward": avg,
            "nonzero_rate": nonzero,
            "last_tick": last_tick_by_name.get(name, 0),
        }

    recent = {
        "auto_skill_pct": auto_pct,
        "baseline_non_auto_avg": baseline,
        "non_auto_nonzero_rate": nonzero_rate,
        "sample_size": total_count,
    }
    return recent, per_auto

def choose_candidate(per_auto: Dict[str, Any], baseline: float, registry: Dict[str, Any]) -> Optional[str]:
    """
    Choose a single probation candidate via simple UCB on probation skills.
    status in {"probation","active","retired"}; default -> probation
    """
    ranks: List[Tuple[float, str]] = []
    N = sum(v.get("uses", 0) for v in per_auto.values()) + 1
    for name, stats in per_auto.items():
        sk = registry.get("skills", {}).get(name, {})
        status = sk.get("status", "probation")
        if status != "probation":
            continue
        n = max(1, int(stats.get("uses", 0)))
        avg = float(stats.get("avg_reward", 0.0))
        bonus = math.sqrt(2.0 * math.log(N) / n)
        ucb = avg + bonus + (0.5 * max(0.0, avg - baseline))
        ranks.append((ucb, name))

    if not ranks:
        return None
    ranks.sort(reverse=True)
    return ranks[0][1]

def update_registry(recent: Dict[str, Any], per_auto: Dict[str, Any]) -> Dict[str, Any]:
    reg = _load_json(REGISTRY_FILE, {})
    skills = reg.get("skills", {})

    baseline = float(recent.get("baseline_non_auto_avg", 0.0))

    # Update stats and statuses
    for name, st in per_auto.items():
        sk = skills.get(name, {"status": "probation", "strikes": 0, "q_fence": Q_FENCE_DISABLED})
        uses = int(st.get("uses", 0))
        avg = float(st.get("avg_reward", 0.0))
        last_tick = int(st.get("last_tick", 0))

        prev_uses = int(sk.get("uses", 0))
        if uses > prev_uses and avg < baseline:
            sk["strikes"] = int(sk.get("strikes", 0)) + 1

        status = sk.get("status", "probation")
        if uses >= PROMOTE_MIN_USES and (avg - baseline) >= PROMOTE_DELTA:
            status = "active"
            sk["strikes"] = 0
        elif uses >= RETIRE_MIN_USES and avg < RETIRE_AVG_LT:
            status = "retired"
        elif int(sk.get("strikes", 0)) >= UNDERPERF_STRIKES_MAX:
            status = "retired"

        sk.update({
            "uses": uses,
            "avg_reward": avg,
            "last_tick": last_tick,
            "status": status,
        })
        skills[name] = sk

    auto_pct = float(recent.get("auto_skill_pct", 0.0))
    budget_enabled = auto_pct <= BUDGET_TARGET_PCT

    candidate = choose_candidate(per_auto, baseline, {"skills": skills}) if per_auto else None

    reg.update({
        "updated_at": _now(),
        "budget": {
            "target_pct": BUDGET_TARGET_PCT,
            "burst_pct": BUDGET_BURST_PCT,
            "auto_skill_pct": auto_pct,
            "enabled": budget_enabled,
        },
        "baseline": {
            "non_auto_avg": baseline,
            "non_auto_nonzero_rate": float(recent.get("non_auto_nonzero_rate", 0.0)),
            "sample": int(recent.get("sample_size", 0)),
        },
        "designated_candidate": candidate,
        "skills": skills,
    })

    _save_json(REGISTRY_FILE, reg)
    return reg

# -----------------------------------------------------------------------------
# Need detection
# -----------------------------------------------------------------------------

def identify_need(state: Dict[str, Any], patterns: Dict[str, Any], goals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer targeted generation around strong, already-good actions (local search).
    Falls back to gentle exploration helper.
    """
    q = state.get("q", {}) or {}
    strong = [(a, v) for a, v in q.items() if not _is_auto(a)]
    strong.sort(key=lambda x: x[1], reverse=True)
    if strong:
        top = [a for a, _ in strong[:3]]
        return {"type": "mutate_top_actions", "actions": top}
    return {"type": "exploration_booster"}

# -----------------------------------------------------------------------------
# Code generation: embed runtime gate inside each new auto_skill
# -----------------------------------------------------------------------------

RUNTIME_GATE_TEMPLATE = (
    "# Auto-generated by self_coder with runtime quality gate\n"
    "from pathlib import Path\n"
    "import json\n"
    "\n"
    "REGISTRY_FILE = Path('data/self_coder_registry.json')\n"
    "ACT_NAME = '{skill_name}'\n"
    "\n"
    "def _load_registry():\n"
    "    try:\n"
    "        return json.loads(REGISTRY_FILE.read_text())\n"
    "    except Exception:\n"
    "        return {{}}\n"
    "\n"
    "def _allowed(my_name, state):\n"
    "    reg = _load_registry()\n"
    "    budget = reg.get('budget', {{}})\n"
    "    if not budget.get('enabled', True):\n"
    "        cand = reg.get('designated_candidate')\n"
    "        sk = reg.get('skills', {{}}).get(my_name, {{}})\n"
    "        if sk.get('status') != 'active' and my_name != cand:\n"
    "            q = state.get('q', {{}})\n"
    "            if ACT_NAME in q:\n"
    "                q[ACT_NAME] = {q_fence}\n"
    "            return False\n"
    "    sk = reg.get('skills', {{}}).get(my_name, {{}})\n"
    "    if sk.get('status') == 'retired':\n"
    "        q = state.get('q', {{}})\n"
    "        if ACT_NAME in q:\n"
    "            q[ACT_NAME] = {q_fence}\n"
    "        return False\n"
    "    return True\n"
    "\n"
    "def act(ctx):\n"
    "    s = ctx['state']\n"
    "    if not _allowed(ACT_NAME, s):\n"
    "        return 0.0\n"
    "\n"
    "{body}\n"
)

def _gen_body_mutate_top(actions: List[str]) -> str:
    lines: List[str] = []
    lines.append("    q = s.get('q', {})")
    lines.append("    if not q:")
    lines.append("        return 0.0")
    lines.append("    boosted = 0")
    for a in actions[:3]:
        lines.append(f"    if '{a}' in q and q['{a}'] > 0.0:")
        lines.append(f"        q['{a}'] = q['{a}'] * 1.02")
        lines.append("        boosted += 1")
    lines.append("    eps = float(s.get('epsilon', 0.2))")
    lines.append("    if eps < 0.10:")
    lines.append("        s['epsilon'] = 0.12")
    lines.append("    if eps > 0.40:")
    lines.append("        s['epsilon'] = 0.35")
    lines.append("    return 0.002 if boosted else 0.0")
    return "\n".join(lines)

def _gen_body_exploration_booster() -> str:
    return "\n".join([
        "    eps = float(s.get('epsilon', 0.2))",
        "    if eps < 0.15:",
        "        s['epsilon'] = min(0.35, eps + 0.05)",
        "        return 0.01",
        "    return 0.0",
    ])

def generate_skill_code(need: Dict[str, Any], skill_number: int) -> Tuple[str, str]:
    ts = int(_now())
    skill_name = f"auto_skill_{ts}_{skill_number}"

    if need.get("type") == "mutate_top_actions":
        body = _gen_body_mutate_top(need.get("actions", []))
    else:
        body = _gen_body_exploration_booster()

    code = RUNTIME_GATE_TEMPLATE.format(
        skill_name=skill_name,
        q_fence=repr(Q_FENCE_DISABLED),
        body=body
    )
    return skill_name, code

# -----------------------------------------------------------------------------
# Generation rate-limit & probation slot control
# -----------------------------------------------------------------------------

def _can_generate_today(gen_log: Dict[str, Any]) -> bool:
    day_ago = _now() - 86400.0
    recent = [g for g in gen_log.get("generated", []) if g.get("timestamp", 0) > day_ago]
    return len(recent) < MAX_SKILLS_PER_DAY

def _probation_count(registry: Dict[str, Any]) -> int:
    return sum(1 for sk in registry.get("skills", {}).values() if sk.get("status", "probation") == "probation")

# -----------------------------------------------------------------------------
# Main action
# -----------------------------------------------------------------------------

def act(ctx: Dict[str, Any]) -> float:
    s = ctx.get("state", {})
    ticks = int(s.get("ticks", 0))

    # Throttle heavy work
    registry = _load_json(REGISTRY_FILE, {})
    last_run_tick = int(registry.get("last_run_tick", -10**9))
    if ticks - last_run_tick < TICK_COOLDOWN:
        return 0.0

    # Refresh metrics & registry
    recent, per_auto = compute_recent_metrics()
    registry = update_registry(recent, per_auto)
    registry["last_run_tick"] = ticks
    _save_json(REGISTRY_FILE, registry)

    # Budget check: if over target and no candidate burst, skip spawn
    budget = registry.get("budget", {})
    over_budget = float(budget.get("auto_skill_pct", 0.0)) > BUDGET_TARGET_PCT and registry.get("designated_candidate") is None

    # Probation and daily limits
    gen_log = _load_json(GENERATED_LOG, {"generated": [], "last_generation": 0})
    if not _can_generate_today(gen_log):
        return 0.0
    if _probation_count(registry) >= MAX_CONCURRENT_PROB:
        return 0.0
    if over_budget:
        return 0.0

    # Identify need and generate a single skill
    patterns = _load_json(PATTERNS_FILE, {})
    goals = _load_json(GOALS_FILE, {})
    need = identify_need(s, patterns, goals)
    if not need:
        return 0.0

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_name, code = generate_skill_code(need, len(gen_log.get("generated", [])) + 1)
    path = SKILLS_DIR / f"skill_{skill_name}.py"

    try:
        path.write_text(code)

        # Log generation
        gen_log.setdefault("generated", []).append({
            "name": skill_name,
            "timestamp": _now(),
            "tick": ticks,
            "need": need,
        })
        gen_log["last_generation"] = _now()
        _save_json(GENERATED_LOG, gen_log)

        # Seed registry entry for probation
        skills = registry.setdefault("skills", {})
        skills[skill_name] = {
            "status": "probation",
            "uses": 0,
            "avg_reward": 0.0,
            "strikes": 0,
            "q_fence": Q_FENCE_DISABLED,
            "last_tick": ticks,
        }
        _save_json(REGISTRY_FILE, registry)

        # Note for visibility
        notes = s.setdefault("notes", [])
        notes.append(f"self_coder: generated {skill_name} ({need.get('type')}) at tick {ticks}")

        # Optional hint to inbox (best-effort)
        try:
            INBOX.parent.mkdir(parents=True, exist_ok=True)
            with INBOX.open("a") as f:
                f.write(f"note SELF_CODER new_skill {skill_name} need={need.get('type')} tick={ticks}\n")
        except Exception:
            pass

        return NEW_SKILL_REWARD

    except Exception as e:
        notes = s.setdefault("notes", [])
        notes.append(f"self_coder: generation failed -> {str(e)[:60]}")
        return -0.01
