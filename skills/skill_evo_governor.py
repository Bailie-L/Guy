# skills/skill_evo_governor.py
# Purpose: Stable long-run improvement via cautious UCB-driven nudges.
# - Maintains EMA reward and counts per action (file-backed).
# - Selects a "champion" (best EMA) and a "candidate" (best UCB).
# - If candidate's UCB exceeds champion's EMA by a margin, write tiny
#   off-policy rewards to data/inbox.txt to lift candidate and slightly
#   cool champion. This gently increases sampling of promising actions.
# - Monitors global reward; if a recent dip follows a nudge, auto-rollback.
# - Adds a small maintenance rotation that periodically nudges evo_keeper,
#   removing the chicken-and-egg and guaranteeing floors get enforced.
#
# Notes:
# - Rollback now waits for fresh evidence (ticks/time and new samples) and
#   only clears state if its writes actually happen (no phantom rollbacks).
#
# Safety: stdlib-only, writes only data/*.json and data/inbox.txt. No OS/net.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json, math, time

# ---- Paths ----
ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / "data"
INBOX = DATA / "inbox.txt"
STORE = DATA / "evo_governor.json"
CFG   = DATA / "evo_governor_config.json"  # optional

# Expose to Guy
ACT_NAME = "evo_governor"

# ---- Defaults (can be overridden via CFG) ----
DEFAULTS = {
    # EMA smoothing for per-action rewards
    "ema_alpha": 0.10,

    # Which actions to consider
    "allow": [],
    "allow_prefixes": ["wolf_", "mc_", "conversation_ai", "communicate"],
    "exclude_prefixes": [
        "auto_skill_", "cleanup_orphans", "heartbeat",
        "compress_memory", "reflect", "signal", "memory_bank",
        "message_curator", "contextual_bandit", "policy_optimizer",
        "entropy_", "stability_pilot", "loop_breaker",
        "escape_trap", "resource_smart", "survival_reflexes",
        "promotion_gate", "evo_governor"
    ],
    "exclude_exact": [],

    # Margins & nudge size (reward scale ~0.00–0.02 typical)
    "promote_margin": 0.004,
    "rollback_margin": 0.0025,
    "delta_reward": 0.01,

    # UCB exploration constant
    "ucb_c": 0.01,

    # Evidence/cooldowns
    "min_samples_champ": 30,
    "decision_cooldown_ticks": 60,
    "decision_cooldown_s": 30.0,
    "max_writes_per_act": 2,

    # Health guard: rollback on short-term dip after a decision
    "health_short": 25, "health_long": 200, "health_drop": 0.0035,
    "cooldown_after_rollback_ticks": 200,
    "cooldown_after_rollback_s": 120.0,

    # Wait before considering rollback after a promote (evidence gate)
    "rollback_wait_ticks": 10,
    "rollback_wait_s": 30.0,
    "rollback_min_new_samples": 1,  # sum of (Δn_cand + Δn_champ) ≥ this

    # --- Maintenance rotation for evo_keeper ---
    "rotate_every_ticks": 150,
    "rotate_bump_reward": 0.012,
    "rotate_cooldown_s": 30.0
}

# ---- Helpers ----
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

def _append_inbox(lines: List[str]):
    if not lines: return
    INBOX.parent.mkdir(parents=True, exist_ok=True)
    with INBOX.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")

def _now() -> float:
    return time.time()

def _to_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _get_tick(s: Dict[str, Any]) -> int:
    """Robustly derive current tick. Supports 'tick', 'ticks', or last_actions entries."""
    t = s.get("tick", None)
    if isinstance(t, (int, float)): return int(t)
    t = s.get("ticks", None)
    if isinstance(t, (int, float)): return int(t)
    la = s.get("last_actions") or []
    for item in reversed(la):
        tt = item.get("tick")
        if isinstance(tt, (int, float)):
            return int(tt)
    return 0

def _filter_actions(all_actions: List[str], cfg: Dict[str, Any]) -> List[str]:
    allow = cfg.get("allow", []) or []
    if allow:
        return [a for a in all_actions if a in allow]
    allow_pfx = tuple(cfg.get("allow_prefixes", DEFAULTS["allow_prefixes"]))
    excl_pfx  = tuple(cfg.get("exclude_prefixes", DEFAULTS["exclude_prefixes"]))
    excl_exact = set(cfg.get("exclude_exact", []))
    out = []
    for a in all_actions:
        if a in excl_exact: continue
        if a.startswith(excl_pfx): continue
        if a.startswith(allow_pfx):
            out.append(a)
    return out

def _ucb(ema: Optional[float], n: int, total_n: int, c: float) -> float:
    base = 0.0 if ema is None else ema
    bonus = c * math.sqrt(max(0.0, math.log(max(2, total_n)) / max(1, n)))
    return base + bonus

def _mean(xs: List[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0

# ---- Core ----
def act(ctx) -> Optional[float]:
    s: Dict[str, Any] = ctx["state"]
    last: List[Dict[str, Any]] = s.get("last_actions", [])

    # Load config & store
    cfg = {**DEFAULTS, **_load_json(CFG, {})}
    store = _load_json(STORE, {
        "ema": {}, "n": {},
        "last_decision": None,
        "last_decision_tick": 0,
        "last_decision_ts": 0.0,
        "cool_until_tick": 0,
        "cool_until_ts": 0.0,
        "notes": [],
        "maint_last_tick": -10**9,
        "maint_last_ts": 0.0
    })
    ema: Dict[str, Optional[float]] = store.get("ema", {})
    n:   Dict[str, int]             = store.get("n", {})
    last_decision: Optional[Dict[str, Any]] = store.get("last_decision")
    cool_until_tick: int = _to_int(store.get("cool_until_tick", 0))
    cool_until_ts: float = float(store.get("cool_until_ts", 0.0))
    now = _now()

    # Robust tick lookup (supports missing/None)
    ticks = _get_tick(s)
    alpha = float(cfg["ema_alpha"])

    # --- Maintenance rotation: keep evo_keeper alive gently ---
    last_rot_tick = _to_int(store.get("maint_last_tick", -10**9), -10**9)
    last_rot_ts   = float(store.get("maint_last_ts", 0.0))
    ready_by_tick = (ticks > 0) and ((ticks - last_rot_tick) >= int(cfg["rotate_every_ticks"]))
    ready_by_time = (now - last_rot_ts) >= float(cfg["rotate_cooldown_s"])
    if ready_by_time and (ready_by_tick or ticks <= 0):
        bump = float(cfg["rotate_bump_reward"])
        _append_inbox([f"reward evo_keeper {bump:.4f}"])
        store["maint_last_tick"] = ticks
        store["maint_last_ts"]   = now
        store["notes"].append({"t": now, "event": "maint_bump", "tick": ticks, "bump": round(bump, 4)})

    # 1) Update per-action EMA & counts from the rolling window
    for item in last:
        a = item.get("action")
        r = float(item.get("reward", 0.0))
        if not isinstance(a, str):
            continue
        prev = ema.get(a)
        ema[a] = (r if prev is None else (prev + alpha * (r - prev)))
        n[a] = int(n.get(a, 0)) + 1

    store["ema"] = ema; store["n"] = n

    # 2) Health guard: rolling reward dip detection → rollback if needed
    short_w = int(cfg["health_short"]); long_w = int(cfg["health_long"])
    short = [float(x.get("reward", 0.0)) for x in last[-short_w:]]
    long  = [float(x.get("reward", 0.0)) for x in last[-long_w:]]
    short_m, long_m = _mean(short), _mean(long)
    dip = (short_m < long_m - float(cfg["health_drop"]))

    inbox_lines: List[str] = []
    writes = 0

    # 3) Selection (skip if cooling by tick or time)
    if ticks >= cool_until_tick or now >= cool_until_ts:
        actions_all = list(s.get("q", {}).keys())
        actions = _filter_actions(actions_all, cfg)
        if len(actions) >= 2:
            total_n = sum(int(n.get(a, 0)) for a in actions) + 1
            # Champion: highest EMA with enough samples
            champ: Optional[Tuple[str, float, int]] = None
            for a in actions:
                ea = ema.get(a)
                na = int(n.get(a, 0))
                if ea is None or na < int(cfg["min_samples_champ"]):
                    continue
                if (not champ) or (ea > champ[1]):
                    champ = (a, ea, na)
            # Candidate: highest UCB among the rest
            cand: Optional[Tuple[str, float, float, int]] = None  # (name, ema, ucb, n)
            if champ:
                for a in actions:
                    if a == champ[0]:
                        continue
                    ea = ema.get(a)
                    na = int(n.get(a, 0))
                    u  = _ucb(ea, na, total_n, float(cfg["ucb_c"]))
                    if (not cand) or (u > cand[2]):
                        cand = (a, (0.0 if ea is None else ea), u, na)

            # Decision cooldown: ticks or seconds fallback
            ticks_since_last = ticks - _to_int(store.get("last_decision_tick", 0))
            secs_since_last  = now - float(store.get("last_decision_ts", 0.0))
            tick_ready = ticks_since_last >= int(cfg["decision_cooldown_ticks"])
            time_ready = secs_since_last >= float(cfg["decision_cooldown_s"])

            # 4) Promotion decision using UCB gap
            if champ and cand:
                champ_name, champ_ema, champ_n = champ
                cand_name, cand_ema, cand_ucb, cand_n = cand
                gap = cand_ucb - champ_ema
                if (gap >= float(cfg["promote_margin"])
                    and (not store.get("last_decision"))
                    and (tick_ready or (ticks <= 0 and time_ready))):
                    delta = float(cfg["delta_reward"])
                    inbox_lines.append(f"reward {cand_name} {+delta:.4f}")
                    inbox_lines.append(f"reward {champ_name} {-delta:.4f}")
                    writes += 2
                    store["last_decision"] = {
                        "tick": ticks, "ts": now,
                        "champion": champ_name, "candidate": cand_name,
                        "delta": delta, "champ_ema": round(champ_ema, 4),
                        "cand_ucb": round(cand_ucb, 4), "cand_ema": round(cand_ema, 4),
                        "cand_n_at_decision": cand_n, "champ_n_at_decision": champ_n
                    }
                    store["last_decision_tick"] = ticks
                    store["last_decision_ts"]   = now
                    store["notes"].append({"t": now, "event": "promote",
                                           "champion": champ_name, "candidate": cand_name,
                                           "gap": round(gap, 4)})

            # 5) If already promoted and candidate underperforms by margin → rollback
            if store.get("last_decision") and not dip and (ticks >= cool_until_tick or now >= cool_until_ts):
                # Evidence gate: must have time/ticks elapsed AND some new samples
                ld = store["last_decision"]
                last_tick_at_decision = _to_int(store.get("last_decision_tick", -10**9), -10**9)
                ticks_since = ticks - last_tick_at_decision
                secs_since  = now - float(store.get("last_decision_ts", 0.0))
                waited_enough = (ticks_since >= int(cfg["rollback_wait_ticks"])) or (ticks <= 0 and secs_since >= float(cfg["rollback_wait_s"]))
                cand_name  = ld.get("candidate"); champ_name = ld.get("champion")
                ce = ema.get(cand_name); he = ema.get(champ_name)
                new_samples = max(0, int(n.get(cand_name,0)) - int(ld.get("cand_n_at_decision",0))) + \
                              max(0, int(n.get(champ_name,0)) - int(ld.get("champ_n_at_decision",0)))
                if waited_enough and new_samples >= int(cfg["rollback_min_new_samples"]) and (ce is not None) and (he is not None) and (ce + float(cfg["rollback_margin"]) < he):
                    delta = float(ld.get("delta", cfg["delta_reward"]))
                    # Respect per-act write budget; only proceed if we can write both lines
                    writes_remaining = max(0, int(cfg["max_writes_per_act"]) - writes)
                    if writes_remaining >= 2:
                        inbox_lines.append(f"reward {cand_name} {-delta:.4f}")
                        inbox_lines.append(f"reward {champ_name} {+delta:.4f}")
                        writes += 2
                        store["notes"].append({"t": now, "event": "rollback_gap",
                                               "champion": champ_name, "candidate": cand_name,
                                               "cand_ema": round(ce, 4), "champ_ema": round(he, 4)})
                        store["last_decision"] = None
                        store["cool_until_tick"] = ticks + int(cfg["cooldown_after_rollback_ticks"])
                        store["cool_until_ts"]   = now + float(cfg["cooldown_after_rollback_s"])
                    else:
                        store["notes"].append({"t": now, "event": "rollback_deferred_no_budget",
                                               "champion": champ_name, "candidate": cand_name})

    # Safety: cap write count (does not affect maintenance bump which writes immediately)
    if writes > int(cfg["max_writes_per_act"]):
        inbox_lines = inbox_lines[:int(cfg["max_writes_per_act"])]

    # Emit inbox lines and persist store
    if inbox_lines:
        _append_inbox(inbox_lines)

    _save_json_atomic(STORE, {
        "ema": ema,
        "n": n,
        "last_decision": store.get("last_decision"),
        "last_decision_tick": store.get("last_decision_tick", 0),
        "last_decision_ts": store.get("last_decision_ts", 0.0),
        "cool_until_tick": store.get("cool_until_tick", 0),
        "cool_until_ts": store.get("cool_until_ts", 0.0),
        "maint_last_tick": store.get("maint_last_tick", -10**9),
        "maint_last_ts": store.get("maint_last_ts", 0.0),
        "notes": store.get("notes", [])[-200:]
    })

    # Breadcrumb in state notes (first few EMAs for visibility)
    preview = {k: (None if v is None else round(v, 4)) for k, v in list(ema.items())[:4]}
    s.setdefault("notes", []).append(
        f"evo_governor: writes={len(inbox_lines)} short={short_m:.4f} long={long_m:.4f} ema_head={preview}"
    )

    # Tiny neutral-positive reward so the bandit doesn't downrank maintenance
    return 0.005
