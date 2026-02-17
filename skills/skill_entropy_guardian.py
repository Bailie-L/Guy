#!/usr/bin/env python3
# Guy Skill: Entropy Guardian
# Purpose: Monitor Shannon entropy of recent actions and gently rebalance
# when diversity collapses for an extended period.
#
# Contract: stdlib only, exposes ACT_NAME and act(ctx) -> float
# Side-effects: writes hints to data/inbox.txt and a tiny state file.

from __future__ import annotations
import json, math, pathlib
from collections import Counter, deque
from typing import Dict, List, Tuple, Any, Optional

ACT_NAME = "entropy_guardian"

DATA_DIR = pathlib.Path("data")
INBOX    = DATA_DIR / "inbox.txt"
STATEF   = DATA_DIR / "entropy_guardian.json"
EVENTS   = DATA_DIR / "events.log"

# ---- Tunables (conservative) ----
ENTROPY_WINDOW            = 300    # last N actions to measure diversity
# Natural-log entropy (nats), aligned with check_guy.sh
ENTROPY_THRESHOLD         = 1.20   # ~1.73 bits; below this = low diversity
DOMINANCE_THRESHOLD_PCT   = 55.0   # only intervene if top action >= 55%
CRITICAL_STREAK           = 3      # consecutive low-entropy readings before intervention
REBALANCE_COOLDOWN_TICKS  = 500    # min ticks between rebalances
REBALANCE_DURATION_TICKS  = 50     # bookkeeping window (no epsilon edits)
# Nudge strengths (small, reversible)
DOMINANT_PENALTY          = -0.02
UNDERUSED_BONUS           = +0.01
UNDERUSED_PICK            = 5

# ---- Safe helpers ----
def _ctx_state(ctx: Any) -> Dict[str, Any]:
    """Return a dict-like state regardless of ctx shape (dict vs object with .state)."""
    try:
        if isinstance(ctx, dict):
            return ctx.get("state", ctx)
        st = getattr(ctx, "state", None)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}

def _set_note(ctx: Any, msg: str) -> None:
    """Attach a note for logging without assuming ctx shape."""
    try:
        if isinstance(ctx, dict):
            ctx["note"] = msg
        else:
            setattr(ctx, "note", msg)
    except Exception:
        pass  # non-fatal

def _load_persist() -> Dict[str, Any]:
    if STATEF.exists():
        try:
            return json.loads(STATEF.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_persist(d: Dict[str, Any]) -> None:
    try:
        STATEF.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # non-fatal

def _append_inbox(lines: List[str]) -> None:
    """Append safe control hints to inbox."""
    if not lines:
        return
    try:
        INBOX.parent.mkdir(parents=True, exist_ok=True)
        with INBOX.open("a", encoding="utf-8") as f:
            for line in lines:
                if line.strip():
                    f.write(line.rstrip() + "\n")
    except Exception:
        pass  # non-fatal

def _coerce_action_name(x: Any) -> Optional[str]:
    """Normalize various shapes into a clean action name or None."""
    try:
        if isinstance(x, str):
            s = x.strip()
            return s or None
        if isinstance(x, dict):
            # common shapes: {"action":"wolf_actions"} or event dicts
            for key in ("action", "name", "act"):
                v = x.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None
        if isinstance(x, (list, tuple)) and x:
            # if first element is a string, use it
            v = x[0]
            if isinstance(v, str) and v.strip():
                return v.strip()
        # fallback to string form only if short and simple
        s = str(x)
        return s if s and len(s) <= 64 and "{" not in s and "[" not in s else None
    except Exception:
        return None

def _recent_actions(limit: int, state: Dict[str, Any]) -> List[str]:
    """Prefer in-memory last_actions if provided; fall back to events.log deque."""
    # 1) Try last_actions from state
    try:
        la = state.get("last_actions", [])
        names = []
        for item in la:
            n = _coerce_action_name(item)
            if n:
                names.append(n)
        if names:
            return names[-limit:]
    except Exception:
        pass

    # 2) Fallback: parse events.log (stream action strings)
    dq: deque[str] = deque(maxlen=limit)
    if EVENTS.exists():
        try:
            with EVENTS.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        n = _coerce_action_name(e.get("action"))
                        if n:
                            dq.append(n)
                    except Exception:
                        continue
        except Exception:
            pass
    return list(dq)

def _shannon_entropy_nats(actions: List[str]) -> float:
    """Shannon entropy using natural log; ignores any non-string junk defensively."""
    clean = [a for a in actions if isinstance(a, str) and a]
    if not clean:
        return 0.0
    counts = Counter(clean)
    total = float(sum(counts.values()))
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            ent -= p * math.log(p)  # nats
    return ent

def _dominant(actions: List[str]) -> Tuple[str, float]:
    clean = [a for a in actions if isinstance(a, str) and a]
    if not clean:
        return ("none", 0.0)
    counts = Counter(clean)
    act, c = max(counts.items(), key=lambda kv: kv[1])
    pct = (100.0 * c / len(clean)) if clean else 0.0
    return (act, pct)

def _identify_underused(q: Dict[str, float], dominant: str) -> List[str]:
    """Pick a handful of non-dominant actions to gently encourage (by current Q)."""
    try:
        items = sorted(q.items(), key=lambda kv: kv[1], reverse=True)
    except Exception:
        items = []
    out: List[str] = []
    for k, _v in items:
        if not isinstance(k, str) or not k:
            continue
        if k == dominant:
            continue
        out.append(k)
        if len(out) >= UNDERUSED_PICK:
            break
    return out

# ---- Skill entrypoint ----
def act(ctx) -> float:
    """
    Returns a small reward for healthy diversity,
    and writes gentle reward hints to data/inbox.txt when needed.
    """
    st       = _ctx_state(ctx)
    tick     = int(st.get("ticks", 0))
    epsilon  = float(st.get("epsilon", 0.35))
    q_table  = dict(st.get("q", {}))  # action -> Q

    # Persisted control window / streaks
    persist = _load_persist()
    boosted_until = int(persist.get("boost_expires", -1))
    last_rebal    = int(persist.get("last_rebalance_tick", -REBALANCE_COOLDOWN_TICKS))
    low_streak    = int(persist.get("low_entropy_streak", 0))

    # Auto-expire prior boost window (we don't alter epsilon; bookkeeping only)
    if boosted_until >= 0 and tick >= boosted_until:
        persist["boost_expires"] = -1
        _save_persist(persist)

    # Measure current diversity
    recent = _recent_actions(ENTROPY_WINDOW, st)
    ent = _shannon_entropy_nats(recent)
    dom_act, dom_pct = _dominant(recent)

    # Update low-entropy streak (only counts if a single action is truly dominant)
    if ent < ENTROPY_THRESHOLD and dom_pct >= DOMINANCE_THRESHOLD_PCT:
        low_streak += 1
    else:
        low_streak = 0

    # Decide whether to rebalance
    should_rebalance = (
        ent < ENTROPY_THRESHOLD and
        dom_pct >= DOMINANCE_THRESHOLD_PCT and
        low_streak >= CRITICAL_STREAK and
        (tick - last_rebal) >= REBALANCE_COOLDOWN_TICKS
    )

    if should_rebalance:
        # Choose targets (no epsilon auto-tuning; reward shaping only)
        underused = _identify_underused(q_table, dom_act)
        inbox: List[str] = []
        if dom_act != "none":
            inbox.append(f"reward {dom_act} {DOMINANT_PENALTY:+.3f}")
        for a in underused:
            inbox.append(f"reward {a} {UNDERUSED_BONUS:+.3f}")
        _append_inbox(inbox)

        # Persist control window & cooldown
        persist.update({
            "boost_expires": tick + REBALANCE_DURATION_TICKS,
            "last_rebalance_tick": tick,
            "low_entropy_streak": 0
        })
        _save_persist(persist)

        _set_note(ctx, f"🔄 entropy_guardian: rebalance @tick {tick} | H={ent:.3f} dom={dom_act}({dom_pct:.1f}%) eps={epsilon:.3f}")
        return 0.02  # successful supervision

    # No intervention: persist streak & telemetry
    persist.update({
        "low_entropy_streak": low_streak,
        "last_rebalance_tick": last_rebal
    })
    _save_persist(persist)

    # Monitoring rewards: small positive if healthy; tiny negative if trending low
    _set_note(ctx, f"entropy_guardian: H={ent:.3f} dom={dom_act}({dom_pct:.1f}%) streak={low_streak}")
    if ent >= (ENTROPY_THRESHOLD + 0.3):
        return 0.005    # excellent diversity
    elif ent >= ENTROPY_THRESHOLD:
        return 0.002    # healthy
    else:
        return -0.001   # warning trend
