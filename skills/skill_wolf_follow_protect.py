"""
Skill: wolf_follow_protect
Goal : Keep close to owner even when it's unsafe (caves/dark/threats),
       using only natural movement (no teleport). Re-issues follow to re-path,
       injects short patrol "jiggles" when stuck, escalates to guard/attack and
       battle howl if threats are detected.

I/O   : Emits GuyBridge outbox JSON events:
          - {"action":"wolf_actions","wolf_action":"follow","target":"owner"}
          - {"action":"wolf_actions","wolf_action":"patrol","pattern":"arc_left","radius":2,"duration_s":2}
          - {"action":"wolf_actions","wolf_action":"guard"}
          - {"action":"wolf_actions","wolf_action":"attack"}
          - {"action":"wolf_howl","howl":"battle"}
        Optionally writes a tiny reward line to inbox to bias learning.

Context (optional, best-effort):
  ctx = {
    "owner_distance": float blocks,
    "light_level": int 0..15,
    "is_underground": bool,
    "hostiles_nearby": int,
    "owner_under_attack": bool,
    "wolf_health": float 0..1,
    "recent_stuck": bool
  }
If fields are missing, defaults are used.
"""

from __future__ import annotations
import json, time, random
from pathlib import Path
from typing import Dict, Any

NAME = "wolf_follow_protect"

# ---------- Paths ----------
def _root() -> Path:
    here = Path(__file__).resolve()
    for _ in range(6):
        if (here.parent / "data").is_dir():
            return here.parent
        here = here.parent
    return Path.cwd()

def _outbox() -> Path:
    p = _root() / "data" / "outbox"
    p.mkdir(parents=True, exist_ok=True)
    return p

def _inbox() -> Path:
    _root().mkdir(parents=True, exist_ok=True)
    ib = _root() / "data" / "inbox.txt"
    if not ib.exists():
        ib.touch()
    return ib

def _state_path() -> Path:
    return _root() / "data" / "wolf_follow_protect_state.json"

# ---------- State ----------
def _load_state() -> Dict[str, Any]:
    sp = _state_path()
    if sp.exists():
        try:
            return json.loads(sp.read_text())
        except Exception:
            return {}
    return {}

def _save_state(s: Dict[str, Any]):
    _state_path().write_text(json.dumps(s, indent=2))

def _emit(payload: Dict[str, Any]):
    ts = int(time.time())
    fn = _outbox() / f"{('howl' if payload.get('action')=='wolf_howl' else 'action')}-{ts}.json"
    with fn.open("w") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)

# ---------- Policy ----------
def _danger_score(ctx: Dict[str, Any]) -> float:
    hostiles = int(ctx.get("hostiles_nearby", 0))
    under_attack = 1 if ctx.get("owner_under_attack") else 0
    low_light = 1 if int(ctx.get("light_level", 15)) <= 7 else 0
    cave = 1 if ctx.get("is_underground") else 0
    low_health = 1 if float(ctx.get("wolf_health", 1.0)) <= 0.35 else 0
    # Weighted sum
    return 0.45*hostiles + 0.8*under_attack + 0.2*low_light + 0.2*cave + 0.4*low_health

def _should_patrol_jiggle(ctx: Dict[str, Any], state: Dict[str, Any]) -> bool:
    if ctx.get("recent_stuck"):
        return True
    # If distance isn't shrinking (tracked in state), jiggle
    last_d = float(state.get("last_owner_dist2", 1e9))
    d = float(ctx.get("owner_distance", 6.0))
    d2 = d*d
    state["last_owner_dist2"] = d2
    stagnant = d2 >= last_d - 1.0
    # 30% chance to jiggle when stagnant, else no
    return stagnant and random.random() < 0.3

# ---------- Main entry ----------
def act(ctx: Dict[str, Any]) -> float:
    """
    Called by Guy's RL loop. Emits one event (or none) per call.
    Returns a tiny intrinsic reward; external rewards should be appended via inbox by the bridge/game.
    """
    now = time.time()
    s = _load_state()
    last_ts = float(s.get("last_ts", 0))
    last_follow = float(s.get("last_follow_ts", 0))
    # Cooldowns (seconds)
    cd_follow = 2.0           # reissue follow ~every 2s for stickiness
    cd_patrol = 3.0           # small jiggle spacing
    cd_howl = 8.0             # avoid howl spam
    cd_guard_attack = 1.0     # allow quick reaction

    # Defaults if ctx missing
    owner_dist = float(ctx.get("owner_distance", 8.0))
    danger = _danger_score(ctx)

    emitted = False
    reward = 0.0

    # 1) If danger is high, escalate posture first
    if danger >= 1.2 and (now - last_ts) > cd_guard_attack:
        # Battle howl (signal pack) with guard/attack depending on hostiles
        hostiles = int(ctx.get("hostiles_nearby", 0))
        if (now - float(s.get("last_howl_ts", 0))) > cd_howl:
            _emit({"protocol":"guy.v1","action":"wolf_howl","howl":"battle","tick":int(now)})
            s["last_howl_ts"] = now
        action = "attack" if hostiles >= 2 or ctx.get("owner_under_attack") else "guard"
        _emit({"protocol":"guy.v1","action":"wolf_actions","wolf_action":action,"tick":int(now)})
        emitted = True
        reward += 0.03

    # 2) Keep follow sticky (natural re-path) if not close enough
    if (now - last_follow) > cd_follow and owner_dist > 2.2:
        _emit({"protocol":"guy.v1","action":"wolf_actions","wolf_action":"follow","target":"owner","tick":int(now)})
        s["last_follow_ts"] = now
        emitted = True
        # Small intrinsic reward to bias following
        reward += 0.03

    # 3) If progress seems stagnant or recently stuck, insert a tiny patrol jiggle (no teleport)
    if (now - float(s.get("last_patrol_ts", 0))) > cd_patrol and _should_patrol_jiggle(ctx, s):
        _emit({"protocol":"guy.v1","action":"wolf_actions","wolf_action":"patrol","pattern":"arc_left","radius":2,"duration_s":2,"tick":int(now)})
        s["last_patrol_ts"] = now
        emitted = True
        reward += 0.02

    if emitted:
        s["last_ts"] = now
        _save_state(s)
        try:
            _inbox().write_text(f"reward {NAME} +{reward:.2f}\n", append=True)  # type: ignore
        except TypeError:
            # Python <3.11 Path.write_text has no append; fall back:
            with _inbox().open("a") as f:
                f.write(f"reward {NAME} +{reward:.2f}\n")
        return reward

    # Nothing to do this tick
    _save_state(s)
    return 0.0
