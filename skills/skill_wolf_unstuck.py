"""
Skill: wolf_unstuck
Purpose: Reduce pathing stalls with natural movement (no teleport).
Strategy: Occasionally issue a short patrol arc, then re-issue follow.
Outputs: JSON files to data/outbox/ using Guy protocol (wolf_actions).
"""

from __future__ import annotations
import json, time
from pathlib import Path
from typing import Dict, Any

ACT_NAME = "wolf_unstuck"

def _project_root() -> Path:
    # Resolve to guy_test root by walking up from this file
    p = Path(__file__).resolve()
    for _ in range(6):
        if (p.parent / "data").is_dir():
            return p.parent
        p = p.parent
    # Fallback to CWD/data
    return Path.cwd()

def _outbox() -> Path:
    root = _project_root()
    out = root / "data" / "outbox"
    out.mkdir(parents=True, exist_ok=True)
    return out

def _state_file() -> Path:
    return _project_root() / "data" / "wolf_unstuck_state.json"

def _load_state() -> Dict[str, Any]:
    sf = _state_file()
    if sf.exists():
        try:
            return json.loads(sf.read_text())
        except Exception:
            return {}
    return {}

def _save_state(s: Dict[str, Any]) -> None:
    _state_file().write_text(json.dumps(s, indent=2))

def _write_event(payload: Dict[str, Any]) -> None:
    ts = int(time.time())
    fn = _outbox() / f"action-{ts}.json"
    with fn.open("w") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)

def act(ctx: Dict[str, Any]) -> float:
    """
    Emits either:
      1) patrol arc (very short) to jiggle around obstacles; or
      2) follow (to re-acquire owner) after a patrol.
    Cooldown prevents spam; uses only natural navigation.
    Returns a small positive reward for variety; real reward comes from external feedback.
    """
    now = time.time()
    state = _load_state()
    last_ts = float(state.get("last_ts", 0))
    phase = state.get("phase", "follow")  # next action to emit
    cooldown = float(state.get("cooldown_s", 4.0))  # minimum seconds between emits

    if now - last_ts < cooldown:
        return 0.0  # skip this tick (respect cooldown)

    # Prepare parameters (optional hints ignored by bridge if unknown)
    if phase == "patrol":
        # Short arc to slide around corners (natural movement)
        payload = {
            "protocol": "guy.v1",
            "action": "wolf_actions",
            "wolf_action": "patrol",
            "pattern": "arc_left",
            "radius": 2,          # blocks
            "duration_s": 2,      # seconds, tiny burst
            "tick": int(now)
        }
        next_phase = "follow"
        next_cooldown = 2.0
    else:
        # Default: keep following owner (repath)
        payload = {
            "protocol": "guy.v1",
            "action": "wolf_actions",
            "wolf_action": "follow",
            "target": "owner",
            "tick": int(now)
        }
        # Occasionally insert a patrol jiggle next time
        next_phase = "patrol"
        next_cooldown = 3.0

    try:
        _write_event(payload)
        # Update state
        state["last_ts"] = now
        state["phase"] = next_phase
        state["cooldown_s"] = next_cooldown
        _save_state(state)
        # Small intrinsic reward to keep it in the mix
        return 0.02
    except Exception:
        return -0.01
