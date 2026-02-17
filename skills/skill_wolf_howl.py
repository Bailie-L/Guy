"""
Wolf Howl Skill for Guy (AAA-grade)
- Chooses howl type (lonely / battle / play) from current state.
- Emits structured JSON to outbox for observability.
- Writes a bridge file for server-side listeners (pack spawner).
- Appends a reward hint into Guy's inbox for reinforcement learning.
- Cooldown + anti-spam with tiny persisted state.

State inputs expected in ctx["state"] (all optional, with safe defaults):
    owner_absent_secs: int         # seconds since owner seen (default 0)
    nearby_hostiles: int           # count of hostile mobs nearby (default 0)
    wolf_health_pct: float         # 0..100 (default 100)
    owner_under_attack: bool       # default False
    time_of_day: str               # 'day','night','dusk','dawn','evening','morning' (default 'day')
    mood: str                      # 'happy','sad','lonely','brave','scared' (default 'neutral')
    nearby_wolves: int             # pack proximity (default 0)
    safe: bool                     # area safety for play howls (default True)
    tick: int                      # optional simulation tick

Priority: BATTLE > LONELY > PLAY (only one howl per invocation).
"""

from __future__ import annotations

from pathlib import Path
import json, time, random
from typing import Dict, Any, Tuple

ACT_NAME = "wolf_howl"

# Resolve project paths robustly (…/guy_test/)
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUTBOX = DATA / "outbox"
TMP = DATA / "tmp"
INBOX = DATA / "inbox.txt"

OUTBOX.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

STATE_FILE = TMP / "wolf_howl_state.json"

# Cooldowns per howl type (seconds)
COOLDOWN = {
    "battle": 60,   # allow quicker re-calls in danger
    "lonely": 180,  # limit to keep it special
    "play": 120,    # occasional fun
}

def _now() -> float:
    return time.time()

def _read_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text() or "{}")
        except Exception:
            return {}
    return {}

def _write_state(d: Dict[str, Any]) -> None:
    try:
        STATE_FILE.write_text(json.dumps(d, indent=2))
    except Exception:
        pass

def _cooldown_ok(howl_type: str, now_ts: float, st: Dict[str, Any]) -> Tuple[bool, float]:
    last_ts = st.get("last_ts", 0)
    last_type = st.get("last_type")
    cd = COOLDOWN.get(howl_type, 90)
    # If repeating same type, enforce cooldown fully; if different type, allow 50% cooldown
    required = cd if last_type == howl_type else cd * 0.5
    remaining = max(0.0, required - (now_ts - last_ts))
    return (remaining <= 0.0), remaining

def _choose_howl(state: Dict[str, Any]) -> Tuple[str|None, Dict[str, Any]]:
    reasons = {}
    owner_absent = int(state.get("owner_absent_secs", 0))
    nearby_hostiles = int(state.get("nearby_hostiles", 0))
    wolf_hp = float(state.get("wolf_health_pct", 100.0))
    owner_under_attack = bool(state.get("owner_under_attack", False))
    tod = str(state.get("time_of_day", "day")).lower()
    mood = str(state.get("mood", "neutral")).lower()
    nearby_wolves = int(state.get("nearby_wolves", 0))
    safe = bool(state.get("safe", True))

    # --- Battle Howl (highest priority)
    battle = (nearby_hostiles >= 3) or (wolf_hp < 50.0) or owner_under_attack
    if battle:
        reasons["battle"] = {
            "nearby_hostiles": nearby_hostiles,
            "wolf_health_pct": wolf_hp,
            "owner_under_attack": owner_under_attack
        }
        return "battle", reasons

    # --- Lonely Howl
    nightish = tod in {"night", "dusk", "evening"}
    mood_bonus = 0.15 if mood in {"sad", "lonely"} else 0.05
    lonely_prob = 0.10 + mood_bonus  # ~10–25% chance when conditions met
    lonely_conditions = (owner_absent >= 600) and (nearby_wolves == 0) and nightish
    if lonely_conditions and random.random() < lonely_prob:
        reasons["lonely"] = {
            "owner_absent_secs": owner_absent,
            "nearby_wolves": nearby_wolves,
            "time_of_day": tod,
            "mood": mood,
            "roll_under": lonely_prob
        }
        return "lonely", reasons

    # --- Play Howl (rarest)
    happy = (mood == "happy")
    daylight = tod in {"day", "dawn", "morning"}
    play_prob = 0.07  # rare by design
    if happy and daylight and safe and random.random() < play_prob:
        reasons["play"] = {
            "mood": mood,
            "time_of_day": tod,
            "safe": safe,
            "roll_under": play_prob
        }
        return "play", reasons

    return None, reasons

def act(ctx: Dict[str, Any]) -> float:
    state = ctx.get("state", {})
    tick = state.get("tick")

    howl_type, rule_reasons = _choose_howl(state)
    ts = _now()

    # If no howl selected, emit a light heartbeat log for observability and exit.
    if not howl_type:
        msg = {
            "ts": ts,
            "tick": tick,
            "action": ACT_NAME,
            "howl": None,
            "reason": rule_reasons,
            "note": "no_howl",
        }
        (OUTBOX / f"howl-{int(ts)}.json").write_text(json.dumps(msg, indent=2))
        return 0.005  # tiny baseline so the scheduler doesn't starve this skill

    # Cooldown check
    st = _read_state()
    ok, remaining = _cooldown_ok(howl_type, ts, st)
    if not ok:
        msg = {
            "ts": ts,
            "tick": tick,
            "action": ACT_NAME,
            "howl": howl_type,
            "reason": rule_reasons,
            "cooldown_applied": True,
            "cooldown_remaining_s": round(remaining, 2),
        }
        (OUTBOX / f"howl-{int(ts)}.json").write_text(json.dumps(msg, indent=2))
        return 0.003  # slight penalty to discourage spamming paths

    # We will howl 🎵
    msg = {
        "ts": ts,
        "tick": tick,
        "action": ACT_NAME,
        "howl": howl_type,  # 'battle' | 'lonely' | 'play'
        "reason": rule_reasons,
        "cooldown_applied": False,
    }

    # Write structured outbox event
    (OUTBOX / f"howl-{int(ts)}.json").write_text(json.dumps(msg, indent=2))

    # Bridge file for the server plugin (read & react)
    (OUTBOX / "wolf_howl.txt").write_text(howl_type)

    # Reward hint for RL loop (slightly higher than wolf_actions baseline)
    try:
        with INBOX.open("a") as f:
            f.write(f"reward ghost_howl_{howl_type} +0.02\n")
    except Exception:
        # Best-effort; don't crash the skill
        pass

    # Persist state for cooldown tracking
    st = {"last_ts": ts, "last_type": howl_type}
    _write_state(st)

    # Positive baseline reward to encourage evaluation
    return 0.02
