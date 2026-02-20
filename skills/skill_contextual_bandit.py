# skills/skill_contextual_bandit.py
# Level 1 -> Level 2: Contextual Bandit as a drop-in skill.
# Reads recent context (events/outbox/state), then writes small reward nudges
# to data/inbox.txt so the core bandit favors context-appropriate actions next tick.
# Safe, conservative, cooldown-protected.

from __future__ import annotations
import json, re, time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

ACT_NAME = "contextual_bandit"

# ---------- Files ----------
EVENTS_LOG = Path("data/events.log")
OUTBOX_DIR = Path("data/outbox")
STATE_JSON = Path("data/state.json")
INBOX      = Path("data/inbox.txt")
STATE_FILE = Path("data/.skills/contextual_bandit_state.json")

# ---------- Parsing / limits ----------
TAIL_BYTES_EVENTS   = 120_000   # read tail of events.log
MAX_OUTBOX_TO_SCAN  = 12        # newest N outbox messages
COOLDOWN_SECONDS    = 120       # do not act more often than this
MIN_WRITE_ABS       = 0.01      # ignore micro-nudges

# ---------- Policy knobs (conservative) ----------
STREAK_LIMIT        = 12        # if same action repeats >= this, penalize it a bit
DIVERSITY_MIN       = 0.15      # unique_actions / recent_actions
HEARTBEAT_MAX_SHARE = 0.60      # if heartbeat dominates above this, tap brakes
DREAM_TICKS_LIMIT   = 800       # if no 'dream' seen in ~this many recent ticks, nudge dream

NUDGE_STREAK_NEG    = -0.04     # penalty for the repeated action when streak too long
NUDGE_COMM_POS      = +0.12     # encourage communicate
NUDGE_DREAM_POS     = +0.15     # encourage dream
NUDGE_POLICY_POS    = +0.06     # encourage policy_optimizer
NUDGE_VERIFY_POS    = +0.08     # encourage self_verify
NUDGE_BUNDLE_POS    = +0.05     # encourage self_bundle
NUDGE_CURIOUS_POS   = +0.15     # encourage curiosity_engine
NUDGE_ESCAPE_POS    = +0.08     # encourage escape_trap
NUDGE_HEARTBEAT_NEG = -0.02     # gentle baseline dampener if heartbeat too dominant
NUDGE_ALERT_VERIFY  = +0.10     # if alerts observed in outbox, verify more
NUDGE_STABILITY_POS = +0.08     # nudge stability_pilot if present

# Hard caps for any single write
NUDGE_MIN, NUDGE_MAX = -0.15, +0.25

ACTION_RE = re.compile(r"tick\s+\d+:\s*([A-Za-z0-9_]+)")

# Outbox classification heuristics (minimal, reuse of comm_rewarder logic style)
CHECKIN_TITLE_RE = re.compile(r"^\s*check[- ]?in\s*$", re.IGNORECASE)
TICK_PREFS_RE    = re.compile(r"^tick\s+\d+\.\s*prefs", re.IGNORECASE)

def _tail_text(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return ""
    start = max(0, size - max_bytes)
    with path.open("rb") as f:
        f.seek(start)
        data = f.read()
    text = data.decode("utf-8", errors="ignore")
    if start > 0:
        parts = text.splitlines()
        return "\n".join(parts[1:]) if parts else ""
    return text

def _parse_recent_actions(text: str) -> List[str]:
    return ACTION_RE.findall(text)

def _streak_info(actions: List[str]) -> Tuple[str, int]:
    if not actions:
        return ("", 0)
    last = actions[-1]
    i = len(actions) - 1
    c = 0
    while i >= 0 and actions[i] == last:
        c += 1
        i -= 1
    return (last, c)

def _heartbeat_share(actions: List[str]) -> float:
    if not actions:
        return 0.0
    hb = sum(1 for a in actions if a == "heartbeat")
    return hb / max(1, len(actions))

def _since_last(actions: List[str], name: str) -> Optional[int]:
    # returns number of ticks since last occurrence within the actions window (approx by index)
    if not actions:
        return None
    for i in range(len(actions)-1, -1, -1):
        if actions[i] == name:
            return len(actions) - 1 - i
    return None

def _load_state_json() -> Dict:
    try:
        return json.loads(STATE_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _load_runtime_state() -> Dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _save_runtime_state(d: Dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def _classify_outbox(j: Dict) -> str:
    title = (j.get("title") or "").strip()
    tags  = [t.lower() for t in (j.get("tags") or [])]
    text  = (j.get("text") or "").strip()

    if "note" in tags or CHECKIN_TITLE_RE.match(title) or TICK_PREFS_RE.match(text):
        return "checkin"
    if any(t in tags for t in ("maintenance","verify","bundle","self_verify","self_bundle")):
        return "maintenance"
    if any(t in tags for t in ("alert","error","failure")):
        return "alert"
    if "dream" in tags:
        return "dream"
    if any(t in tags for t in ("policy","optimizer","stability","entropy")):
        return "policy"
    if len(text) >= 140 and len(text.split()) >= 20:
        return "substantive"
    return "other"

def _scan_outbox() -> Dict[str, int]:
    counts = {"substantive":0, "checkin":0, "alert":0, "maintenance":0, "dream":0, "policy":0, "other":0}
    files = sorted(OUTBOX_DIR.glob("msg-*.json"), key=lambda p: p.stat().st_mtime)[-MAX_OUTBOX_TO_SCAN:]
    for p in files:
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            k = _classify_outbox(j)
            counts[k] = counts.get(k, 0) + 1
        except Exception:
            pass
    return counts

def _append_inbox(lines: List[str]) -> None:
    if not lines:
        return
    INBOX.parent.mkdir(parents=True, exist_ok=True)
    with INBOX.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def _clip(x: float) -> float:
    return max(NUDGE_MIN, min(NUDGE_MAX, x))

def act(ctx) -> Optional[float]:
    """
    Look at recent context and write reward nudges to data/inbox.txt:
      - Break long streaks (esp. heartbeat)
      - Increase action diversity
      - Raise 'substantive comms' ratio
      - Ensure periodic dreaming/planning
      - React to 'alert' signals
    Returns a tiny positive intrinsic reward if nudges were written.
    """
    now = time.time()
    rstate = _load_runtime_state()
    last_ts = float(rstate.get("last_apply_ts", 0.0))
    if now - last_ts < COOLDOWN_SECONDS:
        return 0.0

    text = _tail_text(EVENTS_LOG, TAIL_BYTES_EVENTS)
    actions = _parse_recent_actions(text)
    if not actions:
        return 0.0

    # Context features
    last_action, streak = _streak_info(actions)
    uniq = len(set(actions))
    diversity = uniq / max(1, len(actions))
    hb_share = _heartbeat_share(actions)
    since_dream = _since_last(actions, "dream")
    sjson = _load_state_json()
    epsilon = float(sjson.get("epsilon", 0.0)) if isinstance(sjson, dict) else 0.0

    outbox_counts = _scan_outbox()
    substantive = outbox_counts.get("substantive", 0) + outbox_counts.get("maintenance", 0) + outbox_counts.get("alert", 0) + outbox_counts.get("dream", 0) + outbox_counts.get("policy", 0)
    total_outbox = substantive + outbox_counts.get("checkin", 0) + outbox_counts.get("other", 0)
    substantive_ratio = (substantive / max(1, total_outbox)) if total_outbox else 0.0
    saw_alert = outbox_counts.get("alert", 0) > 0

    nudges: Dict[str, float] = {}

    # 1) Break long streaks
    if streak >= STREAK_LIMIT and last_action:
        nudges[last_action] = nudges.get(last_action, 0.0) + NUDGE_STREAK_NEG
        nudges["communicate"] = nudges.get("communicate", 0.0) + NUDGE_COMM_POS
        nudges["dream"]       = nudges.get("dream", 0.0) + NUDGE_DREAM_POS

    # 2) Cap heartbeat dominance
    if hb_share > HEARTBEAT_MAX_SHARE:
        nudges["heartbeat"] = nudges.get("heartbeat", 0.0) + NUDGE_HEARTBEAT_NEG
        nudges["communicate"] = nudges.get("communicate", 0.0) + (NUDGE_COMM_POS * 0.5)

    # 3) Improve diversity
    if diversity < DIVERSITY_MIN:
        nudges["curiosity_engine"] = nudges.get("curiosity_engine", 0.0) + NUDGE_CURIOUS_POS
        nudges["escape_trap"]      = nudges.get("escape_trap", 0.0) + NUDGE_ESCAPE_POS

    # 4) Ensure periodic dreaming
    if since_dream is None or since_dream >= DREAM_TICKS_LIMIT:
        nudges["dream"] = nudges.get("dream", 0.0) + NUDGE_DREAM_POS

    # 5) Substantive comms low? Encourage comms/policy/verify/bundle
    if substantive_ratio < 0.50:
        nudges["communicate"]     = nudges.get("communicate", 0.0) + NUDGE_COMM_POS
        nudges["policy_optimizer"]= nudges.get("policy_optimizer", 0.0) + NUDGE_POLICY_POS
        nudges["self_verify"]     = nudges.get("self_verify", 0.0) + NUDGE_VERIFY_POS
        nudges["self_bundle"]     = nudges.get("self_bundle", 0.0) + NUDGE_BUNDLE_POS

    # 6) Alerts observed? Harden stability & verification
    if saw_alert:
        nudges["self_verify"]     = nudges.get("self_verify", 0.0) + NUDGE_ALERT_VERIFY
        nudges["stability_pilot"] = nudges.get("stability_pilot", 0.0) + NUDGE_STABILITY_POS

    # 7) Exploration safety: if epsilon is very low and diversity also low, lean harder on curiosity
    if epsilon < 0.20 and diversity < DIVERSITY_MIN:
        nudges["curiosity_engine"] = nudges.get("curiosity_engine", 0.0) + (NUDGE_CURIOUS_POS * 0.5)

    # Build inbox writes
    cmds: List[str] = []
    for action, val in nudges.items():
        val = _clip(val)
        if abs(val) >= MIN_WRITE_ABS:
            cmds.append(f"reward {action} {val:+.2f}")

    if not cmds:
        return 0.0

    _append_inbox(cmds)

    # Persist runtime stamp + a few diagnostics
    rstate["last_apply_ts"] = now
    rstate["diag"] = {
        "last_action": last_action,
        "streak": streak,
        "diversity": round(diversity, 3),
        "hb_share": round(hb_share, 3),
        "since_dream": since_dream,
        "substantive_ratio": round(substantive_ratio, 3),
        "epsilon": epsilon,
        "nudges": {k: round(_clip(v), 3) for k, v in nudges.items()}
    }
    _save_runtime_state(rstate)

    return 0.04
