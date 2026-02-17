# skills/skill_comm_rewarder.py
# Purpose: Auto-reward substantive outbox messages and gently penalize bland check-ins,
# by appending reward commands to data/inbox.txt for the core loop to apply.

from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from typing import List, Set

ACT_NAME = "comm_rewarder"

# Tunables (conservative)
MAX_FILES_TO_SCAN = 12
SEEN_DB = Path("data/.skills/comm_rewarder_seen.txt")
INBOX = Path("data/inbox.txt")
OUTBOX_DIR = Path("data/outbox")
SEEN_CAP = 1000  # keep last N fingerprints

# Reward policy
R_POS_MAINT  = +0.60   # maintenance/verify/bundle/self_verify/self_bundle
R_POS_ALERT  = +0.40   # alerts/errors worth attention
R_POS_DREAM  = +0.30   # dream reflections
R_POS_POLICY = +0.25   # policy/optimizer/stability/entropy
R_POS_FALLBK = +0.20   # long, informative freeform
R_NEG_CHECK  = -0.10   # low-signal check-ins
R_NEG_STUCK  = -0.05   # stuck spam when exploration already adequate
EPS_STUCK_POS = 0.20   # if epsilon < this, treat "stuck" as asking for explore (+0.20)

# Heuristics
CHECKIN_TITLE_RE = re.compile(r"^\s*check[- ]?in\s*$", re.IGNORECASE)
TICK_PREFS_RE    = re.compile(r"^tick\s+\d+\.\s*prefs", re.IGNORECASE)

def _load_seen() -> Set[str]:
    if SEEN_DB.exists():
        return set(x for x in SEEN_DB.read_text(encoding="utf-8").splitlines() if x.strip())
    return set()

def _save_seen(seen: Set[str]) -> None:
    SEEN_DB.parent.mkdir(parents=True, exist_ok=True)
    # cap size deterministically
    if len(seen) > SEEN_CAP:
        keep = sorted(seen)
        seen = set(keep[-SEEN_CAP:])
    SEEN_DB.write_text("\n".join(sorted(seen)), encoding="utf-8")

def _fingerprint(path: Path, j: dict) -> str:
    h = hashlib.sha1()
    h.update(path.name.encode("utf-8", errors="ignore"))
    for k in ("title","text"):
        h.update((j.get(k,"") or "").encode("utf-8", errors="ignore"))
    h.update(",".join([t.lower() for t in (j.get("tags") or [])]).encode("utf-8", errors="ignore"))
    return h.hexdigest()

def _classify(j: dict) -> str:
    title = (j.get("title") or "").strip()
    tags  = [t.lower() for t in (j.get("tags") or [])]
    text  = (j.get("text") or "").strip()

    # Bland check-ins
    if "note" in tags or CHECKIN_TITLE_RE.match(title) or TICK_PREFS_RE.match(text):
        return "checkin"

    # Maintenance-ish
    if any(t in tags for t in ("maintenance","verify","bundle","self_verify","self_bundle")):
        return "maintenance"
    if any(w in title.lower() for w in ("maintenance","verify","bundle")):
        return "maintenance"

    # Alerts/errors
    if any(t in tags for t in ("alert","error","failure")) or any(w in title.lower() for w in ("alert","error","failure")):
        return "alert"

    # Dream reflections
    if "dream" in tags or "dream" in title.lower():
        return "dream"

    # Policy/meta signals
    if any(t in tags for t in ("policy","optimizer","stability","entropy")):
        return "policy"

    # Stuck notices
    if "stuck" in tags or "stuck" in title.lower():
        return "stuck"

    # Long informative freeform
    if len(text) >= 140 and len(text.split()) >= 20:
        return "substantive"

    return "other"

def _suggest_reward(kind: str, epsilon: float) -> float | None:
    if kind == "maintenance": return R_POS_MAINT
    if kind == "alert":       return R_POS_ALERT
    if kind == "dream":       return R_POS_DREAM
    if kind == "policy":      return R_POS_POLICY
    if kind == "substantive": return R_POS_FALLBK
    if kind == "checkin":     return R_NEG_CHECK
    if kind == "stuck":       return (+0.20 if epsilon < EPS_STUCK_POS else R_NEG_STUCK)
    return None  # neutral

def _append_inbox(lines: List[str]) -> None:
    if not lines:
        return
    INBOX.parent.mkdir(parents=True, exist_ok=True)
    with INBOX.open("a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")

def act(ctx) -> float | None:
    """
    Guy skill contract: expose ACT_NAME and act(ctx).
    We scan recent outbox messages, score them, and write reward commands like:
      reward communicate +0.40
    Return a tiny positive reward only if we actually applied feedback.
    """
    state = ctx.get("state", {}) if isinstance(ctx, dict) else {}
    try:
        epsilon = float(state.get("epsilon", 0.0))
    except Exception:
        epsilon = 0.0

    if not OUTBOX_DIR.exists():
        return 0.0

    files = sorted(OUTBOX_DIR.glob("msg-*.json"), key=lambda p: p.stat().st_mtime)[-MAX_FILES_TO_SCAN:]
    if not files:
        return 0.0

    seen = _load_seen()
    to_write: List[str] = []
    changed = False

    for p in reversed(files):  # newest first
        try:
            raw = p.read_text(encoding="utf-8")
            j = json.loads(raw)
        except Exception:
            continue

        fid = _fingerprint(p, j)
        if fid in seen:
            continue

        kind = _classify(j)
        reward = _suggest_reward(kind, epsilon)

        if reward is not None:
            to_write.append(f"reward communicate {reward:+.2f}")
            changed = True

        seen.add(fid)

    if changed and to_write:
        _append_inbox(to_write)

    try:
        _save_seen(seen)
    except Exception:
        pass

    return 0.02 if changed else 0.0
