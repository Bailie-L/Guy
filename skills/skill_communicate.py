# communicate: human messages (alerts/maintenance/dream/stuck/note) to data/outbox/
from pathlib import Path
from collections import deque, Counter, defaultdict
import json, time, os, hashlib

ACT_NAME = "communicate"

OUTBOX_DIR = Path("data/outbox")
FORCE_FLAG = Path("data/force/communicate.force")
LOG_PATH   = Path("data/events.log")
THROTTLE_TICKS = 300         # ~5 min
WINDOW_TICKS   = 500
BACKLOG_MAX    = 100
DUP_PENALTY    = -0.05
COOLDOWN_SECS  = {"stuck":900, "note":600, "maintenance":0, "dream":0, "alert":60}

def _consume_force_flag() -> bool:
    if FORCE_FLAG.exists():
        try:
            FORCE_FLAG.unlink(missing_ok=True)
        except Exception:
            pass
        return True
    return False

def _tail_events(path: Path, want_ticks: int):
    evts = deque(maxlen=want_ticks*2)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                evts.append(json.loads(line))
            except Exception:
                pass
    return [e for e in evts if e.get("kind") in ("tick","skill_error")]

def _sha256_text(txt: str) -> str:
    h = hashlib.sha256()
    h.update(txt.encode("utf-8"))
    return h.hexdigest()

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    forced = _consume_force_flag()
    if not forced and (ticks % THROTTLE_TICKS != 0):
        return 0.0

    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    if len(list(OUTBOX_DIR.glob("msg-*.json"))) > BACKLOG_MAX:
        return -0.05

    events = _tail_events(LOG_PATH, WINDOW_TICKS)
    last_tick = next((e for e in reversed(events) if e.get("kind")=="tick"), None)
    last_action = last_tick.get("action") if last_tick else None
    last_reward = last_tick.get("reward") if last_tick else None

    # gather errors
    err = defaultdict(int)
    for e in events:
        if e.get("kind")=="skill_error":
            a = e.get("action")
            if a:
                err[a]+=1

    # dominance check
    recent_ticks = [e for e in events if e.get("kind")=="tick"]
    actions = [e.get("action") for e in recent_ticks if e.get("action")]
    c = Counter(actions)
    total = sum(c.values()) or 1
    dom = None
    if total >= 50:
        a0, c0 = c.most_common(1)[0]
        if c0/total >= 0.80:
            dom = (a0, c0/total, total)

    # dream recent?
    dream_recent = False
    dream_snip = None
    dreams = s.get("dreams") or []
    if dreams:
        d = dreams[-1]
        if time.time() - float(d.get("ts", 0.0)) < 600:
            dream_recent = True
            summ = d.get("summary") or {}
            top_avg = summ.get("top_avg_reward") or []
            dream_snip = f"Dream adjusted epsilon to {summ.get('epsilon_after','?')}."
            if top_avg:
                dream_snip += " Best avg: " + ", ".join(f"{a}:{v}" for a,v in top_avg)

    # maintenance note?
    maint_msg = None
    if last_action in ("self_verify","self_bundle") and (last_reward or 0) > 0:
        if last_action=="self_verify":
            p = s.get("last_verify_bundle")
            maint_msg = f"Verified latest bundle OK: {p}" if p else "Verification succeeded."
        else:
            p = s.get("last_bundle")
            maint_msg = f"Created new bundle: {p}" if p else "Bundle created."

    # choose message
    title = ""
    text = ""
    tags = []
    if err:
        title = "Alert: recent errors"
        pairs = ", ".join(f"{k}:{v}" for k,v in sorted(err.items(), key=lambda kv:-kv[1])[:4])
        text  = f"I saw errors → {pairs}. Last action {last_action} (reward {last_reward})."
        tags  = ["alert","errors"]
    elif maint_msg:
        title = "Maintenance OK"
        text  = maint_msg
        tags  = ["maintenance","ok", last_action]
    elif dream_recent and dream_snip:
        title = "Dream reflection"
        text  = dream_snip
        tags  = ["dream","reflection"]
    elif dom:
        a0, frac, tot = dom
        title = "Stuck pattern detected"
        text  = f"{a0} dominated {int(frac*100)}% of last {tot} ticks. Considering more variety."
        tags  = ["stuck","policy"]
    else:
        eps = s.get("epsilon")
        topq = sorted(((k,float(v)) for k,v in (s.get("q") or {}).items()), key=lambda kv: kv[1], reverse=True)[:3]
        tops = ", ".join(f"{k}:{round(v,4)}" for k,v in topq) if topq else "none"
        title = "Check-in"
        text  = f"Tick {ticks}. Prefs → {tops}. Epsilon {round(float(eps or 0),4)}."
        tags  = ["note"]

    # per-tag cooldown
    last_ct = s.setdefault("comm_cooldown", {})
    now = time.time()
    if tags:
        tag = tags[0]
        cd = COOLDOWN_SECS.get(tag, 0)
        if cd > 0 and (now - float(last_ct.get(tag, 0))) < cd:
            return 0.0
        last_ct[tag] = now

    # de-dup on message text
    msg_hash = _sha256_text(text)
    if s.get("last_comm_hash") == msg_hash:
        return DUP_PENALTY

    payload = {
        "ts": time.time(),
        "tick": ticks,
        "forced": forced,
        "type": "message",
        "title": title,
        "text": text,
        "tags": tags
    }

    fname = f"msg-{int(payload['ts'])}-{ticks}-{os.urandom(3).hex()}.json"
    tmp = OUTBOX_DIR / (fname + ".tmp")
    final = OUTBOX_DIR / fname
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp.write_bytes(raw)
    tmp.replace(final)

    s["last_comm_hash"] = msg_hash
    s["comm_count"] = int(s.get("comm_count", 0)) + 1
    s["last_comm_path"] = final.as_posix()
    return 0.0
