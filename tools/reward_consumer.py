from pathlib import Path
import json, glob, os, time
from collections import Counter

OUTBOX = Path("data/outbox")
INBOX  = Path("data/inbox.txt")
STATEF = Path("data/.reward_consumer_state.json")

def _load_state():
    try:
        return json.loads(STATEF.read_text(encoding="utf-8"))
    except Exception:
        return {"last_mtime": 0.0, "last_run": 0.0}

def _save_state(s):
    STATEF.parent.mkdir(parents=True, exist_ok=True)
    STATEF.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

def _flatten(d, p=""):
    if isinstance(d, dict):
        for k,v in d.items():
            yield from _flatten(v, f"{p}.{k}" if p else k)
    elif isinstance(d, list):
        for i,v in enumerate(d):
            yield from _flatten(v, f"{p}[{i}]")
    else:
        yield p, d

def _collect_labels(path):
    labels = Counter()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except Exception:
        return labels
    for k,v in _flatten(data):
        leaf = k.rsplit(".",1)[-1]
        if leaf in {"action","subtype","wolf_action","command","cmd","name","type"}:
            if isinstance(v, (str,int,float,bool)):
                labels[str(v)] += 1
    return labels

def main():
    s = _load_state()
    OUTBOX.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(OUTBOX/"*.json")), key=os.path.getmtime, reverse=True)

    watermark = float(s.get("last_mtime", 0.0))
    fresh = [f for f in files if os.path.getmtime(f) > watermark]
    had_fresh = bool(fresh)
    work = fresh[:2000]  # only reward on fresh items; no fallback spamming

    agg = Counter()
    newest_mtime = watermark
    for f in work:
        m = os.path.getmtime(f)
        if m > newest_mtime:
            newest_mtime = m
        agg.update(_collect_labels(f))

    pos_keys = {"guard","follow","patrol","attack"}
    neg_keys = {"howl"}  # keep 'idle' neutral for now
    pos = sum(agg[k] for k in pos_keys)
    neg = sum(agg[k] for k in neg_keys)

    lines = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    top6 = dict(agg.most_common(6))
    lines.append(f"note consumer: {ts} scanned={len(work)} fresh={had_fresh} counts={{pos:{pos}, neg:{neg}}} top={top6}")

    rewards = []
    if had_fresh and pos > 0:
        rewards.append(("wolf_actions", +min(0.05, 0.005 * pos)))  # softer shaping
    if had_fresh and neg > 0:
        rewards.append(("wolf_howl", -min(0.05, 0.005 * neg)))     # softer penalty

    if rewards:
        INBOX.parent.mkdir(parents=True, exist_ok=True)
        with open(INBOX, "a", encoding="utf-8") as f:
            for act, val in rewards:
                sign = "+" if val >= 0 else ""
                f.write(f"reward {act} {sign}{val:.3f}\n")
        lines += [f"reward {a} {('+' if v>=0 else '')}{v:.3f}" for a,v in rewards]

    s["last_run"] = time.time()
    if had_fresh:
        s["last_mtime"] = float(newest_mtime)
    _save_state(s)

    print("== reward_consumer ==")
    print(f"scanned: {len(work)} files  pos:{pos} neg:{neg}  fresh:{had_fresh}")
    for L in lines:
        print("  " + L)

if __name__ == "__main__":
    main()
