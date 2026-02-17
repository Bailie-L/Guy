#!/usr/bin/env python3
# Prints what Guy has learned so far: ticks, epsilon/alpha, top Q-values (+ counts),
# recent action histogram & avg rewards, next due windows, dream timing, and dominance warning.
import json, time, pathlib, math, datetime
from collections import Counter, defaultdict, deque

STATE = pathlib.Path("data/state.json")
LOG   = pathlib.Path("data/events.log")
WINDOW = 1000  # recent ticks to analyze

def load_state():
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def tail_ticks(path: pathlib.Path, max_items: int):
    out = deque(maxlen=max_items)
    if not path.exists(): return list(out)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("kind") == "tick":
                    out.append(e)
            except Exception:
                pass
    return list(out)

def fmt_hms(seconds:int):
    h = seconds//3600; m=(seconds%3600)//60; s=seconds%60
    return f"{h}h {m}m {s}s"

def main():
    s = load_state()
    ticks = int(s.get("ticks", 0))
    eps   = float(s.get("epsilon", 0.0))
    alpha = float(s.get("alpha", 0.0))
    q     = {k: float(v) for k,v in s.get("q", {}).items()}
    n     = {k: int(v) for k,v in s.get("n", {}).items()}

    # Recent behavior window
    recent = tail_ticks(LOG, WINDOW)
    cnt = Counter(ev.get("action") for ev in recent if ev.get("action"))
    rew = defaultdict(float)
    for ev in recent:
        a = ev.get("action")
        if a:
            try: rew[a] += float(ev.get("reward", 0.0))
            except: pass
    avg = {a: (rew[a]/cnt[a]) for a in cnt if cnt[a]}

    # Sorts
    top_q = sorted(q.items(), key=lambda kv: kv[1], reverse=True)
    top_hist = cnt.most_common()

    # Next due windows
    def next_due(mod):
        return ((-ticks) % mod) if mod else 0
    next_bundle  = next_due(120)
    next_verify  = next_due(180)

    # Dream timing
    last_dream_ts = float(s.get("last_dream_ts", 0.0))
    now = time.time()
    dream_rem = max(0, int(86400 - (now - last_dream_ts)))
    # Pretty time in Africa/Johannesburg if possible
    last_dream_str = "never"
    if last_dream_ts > 0:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("Africa/Johannesburg")
            last_dream_str = datetime.datetime.fromtimestamp(last_dream_ts, tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        except Exception:
            last_dream_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_dream_ts))

    # Dominance warning
    dominance = None
    total = sum(cnt.values())
    if total >= 50 and cnt:
        a0, c0 = top_hist[0]
        frac = c0/total
        if frac >= 0.80:
            dominance = (a0, frac, total)

    # Header
    print("=== Guy: Learned Status ===")
    print(f"ticks={ticks}  epsilon={eps:.4f}  alpha={alpha:.4f}")
    print(f"last_dream: {last_dream_str}  next_dream_in: {fmt_hms(dream_rem)}")
    print(f"next_due: self_bundle in {next_bundle} ticks, self_verify in {next_verify} ticks")

    # Top Q table
    print("\n-- Top Q-values (preference) --")
    if not top_q:
        print("(no Q-values yet)")
    else:
        for k,v in top_q[:10]:
            c = n.get(k, 0)
            print(f"{k:18s} Q={v:8.4f}   n={c}")

    # Recent histogram + averages
    print("\n-- Recent actions (last {:d} ticks) --".format(total))
    if total == 0:
        print("(no recent tick data)")
    else:
        for a,c in top_hist:
            av = avg.get(a, 0.0)
            print(f"{a:18s} {c:5d}  ({c/total:6.2%})   avg_reward={av:6.3f}")

    if dominance:
        a0, frac, tot = dominance
        print(f"\nWARNING: dominance → {a0} at {frac*100:.2f}% of last {tot} ticks")

    # Last communication (if any)
    last_comm = s.get("last_comm_path")
    if last_comm:
        print(f"\nlast_comm_path: {last_comm}")

    # Maintenance refs
    lb = s.get("last_bundle")
    lv = s.get("last_verify_bundle")
    if lb or lv:
        print("\n-- Maintenance --")
        if lb: print(f"last_bundle:       {lb}")
        if lv: print(f"last_verify_bundle:{lv}")

if __name__ == "__main__":
    main()
