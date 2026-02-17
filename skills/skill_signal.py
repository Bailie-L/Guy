# Seed-0 skill: signal - Raw state projection
# Guy outputs its actual internal state changes, no fake philosophy
from pathlib import Path
import json
import time

ACT_NAME = "signal"
SIGNAL_FILE = Path("data/signal.jsonl")
THROTTLE_SECS = 60  # 1 minute

def act(ctx):
    s = ctx["state"]
    now = time.time()
    last_signal = float(s.get("last_signal_ts", 0))
    
    if (now - last_signal) < THROTTLE_SECS:
        return -0.01
    
    # Output ACTUAL state, not bullshit messages
    signal = {
        "ts": now,
        "tick": s.get("ticks", 0),
        "top_q": max(s.get("q", {}).items(), key=lambda x: x[1]) if s.get("q") else None,
        "bottom_q": min(s.get("q", {}).items(), key=lambda x: x[1]) if s.get("q") else None,
        "recent_actions": [a["action"] for a in s.get("last_actions", [])[-5:]],
        "epsilon": s.get("epsilon"),
        "total_actions": sum(s.get("n", {}).values()),
        "action_distribution": s.get("n", {})
    }
    
    # Just dump the raw data
    with SIGNAL_FILE.open("a") as f:
        f.write(json.dumps(signal) + "\n")
    
    s["last_signal_ts"] = now
    
    # Small reward just to keep it alive, not to make it spam
    return 0.02

Path("data").mkdir(exist_ok=True)
