# Emergency skill: heartbeat_dampener - reduces heartbeat reward when overused
from pathlib import Path
from collections import deque
import json

ACT_NAME = "heartbeat_dampener"
LOG_PATH = Path("data/events.log")
RUN_EVERY = 5  # Check frequently

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    
    if ticks % RUN_EVERY != 0:
        return 0.0
    
    # Count recent heartbeats
    recent = deque(maxlen=100)
    if LOG_PATH.exists():
        with LOG_PATH.open("r") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("kind") == "tick":
                        recent.append(e.get("action"))
                except:
                    pass
    
    recent_list = list(recent)[-50:]  # Last 50 actions
    heartbeat_count = recent_list.count("heartbeat")
    
    if heartbeat_count > 25:  # More than 50% heartbeat
        # ACTIVELY PUNISH heartbeat in Q-table
        q = s.setdefault("q", {})
        if "heartbeat" in q:
            # Decay heartbeat Q-value
            q["heartbeat"] = q["heartbeat"] * 0.8
            
        # Boost all other actions
        for action in q:
            if action != "heartbeat":
                q[action] = min(q[action] + 0.01, 0.1)
        
        return 0.1  # Reward for dampening
    
    return 0.0
