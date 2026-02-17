# skill_escape_trap: Aggressive escape from local optima traps
# When one action dominates >75% for 500+ ticks, take drastic measures:
# - Reset dominant Q to 0
# - Boost epsilon to 0.5 temporarily  
# - Penalize the dominant action's N count
# - High reward for successful escape

from pathlib import Path
from collections import Counter, deque
import json

ACT_NAME = "escape_trap"

LOG_PATH = Path("data/events.log")
DOMINANCE_THRESHOLD = 0.75  # 75% dominance triggers escape
MIN_TICKS = 500             # Need this many recent ticks to analyze
THROTTLE_TICKS = 200        # Run every 200 ticks
EPSILON_BOOST = 0.5         # Set epsilon to this when escaping
Q_RESET_VALUE = 0.0         # Reset dominant Q to neutral

def _tail_actions(path: Path, max_lines: int):
    evts = deque(maxlen=max_lines)
    if not path.exists(): 
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("kind") == "tick":
                    action = e.get("action")
                    if action:
                        evts.append(action)
            except Exception:
                pass
    return list(evts)

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    
    # Throttle to avoid spam
    if ticks % THROTTLE_TICKS != 0:
        return 0.0
    
    # Analyze recent actions
    recent_actions = _tail_actions(LOG_PATH, MIN_TICKS)
    if len(recent_actions) < MIN_TICKS:
        return 0.0
    
    # Check for dominance
    c = Counter(recent_actions)
    total = sum(c.values())
    if total == 0:
        return 0.0
    
    dominant_action, dominant_count = c.most_common(1)[0]
    dominance_ratio = dominant_count / total
    
    # If trapped, take drastic action
    if dominance_ratio >= DOMINANCE_THRESHOLD:
        # Store escape attempt metadata
        s["last_escape_trap"] = {
            "tick": ticks,
            "dominant_action": dominant_action,
            "dominance_ratio": round(dominance_ratio, 3),
            "total_analyzed": total
        }
        
        # DRASTIC MEASURES:
        # 1. Reset the dominant action's Q-value to neutral
        q = s.setdefault("q", {})
        old_q = q.get(dominant_action, 0.0)
        q[dominant_action] = Q_RESET_VALUE
        
        # 2. Dramatically increase epsilon for exploration
        old_epsilon = s.get("epsilon", 0.2)
        s["epsilon"] = EPSILON_BOOST
        
        # 3. Add escape note
        notes = s.setdefault("notes", [])
        notes.append(f"ESCAPE TRAP at tick {ticks}: {dominant_action} was {int(dominance_ratio*100)}% dominant. Reset Q from {old_q:.4f} to {Q_RESET_VALUE}, epsilon {old_epsilon:.3f}→{EPSILON_BOOST}")
        
        # High reward for taking escape action
        return 0.8
    
    # No trap detected
    return 0.0
