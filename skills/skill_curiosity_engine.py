# Seed-0 skill: curiosity_engine - intrinsic motivation through novelty bonus
# Breaks loops by rewarding CHANGE itself, not just outcomes
#
# Mechanism:
# - Tracks action frequency over multiple time windows (short/medium/long)
# - Calculates "novelty score" for each action based on inverse frequency
# - Provides bonus reward for choosing rare/underused actions
# - Creates "boredom penalty" for repetitive patterns
# - Maintains curiosity memory separate from Q-values

from pathlib import Path
from collections import deque, Counter
import json, math, time

ACT_NAME = "curiosity_engine"
LOG_PATH = Path("data/events.log")
RUN_EVERY = 15  # frequent evaluation for continuous influence

# Time windows for novelty calculation
SHORT_WINDOW = 60    # last minute
MEDIUM_WINDOW = 300  # last 5 minutes  
LONG_WINDOW = 900    # last 15 minutes

# Curiosity parameters
NOVELTY_BONUS = 0.08      # reward for trying rare actions
BOREDOM_PENALTY = -0.03   # penalty for excessive repetition
CURIOSITY_DECAY = 0.95    # how fast curiosity memory fades
MIN_CURIOSITY = 0.01      # floor for curiosity scores

def _get_action_history(path: Path, window: int):
    """Get recent action history"""
    actions = deque(maxlen=window)
    if not path.exists():
        return []
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("kind") == "tick":
                    action = e.get("action")
                    if action:
                        actions.append(action)
            except:
                pass
    return list(actions)

def _calculate_novelty(action: str, histories: dict) -> float:
    """Calculate novelty score for an action across time windows"""
    novelty = 0.0
    weights = {"short": 0.5, "medium": 0.3, "long": 0.2}
    
    for window, history in histories.items():
        if not history:
            continue
        counter = Counter(history)
        total = sum(counter.values())
        frequency = counter.get(action, 0) / total if total > 0 else 0
        
        # Inverse frequency = higher score for rare actions
        window_novelty = 1.0 - frequency
        novelty += window_novelty * weights.get(window, 0)
    
    return novelty

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    
    # Only run periodically
    if ticks % RUN_EVERY != 0:
        return 0.0
    
    # Get action histories
    histories = {
        "short": _get_action_history(LOG_PATH, SHORT_WINDOW),
        "medium": _get_action_history(LOG_PATH, MEDIUM_WINDOW),
        "long": _get_action_history(LOG_PATH, LONG_WINDOW)
    }
    
    # Initialize curiosity memory if needed
    curiosity = s.setdefault("curiosity_memory", {})
    
    # Decay existing curiosity scores
    for action in list(curiosity.keys()):
        curiosity[action] = max(MIN_CURIOSITY, curiosity[action] * CURIOSITY_DECAY)
    
    # Calculate novelty and update Q-values with curiosity bonus
    q = s.setdefault("q", {})
    last_action = None
    
    # Get the most recent action
    if histories["short"]:
        last_action = histories["short"][-1]
    
    if last_action:
        novelty = _calculate_novelty(last_action, histories)
        
        # Update curiosity memory
        curiosity[last_action] = novelty
        
        # Detect repetitive patterns (same action 5+ times in last 10 ticks)
        recent_10 = histories["short"][-10:] if len(histories["short"]) >= 10 else []
        if recent_10.count(last_action) >= 7:
            # Boredom penalty - make repetitive action less attractive
            if last_action in q:
                q[last_action] = q[last_action] * 0.98
            return BOREDOM_PENALTY
        
        # Reward novelty
        if novelty > 0.7:  # High novelty threshold
            # Boost Q-value of novel actions slightly
            for action in q:
                if action != "heartbeat":  # Don't boost the dominant one
                    action_novelty = _calculate_novelty(action, histories)
                    if action_novelty > 0.6:
                        q[action] = min(q[action] + 0.002, 0.1)
            
            return NOVELTY_BONUS
    
    # Store curiosity state
    s["curiosity_engine_last"] = {
        "tick": ticks,
        "curiosity_scores": {k: round(v, 3) for k, v in list(curiosity.items())[:5]},
        "last_action": last_action
    }
    
    return 0.0
