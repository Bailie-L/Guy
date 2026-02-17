# Seed-0 skill: pattern_detector - Recognizes recurring patterns in rewards and time
# Predicts future outcomes based on observed patterns

from pathlib import Path
from collections import defaultdict, deque
import json, time, math

ACT_NAME = "pattern_detector"
PATTERNS_FILE = Path("data/patterns.json")
LOG_PATH = Path("data/events.log")
MEMORY_FILE = Path("data/memory_bank.json")
WINDOW_SIZE = 2000  # events to analyze
MIN_OCCURRENCES = 3  # minimum pattern instances to consider valid

def get_hour_of_day():
    """Get current hour in 24h format"""
    return time.localtime().tm_hour

def get_day_of_week():
    """Get day of week (0=Monday, 6=Sunday)"""
    return time.localtime().tm_wday

def analyze_temporal_patterns(events):
    """Find time-based patterns in event outcomes"""
    patterns = {
        "hourly": defaultdict(lambda: {"success": 0, "fail": 0}),
        "daily": defaultdict(lambda: {"success": 0, "fail": 0}),
        "action_sequences": defaultdict(list),
        "reward_after": defaultdict(list)
    }
    
    # Analyze each event
    for i, event in enumerate(events):
        if event.get("kind") != "tick":
            continue
            
        action = event.get("action")
        reward = event.get("reward", 0)
        ts = event.get("ts", 0)
        
        if not action:
            continue
            
        # Time-based patterns
        hour = time.localtime(ts).tm_hour
        day = time.localtime(ts).tm_wday
        
        if reward > 0:
            patterns["hourly"][f"{action}_{hour}"]["success"] += 1
            patterns["daily"][f"{action}_{day}"]["success"] += 1
        elif reward < 0:
            patterns["hourly"][f"{action}_{hour}"]["fail"] += 1
            patterns["daily"][f"{action}_{day}"]["fail"] += 1
        
        # Sequential patterns (what follows what)
        if i > 0:
            prev_event = events[i-1]
            if prev_event.get("kind") == "tick":
                prev_action = prev_event.get("action")
                if prev_action:
                    seq_key = f"{prev_action}->{action}"
                    patterns["action_sequences"][seq_key].append(reward)
        
        # Reward patterns (what gets rewarded after what)
        if reward > 0 and i > 0:
            prev_event = events[i-1]
            if prev_event.get("kind") == "tick":
                prev_action = prev_event.get("action")
                if prev_action:
                    patterns["reward_after"][prev_action].append((action, reward))
    
    return patterns

def calculate_pattern_strength(pattern_data):
    """Calculate confidence score for a pattern"""
    if isinstance(pattern_data, dict):
        success = pattern_data.get("success", 0)
        fail = pattern_data.get("fail", 0)
        total = success + fail
        if total < MIN_OCCURRENCES:
            return 0
        return (success / total) if total > 0 else 0
    elif isinstance(pattern_data, list):
        if len(pattern_data) < MIN_OCCURRENCES:
            return 0
        positive = sum(1 for x in pattern_data if (x[1] if isinstance(x, tuple) else x) > 0)
        return positive / len(pattern_data) if pattern_data else 0
    return 0

def predict_next_action(patterns, state):
    """Predict best action based on patterns"""
    predictions = {}
    current_hour = get_hour_of_day()
    current_day = get_day_of_week()
    last_action = state.get("last_actions", [{}])[-1].get("action") if state.get("last_actions") else None
    
    # Check hourly patterns
    for key, data in patterns.get("hourly", {}).items():
        if f"_{current_hour}" in key:
            action = key.split("_")[0]
            strength = calculate_pattern_strength(data)
            if strength > 0.6:  # 60% success rate
                predictions[action] = predictions.get(action, 0) + strength
    
    # Check sequential patterns
    if last_action:
        for seq_key, rewards in patterns.get("action_sequences", {}).items():
            if seq_key.startswith(f"{last_action}->"):
                next_action = seq_key.split("->")[1]
                if rewards and len(rewards) >= MIN_OCCURRENCES:
                    avg_reward = sum(rewards) / len(rewards)
                    if avg_reward > 0:
                        predictions[next_action] = predictions.get(next_action, 0) + avg_reward
    
    # Check reward patterns
    if last_action in patterns.get("reward_after", {}):
        reward_data = patterns["reward_after"][last_action]
        if len(reward_data) >= MIN_OCCURRENCES:
            # Find most common rewarded action after this one
            action_counts = defaultdict(float)
            for action, reward in reward_data:
                action_counts[action] += reward
            if action_counts:
                best_action = max(action_counts, key=action_counts.get)
                predictions[best_action] = predictions.get(best_action, 0) + 0.5
    
    return predictions

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    
    # Load recent events
    events = deque(maxlen=WINDOW_SIZE)
    if LOG_PATH.exists():
        with LOG_PATH.open("r") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except:
                    pass
    
    events_list = list(events)
    
    # Analyze patterns
    patterns = analyze_temporal_patterns(events_list)
    
    # Make predictions
    predictions = predict_next_action(patterns, s)
    
    # Find strong patterns to report
    strong_patterns = []
    
    # Check for time-based patterns
    for key, data in patterns.get("hourly", {}).items():
        strength = calculate_pattern_strength(data)
        if strength > 0.7 and data.get("success", 0) + data.get("fail", 0) >= MIN_OCCURRENCES:
            action, hour = key.rsplit("_", 1)
            if strength > 0.8:
                strong_patterns.append(f"{action} works best at {hour}:00")
            elif strength < 0.3:
                strong_patterns.append(f"{action} fails often at {hour}:00")
    
    # Check for sequence patterns
    for seq_key, rewards in patterns.get("action_sequences", {}).items():
        if len(rewards) >= MIN_OCCURRENCES:
            avg = sum(rewards) / len(rewards)
            if abs(avg) > 0.02:
                prev, next = seq_key.split("->")
                if avg > 0:
                    strong_patterns.append(f"{next} after {prev} → reward {avg:.3f}")
                else:
                    strong_patterns.append(f"avoid {next} after {prev}")
    
    # Save patterns
    pattern_data = {
        "timestamp": time.time(),
        "tick": ticks,
        "patterns_found": len(strong_patterns),
        "predictions": predictions,
        "strong_patterns": strong_patterns[:10],  # Top 10
        "statistics": {
            "hourly_patterns": len(patterns.get("hourly", {})),
            "sequence_patterns": len(patterns.get("action_sequences", {})),
            "reward_patterns": len(patterns.get("reward_after", {}))
        }
    }
    
    PATTERNS_FILE.write_text(json.dumps(pattern_data, indent=2))
    
    # Update state with pattern insights
    s["pattern_insights"] = {
        "patterns_found": len(strong_patterns),
        "best_predicted": max(predictions, key=predictions.get) if predictions else None,
        "prediction_confidence": max(predictions.values()) if predictions else 0
    }
    
    # Nudge Q-values based on strong predictions
    if predictions:
        q = s.get("q", {})
        best_action = max(predictions, key=predictions.get)
        confidence = predictions[best_action]
        
        # Slightly boost predicted good actions
        if confidence > 0.5 and best_action in q:
            q[best_action] = q[best_action] * 1.05  # 5% boost
    
    # Update memory if strong patterns found
    if strong_patterns and MEMORY_FILE.exists():
        try:
            memory = json.loads(MEMORY_FILE.read_text())
            memory["patterns"][f"tick_{ticks}"] = strong_patterns[:3]
            MEMORY_FILE.write_text(json.dumps(memory, indent=2))
        except:
            pass
    
    # Small reward for finding patterns
    return 0.01 if strong_patterns else 0.0
