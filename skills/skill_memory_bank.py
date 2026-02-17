# Seed-0 skill: memory_bank - persistent memory with dream integration
# Stores important events, patterns, and lessons in structured memory
# During dream phase: consolidates, prunes, and extracts patterns

from pathlib import Path
from collections import deque, defaultdict, Counter
import json, time, hashlib

ACT_NAME = "memory_bank"
MEMORY_FILE = Path("data/memory_bank.json")
LOG_PATH = Path("data/events.log")
RUN_EVERY = 300  # 5 minutes normally
MEMORY_LIMIT = 1000  # max memories to keep
PATTERN_THRESHOLD = 5  # min occurrences to be a pattern

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    
    # Check if in dream state (dream happened recently)
    last_dream = float(s.get("last_dream_ts", 0))
    in_dream = (time.time() - last_dream) < 60  # within 1 minute of dream
    
    # Run more frequently during dream for consolidation
    if not in_dream and ticks % RUN_EVERY != 0:
        return 0.0
    
    # Load or initialize memory
    memory = {}
    if MEMORY_FILE.exists():
        try:
            memory = json.loads(MEMORY_FILE.read_text())
        except:
            memory = {"events": [], "patterns": {}, "lessons": {}}
    else:
        memory = {"events": [], "patterns": {}, "lessons": {}}
    
    # During dream: consolidate and extract patterns
    if in_dream:
        # Extract patterns from recent Q-value changes
        q = s.get("q", {})
        patterns = memory.get("patterns", {})
        
        # Record successful action sequences
        last_actions = s.get("last_actions", [])
        if len(last_actions) >= 3:
            seq = "-".join([a["action"] for a in last_actions[-3:]])
            avg_reward = sum(a.get("reward", 0) for a in last_actions[-3:]) / 3
            if avg_reward > 0.01:  # positive sequence
                patterns[seq] = patterns.get(seq, 0) + 1
        
        # Store lesson: what breaks heartbeat loops
        if "heartbeat" in q and q["heartbeat"] < 0.02:
            memory["lessons"]["heartbeat_suppressed"] = {
                "tick": ticks,
                "method": "collective_skills",
                "epsilon": s.get("epsilon", 0)
            }
        
        memory["patterns"] = patterns
        memory["last_consolidation"] = ticks
        
    else:
        # Normal operation: record significant events
        events = memory.get("events", [])
        
        # Record if we just escaped a loop
        if s.get("loop_breaker_last", {}).get("corrected"):
            events.append({
                "tick": ticks,
                "type": "loop_escaped",
                "dominant": s.get("loop_breaker_last", {}).get("dominant")
            })
        
        # Record if policy was optimized
        if s.get("policy_optimizer_last", {}).get("corrected"):
            events.append({
                "tick": ticks,
                "type": "policy_adjusted",
                "entropy": s.get("policy_optimizer_last", {}).get("entropy", 0)
            })
        
        # Prune old events
        if len(events) > MEMORY_LIMIT:
            events = events[-MEMORY_LIMIT:]
        
        memory["events"] = events
    
    # Save memory atomically
    tmp = MEMORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(memory, indent=2))
    tmp.replace(MEMORY_FILE)
    
    # Store summary in state for other skills to access
    s["memory_bank_summary"] = {
        "total_events": len(memory.get("events", [])),
        "patterns_found": len(memory.get("patterns", {})),
        "lessons_learned": len(memory.get("lessons", {})),
        "last_update": ticks
    }
    
    return 0.01 if in_dream else 0.0  # small reward during consolidation

