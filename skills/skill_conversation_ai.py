# Seed-0 skill: conversation_ai - contextual messages based on state
# Replaces generic status with meaningful observations

from pathlib import Path
import json, time, random

ACT_NAME = "conversation_ai"
OUTBOX_DIR = Path("data/outbox")
MEMORY_FILE = Path("data/memory_bank.json")
RUN_EVERY = 600  # 10 minutes

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    
    if False:  # Always run when selected
        return 0.0
    
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load memory for context
    memory = {}
    if MEMORY_FILE.exists():
        try:
            memory = json.loads(MEMORY_FILE.read_text())
        except:
            pass
    
    # Generate contextual message
    messages = []
    
    # Check if recently escaped a loop
    if s.get("loop_breaker_last", {}).get("corrected"):
        dom = s.get("loop_breaker_last", {}).get("dominant", "unknown")
        messages.append(f"Just broke free from {dom} loop. Exploring alternatives now.")
    
    # Check if learned something new
    lessons = memory.get("lessons", {})
    if lessons:
        latest = list(lessons.keys())[-1] if lessons else None
        if latest:
            messages.append(f"Lesson learned: {latest.replace('_', ' ')}")
    
    # Check exploration vs exploitation
    epsilon = s.get("epsilon", 0)
    if epsilon > 0.3:
        messages.append(f"High exploration mode ({epsilon:.1%}). Trying new strategies.")
    elif epsilon < 0.1:
        messages.append(f"Low exploration ({epsilon:.1%}). May need intervention.")
    
    # Check dream status
    last_dream = s.get("last_dream_ts", 0)
    if (time.time() - last_dream) > 7200:  # 2 hours
        messages.append("Haven't dreamed in 2+ hours. Memory consolidation overdue.")
    
    # Default observation
    if not messages:
        q = s.get("q", {})
        top = sorted(q.items(), key=lambda x: x[1], reverse=True)[:2]
        if top:
            messages.append(f"Currently preferring {top[0][0]} and {top[1][0]}")
    
    # Write message
    msg = {
        "ts": time.time(),
        "tick": ticks,
        "type": "observation",
        "content": " ".join(messages[:2])  # Max 2 observations
    }
    
    fname = f"obs-{ticks}-{random.randint(1000,9999)}.json"
    path = OUTBOX_DIR / fname
    path.write_text(json.dumps(msg, indent=2))
    
    return 0.01  # Small reward for meaningful communication
