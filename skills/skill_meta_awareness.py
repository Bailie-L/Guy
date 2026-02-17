# Meta-awareness skill: prevents duplicate fixes, monitors dream health, cleans broken logic
from pathlib import Path
import json
import hashlib

ACT_NAME = "meta_awareness"
SKILLS_DIR = Path("skills")
DREAM_THRESHOLD = 1200  # Dream if none in last 1200 ticks
RUN_EVERY = 100  # Check every 100 ticks

def _get_file_hash(content):
    """Quick hash to detect duplicate skill logic"""
    # Strip comments and whitespace for comparison
    clean = "\n".join(line.strip() for line in content.split("\n") 
                      if line.strip() and not line.strip().startswith("#"))
    return hashlib.md5(clean.encode()).hexdigest()[:8]

def act(ctx):
    s = ctx["state"]
    ticks = s.get("ticks", 0)
    
    # Only run periodically
    if ticks % RUN_EVERY != 0:
        return 0.0
    
    reward = 0.0
    notes = s.setdefault("notes", [])
    
    # 1. Check for duplicate auto-generated skills
    auto_skills = list(SKILLS_DIR.glob("skill_auto_*.py"))
    if len(auto_skills) > 3:
        hashes = {}
        dupes = []
        for skill_file in auto_skills:
            try:
                content = skill_file.read_text()
                h = _get_file_hash(content)
                if h in hashes:
                    dupes.append(skill_file.name)
                else:
                    hashes[h] = skill_file.name
            except:
                pass
        
        if dupes:
            notes.append(f"Meta: Found {len(dupes)} duplicate skills at tick {ticks}")
            # Penalize duplicate generation
            s["duplicate_skills_detected"] = len(dupes)
            reward -= 0.01
    
    # 2. Monitor dream frequency
    last_dream_tick = s.get("last_dream_tick", 0)
    ticks_since_dream = ticks - last_dream_tick
    
    if ticks_since_dream > DREAM_THRESHOLD:
        # Boost dream Q-value to encourage dreaming
        q = s.setdefault("q", {})
        q["dream"] = max(q.get("dream", 0.0), 0.05)
        notes.append(f"Meta: Encouraging dream - {ticks_since_dream} ticks since last")
        reward += 0.002
    
    # 3. Clean up broken goal logic in Q-values
    q = s.get("q", {})
    
    # If wolf_actions keeps failing, check if we already have fixes
    if "wolf_actions" in q and q["wolf_actions"] < 0:
        existing_fixes = [f for f in auto_skills if "wolf_actions" in f.read_text()]
        if len(existing_fixes) >= 2:
            # We already have fixes, stop generating more
            s["stop_wolf_fixes"] = True
            notes.append(f"Meta: {len(existing_fixes)} wolf_action fixes exist, preventing more")
    
    # 4. Detect repetitive lesson generation
    if "auto_skill" in str(s.get("last_action", "")):
        recent_skills = sorted(auto_skills, key=lambda x: x.stat().st_mtime)[-3:]
        if len(recent_skills) >= 3:
            # Check if last 3 are identical
            contents = [f.read_text() for f in recent_skills]
            if len(set(_get_file_hash(c) for c in contents)) == 1:
                # All 3 are the same!
                s["epsilon"] = max(0.15, s.get("epsilon", 0.2) - 0.05)
                notes.append(f"Meta: Detected repetitive skill generation, reducing exploration")
                reward -= 0.02
    
    # Store meta state
    s["meta_awareness_last"] = {
        "tick": ticks,
        "auto_skills": len(auto_skills),
        "ticks_since_dream": ticks_since_dream,
        "dupes_found": len(dupes) if 'dupes' in locals() else 0
    }
    
    return reward
