# Removes Q-values for skills that no longer exist
from pathlib import Path

ACT_NAME = "cleanup_orphans"
SKILLS_DIR = Path("skills")

def act(ctx):
    s = ctx["state"]
    
    # Get actual existing skills
    existing = set()
    for p in SKILLS_DIR.glob("skill_*.py"):
        skill_name = p.stem.replace("skill_", "")
        existing.add(skill_name)
    
    # Add built-in skills
    existing.update(["heartbeat", "compress_memory", "reflect"])
    
    # Find orphans in Q-table
    q = s.get("q", {})
    orphans = [k for k in q.keys() if k not in existing]
    
    # Remove orphans
    for orphan in orphans:
        del q[orphan]
        if orphan in s.get("n", {}):
            del s["n"][orphan]
    
    if orphans:
        s.setdefault("notes", []).append(f"Cleaned orphans: {orphans}")
        return 0.01
    
    return 0.0
