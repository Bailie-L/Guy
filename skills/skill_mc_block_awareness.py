# Block awareness: avoid head-in-block / feet-in-block & water-at-head situations
from pathlib import Path
import json
from collections import deque, Counter

ACT_NAME = "mc_block_awareness"
RUN_EVERY = 40        # check every 40 ticks
EPS_BUMP = 0.03
EPS_CAP  = 0.35
DAMP     = 0.90       # damp risky action
BOOST    = 0.08       # boost safer action

SOLID = {"stone","dirt","deepslate","granite","andesite","diorite","cobblestone","oak_log","spruce_log","sand","gravel"}
LIQUID = {"water","bubble_column","lava"}

def _mc(s):
    return s.get("mc", {}) if isinstance(s.get("mc"), dict) else {}

def _in_suffocation(mc):
    head = str(mc.get("block_head","")).lower()
    feet = str(mc.get("block_feet","")).lower()
    in_wall = bool(mc.get("in_wall", False))
    # Suffocation risk if in_wall or head is a solid block (not air/water)
    return in_wall or (head not in ("air","cave_air") and head not in LIQUID and (head in SOLID or head not in ("air","water","lava","void")))

def _head_in_liquid(mc):
    head = str(mc.get("block_head","")).lower()
    return head in LIQUID

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0

    mc = _mc(s)
    q = s.setdefault("q", {})
    eps = float(s.get("epsilon", 0.2))

    risky = _in_suffocation(mc)
    wet_head = _head_in_liquid(mc)

    reward = 0.0
    if risky or wet_head:
        # Prefer moves that get to air and unstick
        q["wolf_unstuck"] = max(float(q.get("wolf_unstuck", 0.0)), BOOST)
        q["mc_move_to_air"] = max(float(q.get("mc_move_to_air", 0.0)), BOOST)

        # Soften generic wolf_actions to reduce pushing into the block again
        if "wolf_actions" in q:
            q["wolf_actions"] = float(q["wolf_actions"]) * DAMP

        # Explore alternatives slightly more when risky
        s["epsilon"] = min(EPS_CAP, eps + EPS_BUMP)

        notes = s.setdefault("notes", [])
        notes.append(f"MC-BlockAwareness: risky={risky} wet_head={wet_head} -> boost move_to_air/unstuck at tick {ticks}")
        reward = 0.02

    # Bound Q
    for k,v in list(q.items()):
        if v > 1.0: q[k] = 1.0
        if v < -1.0: q[k] = -1.0

    return reward
