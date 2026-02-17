# Hazard avoidance: avoid lava/fire/fall and re-path aggressively
ACT_NAME = "mc_hazard_avoidance"
RUN_EVERY = 45
BOOST = 0.09
DAMP  = 0.88
EPS_BUMP = 0.03
EPS_CAP  = 0.35

def _mc(s):
    return s.get("mc", {}) if isinstance(s.get("mc"), dict) else {}

def _hazards(mc):
    return {
        "lava": bool(mc.get("near_lava", False) or str(mc.get("block_head","")).lower()=="lava"),
        "fire": bool(mc.get("on_fire", False)),
        "fall": bool(mc.get("fall_risk", False) or (mc.get("fall_distance", 0) or 0) > 3),
        "suffo": bool(mc.get("in_wall", False))
    }

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0

    mc = _mc(s)
    q  = s.setdefault("q", {})
    eps = float(s.get("epsilon", 0.2))
    hz = _hazards(mc)
    risky = any(hz.values())

    reward = 0.0
    if risky:
        # Prefer re-path/escape
        q["mc_path_recalc"] = max(float(q.get("mc_path_recalc", 0.0)), BOOST)
        q["wolf_unstuck"]   = max(float(q.get("wolf_unstuck", 0.0)), 0.07)

        if "wolf_actions" in q:
            q["wolf_actions"] = float(q["wolf_actions"]) * DAMP

        s["epsilon"] = min(EPS_CAP, eps + EPS_BUMP)

        notes = s.setdefault("notes", [])
        notes.append(f"MC-HazardAvoid: lava={hz['lava']} fire={hz['fire']} fall={hz['fall']} suffo={hz['suffo']} -> path_recalc/unstuck at tick {ticks}")
        reward = 0.02

    # Bound Q
    for k,v in list(q.items()):
        if v > 1.0: q[k] = 1.0
        if v < -1.0: q[k] = -1.0

    return reward
