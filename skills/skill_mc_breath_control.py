# Breath control: watch air level & underwater state; climb/swim to air when needed
ACT_NAME = "mc_breath_control"
RUN_EVERY = 30
AIR_LOW   = 6          # if remaining air <= this, treat as urgent (Minecraft max air ~300 ticks, but bridges often normalize)
EPS_BUMP  = 0.04
EPS_CAP   = 0.35
BOOST     = 0.10
DAMP      = 0.90

def _mc(s):
    return s.get("mc", {}) if isinstance(s.get("mc"), dict) else {}

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0

    mc = _mc(s)
    q  = s.setdefault("q", {})
    eps = float(s.get("epsilon", 0.2))

    air  = int(mc.get("air", 300) or 300)
    underwater = bool(mc.get("is_underwater", False))

    reward = 0.0
    if underwater or air <= AIR_LOW:
        # Encourage reaching air
        q["mc_swim_up"] = max(float(q.get("mc_swim_up", 0.0)), BOOST)
        q["mc_move_to_air"] = max(float(q.get("mc_move_to_air", 0.0)), BOOST)
        q["wolf_unstuck"] = max(float(q.get("wolf_unstuck", 0.0)), 0.06)

        # Reduce persistence on generic actions that might force bad pathing
        if "wolf_actions" in q:
            q["wolf_actions"] = float(q["wolf_actions"]) * DAMP

        s["epsilon"] = min(EPS_CAP, eps + EPS_BUMP)

        notes = s.setdefault("notes", [])
        notes.append(f"MC-BreathControl: underwater={underwater} air={air} -> swim_up/move_to_air at tick {ticks}")
        reward = 0.02

    # Bound Q
    for k,v in list(q.items()):
        if v > 1.0: q[k] = 1.0
        if v < -1.0: q[k] = -1.0

    return reward
