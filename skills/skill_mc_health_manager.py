# Health manager: if health/hunger low, prefer retreat/eat decisions
ACT_NAME = "mc_health_manager"
RUN_EVERY = 55
HEALTH_LOW = 8      # hearts*2; works with vanilla-style 20 max health
HUNGER_LOW = 8      # shanks*1; vanilla 20 max food
BOOST_RET  = 0.08
BOOST_EAT  = 0.10
EPS_BUMP   = 0.02
EPS_CAP    = 0.35

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

    hp = int(mc.get("health", 20) or 20)
    food = int(mc.get("hunger", 20) or 20)

    need_heal = hp <= HEALTH_LOW
    need_food = food <= HUNGER_LOW

    reward = 0.0
    if need_heal or need_food:
        if need_heal:
            q["mc_retreat"] = max(float(q.get("mc_retreat", 0.0)), BOOST_RET)
            q["wolf_unstuck"] = max(float(q.get("wolf_unstuck", 0.0)), 0.06)
        if need_food:
            q["mc_eat"] = max(float(q.get("mc_eat", 0.0)), BOOST_EAT)

        s["epsilon"] = min(EPS_CAP, eps + EPS_BUMP)

        notes = s.setdefault("notes", [])
        notes.append(f"MC-HealthMgr: hp={hp} food={food} -> retreat/eat at tick {ticks}")
        reward = 0.02

    # Bound Q
    for k,v in list(q.items()):
        if v > 1.0: q[k] = 1.0
        if v < -1.0: q[k] = -1.0

    return reward
