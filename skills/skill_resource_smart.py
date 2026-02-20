# Seed-0 skill: resource_smart — adaptive resource governor for quiet background behavior.
# Goal: Use system resources smartly so Guy stays responsive and unobtrusive.
# Safe: stdlib-only, local state edits (epsilon/Q). No network/OS changes.
#
# What it does (every RUN_EVERY ticks):
# - Measures system pressure: CPU load, memory available, disk free, and this process' CPU usage.
# - When BUSY: damp Q of heavy actions, softly prefer light actions, reduce exploration a bit.
# - When IDLE: gently encourage heavy maintenance near their windows, allow more exploration.
# - Neutral reward unless it applies corrective adjustments (+0.02).
#
# Notes:
# - Heavy actions: self_bundle (zip+hash), self_verify (extract+hash), dream (log scan).
# - Light actions: heartbeat, reflect, compress_memory, signal, communicate.
# - Works without changing main.py by nudging epsilon and Q-values.

from pathlib import Path
import os, time, shutil

ACT_NAME = "resource_smart"

LOG_PATH     = Path("data/events.log")
RUN_EVERY    = 20            # evaluate ~every 20s
HEAVY        = {"self_bundle","self_verify","dream"}
LIGHT        = {"heartbeat","reflect","compress_memory","signal","communicate"}

# Load thresholds (normalized by CPU cores)
LOAD_BUSY    = 0.80
LOAD_IDLE    = 0.35

# Memory thresholds (MB)
MEM_LOW_MB   = 256
MEM_OK_MB    = 1024

# Disk thresholds (GB)
DISK_LOW_GB  = 1.0

# Epsilon band
EPS_FLOOR    = 0.08
EPS_IDLE_UP  = 0.04         # + when idle (capped later)
EPS_BUSY_DOWN= 0.03         # - when busy (floored later)
EPS_CAP      = 0.35
EPS_BASELINE = 0.15
EPS_COOL     = 0.998        # gentle drift toward baseline when normal

# Q adjustments
Q_DAMP_HEAVY = 0.80         # multiply heavy Q when busy
Q_MIN_LIGHT  = 0.03         # ensure light actions have at least this Q when busy
Q_MIN_HEAVY_IDLE = 0.05     # ensure heavy actions have at least this Q when idle
Q_NEAR_WIN   = 0.10         # bump heavy Q near their due windows
Q_MAX_ABS    = 1.0

def _norm_load():
    try:
        la1, _, _ = os.getloadavg()
        cores = max(1, os.cpu_count() or 1)
        return float(la1) / float(cores)
    except Exception:
        return 0.0

def _mem_available_mb():
    try:
        with open("/proc/meminfo","r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    kb = float(parts[1])
                    return kb/1024.0
    except Exception:
        pass
    return None  # unknown

def _disk_free_gb(path="."):
    try:
        du = shutil.disk_usage(path)
        return du.free / (1024**3)
    except Exception:
        return None

def _proc_cpu_fraction(state):
    # Estimate process CPU usage over the last interval
    now = time.time()
    cpu = time.process_time()
    last = state.get("_rs_last", {})
    last_cpu = float(last.get("cpu", cpu))
    last_ts  = float(last.get("ts", now))
    dt = max(1e-3, now - last_ts)
    frac = max(0.0, (cpu - last_cpu) / dt)  # ~0..Ncores but we treat it as fraction heuristically
    state["_rs_last"] = {"cpu": cpu, "ts": now}
    return frac

def _next_due(ticks: int, mod: int) -> int:
    return (-ticks) % mod

def act(ctx):
    s = ctx["state"]
    ticks = int(s.get("ticks", 0))
    if ticks % RUN_EVERY != 0:
        return 0.0

    q = s.setdefault("q", {})
    eps = float(s.get("epsilon", EPS_BASELINE))

    # Measurements
    load = _norm_load()
    mem_mb = _mem_available_mb()
    disk_gb = _disk_free_gb(".")
    pfrac = _proc_cpu_fraction(s)

    # Busy / Idle heuristics
    is_busy = (load >= LOAD_BUSY) or (pfrac >= 0.6) or (disk_gb is not None and disk_gb < DISK_LOW_GB) \
              or (mem_mb is not None and mem_mb < MEM_LOW_MB)
    is_idle = (load <= LOAD_IDLE) and (mem_mb is None or mem_mb >= MEM_OK_MB) and (pfrac <= 0.15)

    adjusted = False

    if is_busy:
        # Reduce exploration a bit (but keep floor)
        new_eps = max(EPS_FLOOR, eps - EPS_BUSY_DOWN)
        if new_eps != eps:
            s["epsilon"] = new_eps; eps = new_eps; adjusted = True
        # Damp heavy action Q, favor light actions minimum
        for a in list(q.keys()):
            if a in HEAVY:
                q[a] = float(q.get(a, 0.0)) * Q_DAMP_HEAVY; adjusted = True
            elif a in LIGHT and float(q.get(a, 0.0)) < Q_MIN_LIGHT:
                q[a] = Q_MIN_LIGHT; adjusted = True
        s["resource_mode"] = "busy"
    elif is_idle:
        # Allow a bit more exploration (cap at EPS_CAP)
        new_eps = min(EPS_CAP, eps + EPS_IDLE_UP)
        if new_eps != eps:
            s["epsilon"] = new_eps; eps = new_eps; adjusted = True
        # Encourage heavy maintenance slightly (especially near windows)
        for a in HEAVY:
            if float(q.get(a, 0.0)) < Q_MIN_HEAVY_IDLE:
                q[a] = Q_MIN_HEAVY_IDLE; adjusted = True
        nb = _next_due(ticks, 120)   # self_bundle window
        nv = _next_due(ticks, 180)   # self_verify window
        if nb <= 2:
            q["self_bundle"] = max(float(q.get("self_bundle", 0.0)), Q_NEAR_WIN); adjusted = True
        if nv <= 2:
            q["self_verify"] = max(float(q.get("self_verify", 0.0)), Q_NEAR_WIN); adjusted = True
        s["resource_mode"] = "idle"
    else:
        # Normal — drift epsilon gently toward a baseline
        target = EPS_BASELINE
        if abs(eps - target) > 1e-4:
            s["epsilon"] = max(EPS_FLOOR, min(EPS_CAP, (eps*0.98 + target*0.02)))
            adjusted = True
        s["resource_mode"] = "normal"

    # Clamp Q range
    for k, v in list(q.items()):
        if v > Q_MAX_ABS: q[k] = Q_MAX_ABS
        elif v < -Q_MAX_ABS: q[k] = -Q_MAX_ABS

    # Trace for monitoring
    s["resource_smart_last"] = {
        "tick": ticks,
        "mode": s.get("resource_mode"),
        "load_norm": round(load, 3),
        "mem_mb": None if mem_mb is None else int(mem_mb),
        "disk_gb": None if disk_gb is None else round(disk_gb, 2),
        "proc_cpu_frac": round(pfrac, 3),
        "epsilon_after": round(float(s.get("epsilon", eps)), 4),
        "adjusted": bool(adjusted)
    }

    return 0.02 if adjusted else 0.0
