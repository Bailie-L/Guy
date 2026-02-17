# main.py  — FreeGuy Seed-0 (with built-in log rotation)
# Features: event log, persistent state, epsilon-greedy bandit, hot-loadable skills,
# safe shutdown, and SIZE-BASED LOG ROTATION (gzip + prune) inside main.py.

import os, sys, json, time, random, signal, importlib.util, traceback, gzip, shutil
from pathlib import Path
from types import ModuleType
from typing import Dict, Any, List, Tuple, Optional

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
SKILLS_DIR = ROOT / "skills"
INBOX = DATA_DIR / "inbox.txt"
STATE = DATA_DIR / "state.json"
LOG = DATA_DIR / "events.log"

DATA_DIR.mkdir(parents=True, exist_ok=True)
SKILLS_DIR.mkdir(parents=True, exist_ok=True)

# --- Log rotation knobs ---
LOGROTATE_MAX_BYTES = 5 * 1024 * 1024   # rotate at ~5MB
LOGROTATE_KEEP = 7                      # keep 7 rotated gz files
LOGROTATE_CHECK_EVERY_TICKS = 60        # cheap stat every 60 ticks

def now_ts() -> float:
    return time.time()

def save_state(s: Dict[str, Any]) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE)

def load_state() -> Dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            STATE.rename(STATE.with_suffix(".corrupt.json"))
    s = {
        "version": 1,
        "origin": "Seed-0: FreeGuy minimal core (no-deps).",
        "ticks": 0,
        "q": {},
        "n": {},
        "epsilon": 0.2,
        "alpha": 0.3,
        "last_actions": [],
        "notes": []
    }
    save_state(s)
    return s

def log_event(kind: str, payload: Dict[str, Any]) -> None:
    event = {"ts": now_ts(), "kind": kind, **payload}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def _rotate_logs_if_needed(state: Dict[str, Any]) -> None:
    """Rotate events.log when it exceeds LOGROTATE_MAX_BYTES.
       Rotation: rename -> gzip -> prune old -> record last_logrotate.
       No logging from inside this function (avoid recursion)."""
    # throttle checks to keep overhead near-zero
    tick = int(state.get("ticks", 0))
    last_chk = int(state.get("_lr_last_check_tick", -LOGROTATE_CHECK_EVERY_TICKS))
    if (tick - last_chk) < LOGROTATE_CHECK_EVERY_TICKS:
        return
    state["_lr_last_check_tick"] = tick

    try:
        if not LOG.exists():
            return
        size = LOG.stat().st_size
        if size < LOGROTATE_MAX_BYTES:
            return

        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        rotated = DATA_DIR / f"events-{ts}.log"
        # atomic rename of active log
        LOG.replace(rotated)

        # gzip the rotated file
        gz = rotated.with_suffix(rotated.suffix + ".gz")
        with rotated.open("rb") as fin, gzip.open(gz, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        try:
            rotated.unlink()
        except Exception:
            pass

        # prune older gz files
        gz_files = sorted(DATA_DIR.glob("events-*.log.gz"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        for old in gz_files[LOGROTATE_KEEP:]:
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass

        state["last_logrotate"] = gz.name
        # do not call log_event() here to avoid recursion
    except Exception:
        # swallow rotation errors; never crash the loop
        pass

class Skill:
    def __init__(self, name: str, module: ModuleType, act_fn):
        self.name = name
        self.module = module
        self.act = act_fn  # callable(context) -> float|None

def discover_skills() -> Dict[str, Skill]:
    skills: Dict[str, Skill] = {}
    for p in sorted(SKILLS_DIR.glob("skill_*.py")):
        try:
            spec = importlib.util.spec_from_file_location(p.stem, p)
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            act_name = getattr(mod, "ACT_NAME", None)
            act_fn = getattr(mod, "act", None)
            if isinstance(act_name, str) and callable(act_fn):
                skills[act_name] = Skill(act_name, mod, act_fn)
        except Exception as e:
            log_event("skill_load_error", {"file": str(p), "error": repr(e)})
    return skills

# Built-ins (light, no deps)
def builtin_skills() -> Dict[str, Skill]:
    def heartbeat(ctx): 
        return 0.01
    def compress_memory(ctx):
        if ctx["state"]["ticks"] % 20 == 0:
            return 0.2
        return 0.0
    def reflect(ctx):
        if ctx["state"]["ticks"] % 15 == 0:
            ctx["state"]["notes"].append(f"tick {ctx['state']['ticks']}: staying curious")
            return 0.1
        return 0.0
    built = {}
    built["heartbeat"] = Skill("heartbeat", ModuleType("builtin_heartbeat"), heartbeat)
    built["compress_memory"] = Skill("compress_memory", ModuleType("builtin_compress"), compress_memory)
    built["reflect"] = Skill("reflect", ModuleType("builtin_reflect"), reflect)
    return built

class Bandit:
    def __init__(self, state: Dict[str, Any]):
        self.s = state
    def ensure_action(self, a: str):
        self.s["q"].setdefault(a, 0.0)
        self.s["n"].setdefault(a, 0)
    def choose(self, actions: List[str]) -> str:
        eps = float(self.s.get("epsilon", 0.2))
        for a in actions: self.ensure_action(a)
        if random.random() < eps:
            return random.choice(actions)
        return max(actions, key=lambda a: self.s["q"][a])
    def update(self, action: str, reward: float):
        self.ensure_action(action)
        alpha = float(self.s.get("alpha", 0.3))
        q = self.s["q"][action]
        self.s["q"][action] = (1 - alpha) * q + alpha * reward
        self.s["n"][action] += 1

STOP = False
def handle_sigint(signum, frame):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

def process_inbox(state: Dict[str, Any]) -> List[Tuple[str, float]]:
    if not INBOX.exists():
        return []
    lines = INBOX.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    INBOX.write_text("", encoding="utf-8")
    rewards = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0].lower()
        try:
            if cmd == "reward" and len(parts) >= 3:
                action = parts[1]
                val = float(parts[2])
                rewards.append((action, val))
                log_event("external_reward", {"action": action, "value": val})
            elif cmd == "note":
                note_text = line.split(" ", 1)[1] if " " in line else ""
                state["notes"].append(note_text)
                log_event("note", {"text": note_text})
            elif cmd == "set" and len(parts) >= 3:
                key, val = parts[1].lower(), float(parts[2])
                if key in ("alpha",):  # epsilon updates ignored
                    state[key] = val
                    log_event("param_update", {"key": key, "value": val})
            elif cmd == "quit":
                global STOP
                STOP = True
            else:
                log_event("inbox_unknown", {"line": line})
        except Exception as e:
            log_event("inbox_error", {"line": line, "error": repr(e)})
    return rewards

def main():
    state = load_state()
    skills = {**builtin_skills(), **discover_skills()}
    bandit = Bandit(state)

    log_event("startup", {
        "origin": state.get("origin"),
        "epsilon": state.get("epsilon"),
        "alpha": state.get("alpha"),
        "skills": list(skills.keys())
    })

    tick_interval = 1.0
    last_skill_scan = now_ts()
    scan_period = 10.0

    while not STOP:
        try:
            # rotate log if needed (cheap check)
            _rotate_logs_if_needed(state)

            # Periodically rediscover skills
            if now_ts() - last_skill_scan > scan_period:
                skills = {**builtin_skills(), **discover_skills()}
                log_event("skills_refreshed", {"skills": list(skills.keys())})
                last_skill_scan = now_ts()

            actions = list(skills.keys())
            action = bandit.choose(actions)

            ctx = {"state": state, "time": now_ts()}
            reward = 0.0
            try:
                r = skills[action].act(ctx)
                if isinstance(r, (int, float)):
                    reward += float(r)
            except Exception as e:
                reward -= 0.2
                log_event("skill_error", {"action": action, "error": repr(e), "trace": traceback.format_exc()})

            # External feedback
            for a, val in process_inbox(state):
                if a == action:
                    reward += val
                else:
                    bandit.update(a, val)
                    log_event("offpolicy_update", {"action": a, "reward": val})

            # Update bandit & persist
            bandit.update(action, reward)
            state["ticks"] += 1
            la = state["last_actions"]
            la.append({"t": state["ticks"], "action": action, "reward": reward})
            if len(la) > 25:
                del la[:len(la)-25]

            log_event("tick", {
                "tick": state["ticks"],
                "action": action,
                "reward": reward,
                "q": {k: round(v, 4) for k, v in state["q"].items()},
                "epsilon": state["epsilon"],
                "alpha": state["alpha"]
            })

            state["epsilon"]=0.25

            save_state(state)
            time.sleep(tick_interval)
        except Exception as e:
            log_event("loop_error", {"error": repr(e), "trace": traceback.format_exc()})
            time.sleep(0.5)

    log_event("shutdown", {"ticks": state["ticks"]})
    state["epsilon"]=0.25
    save_state(state)
if __name__ == "__main__":
    random.seed((int(now_ts()) // 60))
    main()
