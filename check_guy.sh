#!/usr/bin/env bash
# Guy check-in: single-file status + health snapshot (stdlib only).
# Runs from any directory; no writes; safe to use while Guy is running.

set -Eeuo pipefail
cd "$(dirname "$0")"

python3 - <<'PY'
import os, sys, json, pathlib, time, math, shutil, subprocess, collections
from datetime import datetime, timedelta

CWD = pathlib.Path.cwd()
DATA = CWD / "data"
LOG  = DATA / "events.log"
STATE= DATA / "state.json"
OUTBOX = DATA / "outbox"
INBOX  = DATA / "inbox.txt"
FORCE  = DATA / "force"
RELEASES = DATA / "releases"
SKILLS = CWD / "skills"

def human_td(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))

def age(path: pathlib.Path) -> str:
    try:
        return human_td(time.time() - path.stat().st_mtime)
    except Exception:
        return "n/a"

def safe_json_lines(path: pathlib.Path):
    if not path.exists():
        return []
    lines = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                # tolerate non-JSON noise lines
                pass
    return lines

def ps_list():
    try:
        # find python processes running main.py
        p = subprocess.run(
            ["pgrep","-af","python.*main\\.py"],
            check=False, capture_output=True, text=True
        )
        lines = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
        procs = []
        for ln in lines:
            parts = ln.strip().split(maxsplit=1)
            if not parts: continue
            pid = parts[0]
            # get elapsed
            p2 = subprocess.run(
                ["ps","-o","pid,etime,cmd","-p",pid],
                check=False, capture_output=True, text=True
            )
            for row in p2.stdout.strip().splitlines()[1:]:
                procs.append(row.strip())
        return procs
    except Exception:
        return []

def disk_free(path: pathlib.Path) -> str:
    try:
        total, used, free = shutil.disk_usage(path)
        # report free GB
        return f"{free/ (1024**3):.1f} GB"
    except Exception:
        return "n/a"

# ---------- Pre-flight ----------
if not DATA.exists():
    print("❌ Not in a Guy project root (data/ missing).")
    sys.exit(1)

# ---------- Load state ----------
state = {}
try:
    state = json.loads(STATE.read_text(encoding='utf-8'))
except Exception as e:
    print(f"❌ state.json unreadable: {e}")
    sys.exit(1)

ticks = int(state.get("ticks", 0))
epsilon = float(state.get("epsilon", 0.0))
alpha   = float(state.get("alpha", 0.0))
qmap    = dict(state.get("q", {}))

# ---------- Load events (tolerant) ----------
events = safe_json_lines(LOG)
ticks_events = [e for e in events if e.get("kind") == "tick"]
recent_ticks = ticks_events[-500:] if ticks_events else []
recent_actions = [e.get("action") for e in recent_ticks if e.get("action")]

now = time.time()
last_event_ts = None
if events:
    last = events[-1]
    last_event_ts = float(last.get("ts")) if isinstance(last.get("ts"), (int,float,str)) and str(last.get("ts")).replace('.','',1).isdigit() else LOG.stat().st_mtime
else:
    last_event_ts = LOG.stat().st_mtime if LOG.exists() else None

# ---------- Liveness ----------
procs = ps_list()
stalled = False
if last_event_ts is not None:
    stalled = (now - last_event_ts) > 15  # >15s without any new event

# ---------- Heartbeat & diversity ----------
action_counts = collections.Counter(recent_actions)
hb_q = float(qmap.get("heartbeat", 0.0))
hb_count = action_counts.get("heartbeat", 0)
total_recent = len(recent_actions)
hb_share = (hb_count/total_recent*100) if total_recent else 0.0

unique_recent_100 = len(set(recent_actions[-100:])) if total_recent >= 1 else 0
# Shannon entropy over last 500 ticks
entropy = 0.0
for c in action_counts.values():
    p = c / (total_recent or 1)
    if p > 0:
        entropy -= p * math.log(p)

# top Qs
sorted_q = sorted(qmap.items(), key=lambda kv: kv[1], reverse=True)
# bar scaled to 60 chars max
def bar(v):
    try:
        return "█" * min(60, max(0, int(round(v*600))))
    except Exception:
        return ""

# dream summary (conservative: use what's in state)
dreams = state.get("dreams", [])
last_dream_tick = (dreams[-1].get("tick", 0) if dreams else 0) or 0
ticks_since_dream = max(0, ticks - last_dream_tick)

# trend over two windows of 100 if available
trend_info = None
if total_recent >= 200:
    old_100 = recent_actions[:100]
    new_100 = recent_actions[-100:]
    old_hb = (old_100.count('heartbeat') / 100) * 100
    new_hb = (new_100.count('heartbeat') / 100) * 100
    trend = new_hb - old_hb
    trend_info = (old_hb, new_hb, trend)

# system health: file sizes, messages, notes
log_size_mb = (LOG.stat().st_size / (1024*1024)) if LOG.exists() else 0.0
outbox_msgs = len(list(OUTBOX.glob("msg-*.json"))) if OUTBOX.exists() else 0
notes_count = len(state.get("notes", [])) if isinstance(state.get("notes", []), list) else 0

# Force flags
force_flags = []
if FORCE.exists():
    for f in sorted(FORCE.glob("*.force")):
        force_flags.append((f.name, age(f)))

# Releases (last 3)
recent_releases = []
if RELEASES.exists():
    recent_releases = sorted(RELEASES.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
    recent_releases = [(p.name, age(p)) for p in recent_releases]

# Skills (totals + autos + autos today)
auto_skills = []
all_skills  = []
today_start = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
if SKILLS.exists():
    for p in SKILLS.glob("*.py"):
        all_skills.append(p)
        if p.name.startswith("skill_auto_"):
            auto_skills.append(p)
    autos_today = sum(1 for p in auto_skills if p.stat().st_mtime >= today_start)
else:
    autos_today = 0

# Error scan (last 2000 lines)
err_lines = 0
tracebacks = 0
scan_tail = events[-2000:] if len(events) > 2000 else events
for raw in scan_tail:
    # crude scan by serializing back to text
    s = json.dumps(raw, ensure_ascii=False)
    if "Traceback" in s:
        tracebacks += 1
    if '"error"' in s.lower():
        err_lines += 1

# Outbox recency, Inbox backlog (tail preview only)
def newest_mtime(path_glob):
    latest = None
    for p in path_glob:
        mt = p.stat().st_mtime
        latest = mt if (latest is None or mt > latest) else latest
    return latest

outbox_latest = newest_mtime(OUTBOX.glob("msg-*.json")) if OUTBOX.exists() else None
inbox_tail = []
if INBOX.exists():
    try:
        with INBOX.open('r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            inbox_tail = [ln.rstrip() for ln in lines[-5:]]
    except Exception:
        pass

# ---------- Print report ----------
print("="*70)
print("🤖 GUY STATUS REPORT - Enhanced Analytics")
print("="*70)

# Liveness block
print("🧠 Process:")
if procs:
    for row in procs:
        print(f"  {row}")
else:
    print("  ❌ No running python main.py process found")

last_event_age = (now - last_event_ts) if last_event_ts else None
if last_event_age is not None:
    print(f"  Last event age: {human_td(last_event_age)} {'⚠️ STALLED' if stalled else 'OK'}")
else:
    print("  Last event age: n/a")

print(f"\n⏱️  Ticks: {ticks:,}")
print(f"⏳ Uptime (from ticks): {human_td(ticks)}")
print(f"🎲 Epsilon: {epsilon:.4f} (exploration)")
print(f"📚 Alpha:   {alpha:.4f} (learning)")

# Heartbeat analysis
print("\n💓 HEARTBEAT ANALYSIS:")
print(f"  Q-value: {hb_q:+.4f} (hardcoded reward ≈ 0.01)")
print(f"  Recent dominance: {hb_share:.1f}% ({hb_count}/{total_recent} last actions)")
status = "🟢 CONTROLLED"
if hb_share > 50: status = "🔴 DOMINANT"
elif hb_share > 30: status = "🟡 REDUCING"
print(f"  Status: {status}")

# Q-Value rankings
if sorted_q:
    print("\n📊 Q-VALUE RANKINGS (top 8):")
    for i,(act,val) in enumerate(sorted_q[:8],1):
        crown = "👑" if i==1 else " "
        print(f"  {crown} {act:24s} {val:+.5f} {bar(val)}")

# Diversity
print("\n🌈 DIVERSITY (last 500 ticks):")
print(f"  Unique actions used: {len(action_counts)}")
print(f"  Unique (last 100):   {unique_recent_100}")
print(f"  Shannon entropy:     {entropy:.3f}")
if action_counts:
    top3 = ", ".join(f"{a}({c})" for a,c in action_counts.most_common(3))
    print(f"  Top 3 actions:       {top3}")
else:
    print("  Top 3 actions:       n/a")

# Dream status (conservative threshold: > 86400 ticks = 24h)
print("\n💭 DREAM STATUS:")
print(f"  Total dreams:           {len(dreams)}")
print(f"  Ticks since last dream: {ticks_since_dream}")
print(f"  Dream Q-value:          {float(qmap.get('dream',0.0)):+.4f}")
print(f"  Status: {'⚠️ OVERDUE' if ticks_since_dream > 86400 else '✅ OK'}")

# Trend
if trend_info:
    old_hb,new_hb,trend = trend_info
    direction = "📉 improving" if trend < -5 else "📈 worsening" if trend > 5 else "➡️ stable"
    print("\n📈 BEHAVIORAL TREND (two 100-tick windows):")
    print(f"  Heartbeat % (older): {old_hb:.1f}%")
    print(f"  Heartbeat % (newer): {new_hb:.1f}%")
    print(f"  Trend: {trend:+.1f}% {direction}")

# System health
print("\n🔧 SYSTEM HEALTH:")
print(f"  events.log size:   {log_size_mb:.2f} MB")
print(f"  Outbox messages:   {outbox_msgs}")
print(f"  Notes saved:       {notes_count}")
print(f"  Disk free:         {disk_free(CWD)}")

# Force flags
print("\n🧩 FORCE FLAGS:")
if force_flags:
    for name, a in force_flags:
        print(f"  {name:20s} age={a}")
else:
    print("  (none)")

# Releases
print("\n📦 RELEASES (latest 3):")
if recent_releases:
    for n, a in recent_releases:
        print(f"  {n:40s} age={a}")
else:
    print("  (none)")

# Skills
print("\n🛠️  SKILLS:")
print(f"  Total skills:    {len(all_skills)}")
print(f"  Auto skills:     {len(auto_skills)}")
print(f"  Autos today:     {autos_today if 'autos_today' in locals() else 0}")

# Outbox & Inbox recency
if outbox_latest:
    print(f"\n📮 OUTBOX: last msg age {human_td(now - outbox_latest)}")
else:
    print("\n📮 OUTBOX: none")

print("\n📥 INBOX (last 5 lines):")
if inbox_tail:
    for ln in inbox_tail:
        print(f"  {ln}")
else:
    print("  (empty or unreadable)")

# Errors scan
print("\n🚨 ERROR SCAN (tail of events):")
print(f"  Tracebacks: {tracebacks}")
print(f"  'error' lines: {err_lines}")

# Overall assessment (purely descriptive; no side effects)
assessment = "✅ GOOD - Moderate diversity"
if hb_share < 30 and unique_recent_100 > 10:
    assessment = "✨ EXCELLENT - High diversity, low heartbeat"
elif hb_share >= 60:
    assessment = "🔴 STUCK - Heartbeat heavy"
elif hb_share < 50 and unique_recent_100 <= 7:
    assessment = "🟡 IMPROVING - Diversity still low"

if stalled or not procs:
    assessment = "🛑 ATTENTION - Process missing or stalled"

print(f"\n📋 ASSESSMENT: {assessment}")
print("="*70)
PY
