"""
Wolf Actions Skill for Guy
Upgraded AAA-grade version with full logging, JSON outbox,
and reward integration for Ghost (the wolf).
"""

from pathlib import Path
import json, random, time

ACT_NAME = "wolf_actions"
OUTBOX = Path("data/outbox")
INBOX = Path(__file__).resolve().parents[1] / "data" / "inbox.txt"
OUTBOX.mkdir(parents=True, exist_ok=True)


def act(ctx):
    state = ctx.get("state", {})
    q_values = ctx.get("q_values", {})

    wolf_actions = ["guard", "patrol", "idle", "follow", "attack", "howl"]

    # Exploration–exploitation tradeoff
    epsilon = float(state.get("epsilon", 0.3))
    if random.random() < epsilon:
        action = random.choice(wolf_actions)
        mode = "explore"
    else:
        action = max(wolf_actions, key=lambda a: q_values.get(a, 0.0))
        mode = "exploit"

    # Build structured outbox message
    msg = {
        "ts": time.time(),
        "tick": state.get("ticks"),
        "action": ACT_NAME,
        "wolf_action": action,
      "subtype": action,
        "mode": mode,
        "q_value": q_values.get(action, 0.0),
    }

    # Write JSON log to outbox
    out_file = OUTBOX / f"wolf-{int(msg['ts'])}.json"
    out_file.write_text(json.dumps(msg, indent=2))

    # Also update simple flag file for external bridge
    (OUTBOX / "wolf_action.txt").write_text(action)

    # Emit reward hint into inbox for Ghost integration
    with INBOX.open("a") as f:
        f.write("reward wolf_actions +0.01\n")

    # Small baseline return so Guy’s Q loop sees it
    return 0.01
