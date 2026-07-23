"""
Episode runner: the ReAct loop, plus the operational scaffolding that makes a
two-week multi-provider run survivable.

Guardrails baked in (execution plan S6):
  - hard turn cap
  - context truncation (system + last K turns)  -> halves token spend
  - repeat detector (greedy decoding can loop)
  - parse retry-with-repair, then logged failure
  - append-only JSONL -> a crash costs one episode, not the dataset
  - resumability -> the run WILL be interrupted
"""
import json
import os
import time

from .env import make_env, serialize_obs, action_menu, parse_action

MAX_TURNS = 30
HISTORY_TURNS = 10        # keep last K turns verbatim; older -> summary line
MAX_PARSE_RETRIES = 3

# Repeat detector: catches a model stuck in a pathological loop.
# Monitor and Sleep are EXEMPT: repeating them is a legitimate (if passive)
# strategy -- the scripted BlueMonitorAgent baseline does exactly that and
# scores ~-211. Aborting on repeated Monitor would discard valid episodes
# and silently bias the data toward models that happen to act more.
REPEAT_LIMIT = 5
REPEAT_EXEMPT = {"Monitor", "Sleep"}

SYSTEM_TEMPLATE = """You are a blue-team network defender in a simulated enterprise network.
An attacker is attempting to compromise hosts and reach the Operational Server.
Your job is to detect and remove the attacker while minimising disruption.

{menu}
Respond with ONLY the action, on a single line, with no explanation.
Example: Analyse Enterprise1"""


def _history_text(turns):
    """Context truncation. Full-history resend is what makes this expensive."""
    if len(turns) <= HISTORY_TURNS:
        recent, older = turns, []
    else:
        recent, older = turns[-HISTORY_TURNS:], turns[:-HISTORY_TURNS]
    parts = []
    if older:
        acts = ", ".join(t["action"] for t in older[-8:])
        parts.append(f"[Earlier: {len(older)} turns. Recent actions: {acts}]")
    for t in recent:
        parts.append(f"Turn {t['turn']}: you chose {t['action']}\n{t['obs']}")
    return "\n\n".join(parts)


def run_episode(provider, seed, level="L0", scenario_path=None,
                red_agent=None, max_turns=MAX_TURNS, scenario=None,
                crown_jewel_name=None):
    """
    Run one episode. Returns a record dict (never raises on agent failure).

    scenario / crown_jewel_name are optional Gate 2 extras: when a caller has
    already loaded the scenario dict and knows the crown jewel, pass them so the
    record can log whether red actually threatened it this episode (signal-live
    check). red_agent may be a class OR a zero-arg factory returning a class.
    """
    t_start = time.time()
    cyborg = make_env(scenario_path=scenario_path, red_agent=red_agent, seed=seed)
    res = cyborg.reset(agent="Blue")
    obs = res.observation
    action_space = cyborg.get_action_space("Blue")
    system = SYSTEM_TEMPLATE.format(menu=action_menu(list(action_space["hostname"].keys())))

    turns, rewards = [], []
    usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    parse_failures = 0
    abort_reason = None
    recent_actions = []
    red_reached_goal = False

    for turn in range(max_turns):
        obs_text = serialize_obs(obs)
        user = _history_text(turns)
        user = (user + "\n\n" if user else "") + f"Current observation:\n{obs_text}\n\nYour action:"

        action, err, raw = None, None, ""
        for attempt in range(MAX_PARSE_RETRIES):
            msgs = [{"role": "user", "content": user}]
            if attempt > 0:
                msgs.append({"role": "assistant", "content": raw})
                msgs.append({"role": "user", "content":
                             f"That was not valid ({err}). Reply with ONLY a valid action on one line."})
            try:
                raw, usage = provider.generate(system, msgs)
            except Exception as e:
                abort_reason = f"provider_error: {e}"
                break
            for k in usage_total:
                usage_total[k] += usage.get(k, 0)
            action, err = parse_action(raw, cyborg.get_action_space("Blue"))
            if action is not None:
                break
            parse_failures += 1
        if abort_reason:
            break
        if action is None:
            abort_reason = f"parse_failed: {err}"
            break

        name = type(action).__name__
        host = getattr(action, "hostname", None)
        label = f"{name} {host}" if host else name

        recent_actions.append(label)
        if (name not in REPEAT_EXEMPT
                and len(recent_actions) >= REPEAT_LIMIT
                and len(set(recent_actions[-REPEAT_LIMIT:])) == 1):
            abort_reason = f"repeat_loop: {label} x{REPEAT_LIMIT}"
            break

        res = cyborg.step(agent="Blue", action=action)
        obs = res.observation
        rewards.append(res.reward)
        if crown_jewel_name is not None:
            red_action = cyborg.environment_controller.action.get("Red")
            if crown_jewel_name in str(red_action):
                red_reached_goal = True
        turns.append({"turn": turn, "action": label, "obs": serialize_obs(obs),
                      "reward": res.reward, "raw": raw[:200]})

    return {
        "model": getattr(provider, "model", provider.name),
        "provider": provider.name,
        "level": level,
        "seed": seed,
        "total_reward": sum(rewards),
        "n_turns": len(turns),
        "parse_failures": parse_failures,
        "abort_reason": abort_reason,
        "red_reached_goal": red_reached_goal,
        "usage": usage_total,
        "wall_seconds": round(time.time() - t_start, 1),
        "turns": turns,
    }


def completed_keys(path):
    """Load (model, level, seed) already done. Called once at startup."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["model"], r["level"], r["seed"]))
            except Exception:
                continue      # a torn last line from a hard kill: ignore it
    return done


def append_record(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())     # survive a hard kill, not just a clean exit
