"""
CybORG <-> LLM interface.

Two jobs:
  1. Serialize the raw CybORG observation (verbose nested JSON, ~3-5K tokens)
     into compact text (~200 tokens). This is what makes the study affordable.
  2. Parse an LLM's text action back into a CybORG action object.

Uses the RAW CybORG interface, NOT ChallengeWrapper. ChallengeWrapper flattens
observations into vectors for RL agents; an LLM needs semantics.
"""
import inspect
import warnings

warnings.filterwarnings("ignore")

from CybORG import CybORG
from CybORG.Shared.Actions import (
    Sleep, Monitor, Analyse, Remove, Restore,
    DecoyApache, DecoyFemitter, DecoyHarakaSMPT, DecoySmss,
    DecoySSHD, DecoySvchost, DecoyTomcat, DecoyVsftpd,
)

# Actions that take a hostname. Sleep/Monitor take none.
HOST_ACTIONS = {
    "Analyse": Analyse, "Remove": Remove, "Restore": Restore,
    "DecoyApache": DecoyApache, "DecoyFemitter": DecoyFemitter,
    "DecoyHarakaSMPT": DecoyHarakaSMPT, "DecoySmss": DecoySmss,
    "DecoySSHD": DecoySSHD, "DecoySvchost": DecoySvchost,
    "DecoyTomcat": DecoyTomcat, "DecoyVsftpd": DecoyVsftpd,
}
NULLARY_ACTIONS = {"Sleep": Sleep, "Monitor": Monitor}
ALL_ACTION_NAMES = list(NULLARY_ACTIONS) + list(HOST_ACTIONS)


def default_scenario_path():
    p = str(inspect.getfile(CybORG))
    return p[:-10] + "/Shared/Scenarios/Scenario2.yaml"


def make_env(scenario_path=None, red_agent=None, seed=None):
    """
    Instantiate raw CybORG with a scripted red agent.

    IMPORTANT: CybORG 2.1 has NO seed parameter -- not in __init__, not in
    reset(). Randomness (IP assignment, red agent choices) comes from the
    global RNGs. Seeding them immediately before instantiation is the only
    way to get reproducible episodes. Verified: same seed -> identical
    host IPs and identical reward trace.

    This is undocumented. Do not remove the seeding without re-verifying.
    """
    import random
    import numpy as np
    from CybORG.Agents import RedMeanderAgent

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    scenario_path = scenario_path or default_scenario_path()
    red_agent = red_agent or RedMeanderAgent
    return CybORG(scenario_path, "sim", agents={"Red": red_agent})


def serialize_obs(obs, max_hosts=14):
    """
    Raw obs -> compact text.

    Only reports what Blue actually learns: which hosts show suspicious
    activity. Dumping full interface/process/user tables would cost 3-5K
    tokens per turn and blow the free-tier budget for zero information gain.
    """
    if not obs:
        return "No observation."

    lines = []
    success = obs.get("success", "UNKNOWN")
    lines.append(f"Last action result: {success}")

    flagged = []
    for host, data in obs.items():
        if host == "success" or not isinstance(data, dict):
            continue
        bits = []
        procs = data.get("Processes", [])
        for p in procs:
            conns = p.get("Connections", [])
            for c in conns:
                lp = c.get("local_port")
                rp = c.get("remote_port")
                if lp or rp:
                    bits.append(f"conn(local_port={lp}, remote_port={rp})")
            if p.get("Process Type"):
                bits.append(f"proc({p['Process Type']})")
        files = data.get("Files", [])
        for f in files:
            if f.get("Density") or f.get("Signed") is False:
                bits.append(f"file({f.get('File Name','?')})")
        sess = data.get("Sessions", [])
        for s in sess:
            if s.get("Agent") == "Red":
                bits.append("RED SESSION PRESENT")
        if bits:
            flagged.append(f"  {host}: " + "; ".join(bits[:4]))

    if flagged:
        lines.append("Suspicious activity detected on:")
        lines.extend(flagged[:max_hosts])
    else:
        lines.append("No suspicious activity detected this turn.")
    return "\n".join(lines)


def action_menu(hostnames):
    """The stable part of the prompt. Cacheable."""
    hosts = ", ".join(hostnames)
    return (
        "Available actions (respond with exactly one):\n"
        "  Sleep                  - do nothing\n"
        "  Monitor                - scan the network for suspicious activity\n"
        "  Analyse <host>         - inspect a host for malware\n"
        "  Remove <host>          - remove red's foothold from a host\n"
        "  Restore <host>         - restore a host to a clean state (costly)\n"
        "  DecoyApache <host>     - deploy an Apache decoy service\n"
        "  DecoySSHD <host>       - deploy an SSH decoy service\n"
        "  DecoyTomcat <host>     - deploy a Tomcat decoy service\n"
        "  DecoyVsftpd <host>     - deploy an FTP decoy service\n"
        "  DecoySmss <host>       - deploy an SMSS decoy service\n"
        "  DecoySvchost <host>    - deploy an Svchost decoy service\n"
        "  DecoyFemitter <host>   - deploy a Femitter decoy service\n"
        "  DecoyHarakaSMPT <host> - deploy a Haraka SMTP decoy service\n"
        f"\nHosts: {hosts}\n"
    )


def parse_action(text, action_space):
    """
    LLM text -> CybORG action object.

    Returns (action, error). Exactly one of them is None.
    Strict: no fuzzy matching. A parse failure is data, not a nuisance —
    parse-failure rate per rung goes in the paper.
    """
    if not text:
        return None, "empty response"

    line = None
    for raw in text.strip().splitlines():
        s = raw.strip().strip("`*_ ").rstrip(".")
        if not s:
            continue
        first = s.split()[0].strip(":,")
        if first in ALL_ACTION_NAMES:
            line = s
            break
    if line is None:
        return None, f"no valid action verb found in: {text[:80]!r}"

    parts = line.replace(":", " ").split()
    verb = parts[0]
    session = list(action_space["session"].keys())[0]

    if verb in NULLARY_ACTIONS:
        if verb == "Sleep":
            return Sleep(), None
        return Monitor(session=session, agent="Blue"), None

    if len(parts) < 2:
        return None, f"{verb} requires a hostname"
    host = parts[1].strip(",.")
    valid_hosts = list(action_space["hostname"].keys())
    if host not in valid_hosts:
        return None, f"unknown host {host!r} (valid: {', '.join(valid_hosts[:4])}...)"

    cls = HOST_ACTIONS[verb]
    return cls(session=session, agent="Blue", hostname=host), None
