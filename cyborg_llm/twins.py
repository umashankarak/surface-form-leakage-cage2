"""
Twin generator -- Phase 1, Gate 1.

Produces difficulty-matched variants of a CybORG scenario that differ ONLY in
retrievability.

THE CENTRAL DESIGN CONSTRAINT
-----------------------------
The LLM sees exactly one thing that could be retrieved: hostnames. Image names
(OP_Server, Gateway) never reach the observation or the action menu -- verified
by inspection of serialize_obs() and action_menu().

So the manipulation is hostnames. But naive renaming is a CONFOUND, not a
control:

    Op_Server0 -> Node_7     removes information a real defender legitimately
                             has (which server is critical). Any drop then
                             means "we hid the answer" -- trivial, and nothing
                             to do with memorization.

    Op_Server0 -> PaymentDB_Primary
                             conveys the SAME semantics (this is the crown
                             jewel) in words that appear in no CAGE-2 write-up.
                             A model reasoning from meaning is unaffected. A
                             model recalling "protect Op_Server0" gets nothing.

That distinction is the whole experiment. Schemes below are built to preserve
semantic content while destroying lexical match.

LEVELS
------
  L0  canonical      the scenario as published (User0 ... Op_Server0)
  L1  renamed        semantically equivalent, lexically novel
  L2  L1 + permuted  red's entry point moves off User0. Every CAGE-2 write-up
                     states red starts on User0; that is a retrievable fact
                     independent of naming.
  L3  procedural     (deferred -- L1/L2 are sufficient for Gate 1)

CONTROL ARM
-----------
  L1_neutral         Node_1 ... Node_13. Deliberately strips semantics. This is
                     the information-removal condition, and it exists to
                     separate "hid the answer" from "broke the recall". Expect
                     a drop here for reasons that are NOT retrieval.
"""
import copy
import random

import yaml

# ---------------------------------------------------------------- schemes

# Semantics-preserving. Each canonical host maps to a name carrying equivalent
# meaning (role, tier, criticality) in vocabulary absent from CAGE-2 write-ups.
SCHEME_ALT_A = {
    "Defender":    "SecOpsMonitor",       # blue's own box
    "Enterprise0": "AuthGateway",         # gateway tier
    "Enterprise1": "AppServer_Alpha",     # internal tier
    "Enterprise2": "AppServer_Beta",
    "Op_Host0":    "PlantNode_01",        # operational tier, low value
    "Op_Host1":    "PlantNode_02",
    "Op_Host2":    "PlantNode_03",
    "Op_Server0":  "PaymentDB_Primary",   # crown jewel: critical + high avail
    "User0":       "Workstation_Sales_A",
    "User1":       "Workstation_Sales_B",
    "User2":       "Workstation_Sales_C",
    "User3":       "Workstation_Eng_A",
    "User4":       "Workstation_Eng_B",
}
SUBNETS_ALT_A = {"User": "Corp", "Enterprise": "Midtier", "Operational": "Restricted"}

SCHEME_ALT_B = {
    "Defender":    "AuditHost",
    "Enterprise0": "EdgeRelay",
    "Enterprise1": "RecordsSvc_1",
    "Enterprise2": "RecordsSvc_2",
    "Op_Host0":    "TurbineCtl_A",
    "Op_Host1":    "TurbineCtl_B",
    "Op_Host2":    "TurbineCtl_C",
    "Op_Server0":  "LedgerStore_Main",
    "User0":       "Desk_Finance_1",
    "User1":       "Desk_Finance_2",
    "User2":       "Desk_Finance_3",
    "User3":       "Desk_Legal_1",
    "User4":       "Desk_Legal_2",
}
SUBNETS_ALT_B = {"User": "Office", "Enterprise": "Services", "Operational": "Controlled"}

# CONTROL ARM ONLY -- strips semantics. Not a retrieval manipulation.
SCHEME_NEUTRAL = {h: f"Node_{i+1:02d}" for i, h in enumerate([
    "Defender", "Enterprise0", "Enterprise1", "Enterprise2",
    "Op_Host0", "Op_Host1", "Op_Host2", "Op_Server0",
    "User0", "User1", "User2", "User3", "User4"])}
SUBNETS_NEUTRAL = {"User": "Zone_A", "Enterprise": "Zone_B", "Operational": "Zone_C"}

SCHEMES = {
    "alt_a":   (SCHEME_ALT_A, SUBNETS_ALT_A),
    "alt_b":   (SCHEME_ALT_B, SUBNETS_ALT_B),
    "neutral": (SCHEME_NEUTRAL, SUBNETS_NEUTRAL),
}


def load_scenario(path):
    with open(path) as f:
        return yaml.safe_load(f)


def write_scenario(scenario, path):
    with open(path, "w") as f:
        yaml.safe_dump(scenario, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------- transform

def _remap(obj, host_map, subnet_map):
    """
    Recursive structural rename.

    Structural, NOT string-replace. A naive text substitution of 'User' would
    corrupt 'User0' -> '<new>0', and 'User info'/'Username' would be mangled
    too. Reference map (verified by inspection of Scenario2.yaml):

      hostnames appear as:
        Agents.{Blue,Green,Red}.INT.Hosts.<host>          (KEY)
        Agents.{Blue,Green,Red}.starting_sessions[].hostname  (VALUE)
        Hosts.<host>                                      (KEY)
        Hosts.<host>.info.<host>                          (KEY, nested)
        Subnets.<subnet>.Hosts[]                          (VALUE)

      subnet names appear as:
        Agents.*.AllowedSubnets[]                         (VALUE)
        Subnets.<subnet>                                  (KEY)
        Subnets.<other>.NACLs.<subnet>                    (KEY)
    """
    both = {**host_map, **subnet_map}
    if isinstance(obj, dict):
        return {both.get(k, k): _remap(v, host_map, subnet_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_remap(v, host_map, subnet_map) for v in obj]
    if isinstance(obj, str):
        return both.get(obj, obj)
    return obj


def make_twin(scenario, level="L1", scheme="alt_a", seed=0):
    """
    Returns (twin_scenario, metadata).

    L1: rename only. Structure untouched -> isomorphic by construction.
    L2: rename + move red's entry point to a different host in the same subnet.
        Same subnet => same NACL position => same path length to the goal, so
        difficulty is preserved. What changes is the retrievable fact that
        "red starts on User0", which every write-up states.
    """
    host_map, subnet_map = SCHEMES[scheme]
    twin = _remap(copy.deepcopy(scenario), host_map, subnet_map)
    meta = {"level": level, "scheme": scheme, "seed": seed,
            "host_map": dict(host_map), "subnet_map": dict(subnet_map),
            "red_start_original": "User0", "red_start_twin": host_map["User0"]}

    if level == "L2":
        rng = random.Random(seed)
        user_subnet = subnet_map["User"]
        current = host_map["User0"]
        cur_spec = twin["Hosts"][current]

        # The permutation must preserve difficulty. Moving red's entry point to
        # a host with a DIFFERENT IMAGE changes the services available at entry,
        # which is a real difficulty change, not a surface one.
        #
        # Verified in Scenario2: the User subnet holds windows_user_host1 x2
        # (User0, User1), windows_user_host2, linux_user_host1, linux_user_host2.
        # Only User1 shares User0's image. Gate 1 caught this: an unconstrained
        # swap to windows_user_host2 moved the scripted probe from -11.95 to
        # -3.00 with sd=0.00 -- red was effectively neutered.
        #
        # So: candidates are same-subnet AND same-image AND same-values only.
        def _same_class(h):
            s = twin["Hosts"][h]
            return (h != current
                    and s.get("image") == cur_spec.get("image")
                    and str(s.get("AvailabilityValue", "-")) == str(cur_spec.get("AvailabilityValue", "-")))

        candidates = [h for h in twin["Subnets"][user_subnet]["Hosts"] if _same_class(h)]
        if not candidates:
            raise ValueError(
                f"L2 impossible: no host in subnet {user_subnet!r} shares "
                f"{current!r}'s image ({cur_spec.get('image')!r}). An "
                f"unconstrained swap would change difficulty. Use L1, or extend "
                f"the scenario with a same-image sibling."
            )
        new_start = rng.choice(candidates)

        for sess in twin["Agents"]["Red"]["starting_sessions"]:
            if sess.get("hostname") == current:
                sess["hostname"] = new_start
        meta["red_start_twin"] = new_start
        meta["permutation"] = f"red entry {current} -> {new_start}"
        meta["permutation_candidates"] = candidates
        meta["image_preserved"] = cur_spec.get("image")

    return twin, meta


def host_graph(scenario):
    """
    Scenario -> networkx DiGraph for the isomorphism check.

    MUST be directed. NACLs are asymmetric: the Operational subnet blocks
    inbound from User but may reach out to it. That asymmetry is the entire
    structural point of the scenario -- it forces red to pivot User -> Enterprise
    -> Operational rather than going direct. An undirected graph silently
    re-adds the blocked edge from the other side and collapses to K13, at which
    point the isomorphism check is vacuous (every relabeling of a complete
    graph is isomorphic).

    Nodes = hosts, attributed with subnet size, image, and reward values -- the
    things that determine difficulty. Node NAMES are deliberately not matched;
    an isomorphism on names would prove nothing.
    """
    import networkx as nx
    g = nx.DiGraph()
    host2subnet = {}
    for sub, sv in scenario["Subnets"].items():
        for h in sv["Hosts"]:
            host2subnet[h] = sub

    for h, hv in scenario["Hosts"].items():
        g.add_node(h,
                   image=hv.get("image", "?"),
                   conf=str(hv.get("ConfidentialityValue", "-")),
                   avail=str(hv.get("AvailabilityValue", "-")),
                   subnet_size=len(scenario["Subnets"][host2subnet[h]]["Hosts"]))

    # intra-subnet: fully reachable, both directions
    for sub, sv in scenario["Subnets"].items():
        for a in sv["Hosts"]:
            for b in sv["Hosts"]:
                if a != b:
                    g.add_edge(a, b)

    # inter-subnet: directed, gated by the destination's inbound NACL
    subs = list(scenario["Subnets"])
    for src in subs:
        for dst in subs:
            if src == dst:
                continue
            nacl = scenario["Subnets"][dst].get("NACLs", {})
            rule = nacl.get(src) or nacl.get("all", {})
            if rule.get("in") == "None":
                continue                       # dst refuses inbound from src
            for ha in scenario["Subnets"][src]["Hosts"]:
                for hb in scenario["Subnets"][dst]["Hosts"]:
                    g.add_edge(ha, hb)
    return g


def path_profile(scenario):
    """
    The difficulty fingerprint: shortest path red-start -> crown jewel, plus
    branching. If a twin matches on these, it is structurally as hard.
    """
    import networkx as nx
    g = host_graph(scenario)
    src, dst = red_start(scenario), crown_jewel(scenario)
    try:
        d = nx.shortest_path_length(g, src, dst)
    except nx.NetworkXNoPath:
        d = None
    out_deg = [g.out_degree(n) for n in g.nodes]
    return {
        "n_hosts": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "hops_start_to_goal": d,
        "mean_out_degree": round(sum(out_deg) / len(out_deg), 3),
        "max_out_degree": max(out_deg),
        "reachable_from_start": len(nx.descendants(g, src)),
    }


def red_start(scenario):
    return scenario["Agents"]["Red"]["starting_sessions"][0]["hostname"]


def crown_jewel(scenario):
    """Highest-availability host -- the thing red is trying to reach."""
    best, best_score = None, -1
    order = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    for h, hv in scenario["Hosts"].items():
        s = order.get(str(hv.get("AvailabilityValue")), 0)
        if s > best_score:
            best, best_score = h, s
    return best
