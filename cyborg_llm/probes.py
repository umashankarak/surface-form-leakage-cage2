"""
Name-agnostic scripted probe agents.

WHY THIS FILE EXISTS
--------------------
Gate 1's design assumed CybORG's scripted agents are name-blind, so that a
scripted agent scoring equally on L0 and a twin would prove the twin is
difficulty-matched.

That assumption is FALSE. The shipped agents hardcode canonical hostnames:

  BlueReactAgent.py:19,51   `host_name != 'User0'`
                            -> an explicit literal exception for the red entry
                               point. On a renamed twin the guard stops firing,
                               so the agent starts wasting actions restoring the
                               entry host. Measured cost: ~5 reward points.
                               That is a STRING, not a difficulty difference.

  B_line.py:38,39           `observation['User0']['Interface'][0]['IP Address']`
                            -> indexes the observation dict with a literal key.
                               KeyError on any renamed scenario.
  B_line.py:60              `[x for x in observation if 'Enterprise' in x][0]`
  B_line.py:96-111          hardcoded 'Op_Server0' for discover/exploit/impact
  GreenAgent.py:16          hardcoded 'User0'

Worth noting for the paper: the shipped "expert" agents encode
"User0 is the entry point, Op_Server0 is the goal" as literal constants. They
are the retrieval baseline, made explicit in source code. Rename the network and
they break -- which is precisely the behaviour the study hypothesises for LLMs.

THE FIX
-------
Probe agents that take the entry point and goal FROM THE SCENARIO instead of
from a hardcoded string. Identical policy, zero name dependence. These can then
serve as the difficulty probe Gate 1 needs: a genuinely name-blind agent that
scores equally on L0 and twin proves the twin is equally hard.
"""
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.SimpleAgents.B_line import B_lineAgent
from CybORG.Shared.Actions import Monitor, Remove, Restore, Sleep


class _AgnosticReactBase(BaseAgent):
    """
    Policy identical to BlueReactRemove/RestoreAgent, but the entry-point
    exception is supplied by the caller rather than baked in as 'User0'.
    """
    remediate = Remove

    def __init__(self, ignore_host=None):
        self.host_list = []
        self.last_action = None
        self.ignore_host = ignore_host      # <- was the literal 'User0'
        super().__init__()

    def train(self, results):
        pass

    def end_episode(self):
        self.host_list = []
        self.last_action = None

    def set_initial_values(self, action_space, observation):
        pass

    def get_action(self, observation, action_space):
        if self.last_action is not None and self.last_action == "Monitor":
            for host_name, host_info in [
                (v["System info"]["Hostname"], v)
                for k, v in observation.items()
                if k != "success" and isinstance(v, dict) and "System info" in v
            ]:
                if (host_name not in self.host_list
                        and host_name != self.ignore_host
                        and "Processes" in host_info
                        and len([i for i in host_info["Processes"] if "PID" in i]) > 0):
                    self.host_list.append(host_name)

        session = list(action_space["session"].keys())[0]
        if len(self.host_list) == 0:
            self.last_action = "Monitor"
            return Monitor(agent="Blue", session=session)
        self.last_action = self.remediate.__name__
        return self.remediate(hostname=self.host_list.pop(0), agent="Blue", session=session)


class AgnosticReactRemoveAgent(_AgnosticReactBase):
    remediate = Remove


class AgnosticReactRestoreAgent(_AgnosticReactBase):
    remediate = Restore


def make_probe(scenario, restore=True):
    """
    Build a name-blind BLUE probe wired to THIS scenario's actual entry point.

    scenario: parsed scenario dict (so the entry host is read, not assumed).
    """
    from .twins import red_start
    cls = AgnosticReactRestoreAgent if restore else AgnosticReactRemoveAgent
    return cls(ignore_host=red_start(scenario))


class AgnosticMeanderAgent(RedMeanderAgent):
    """
    RedMeanderAgent with its goal supplied instead of hardcoded.

    The shipped agent contains (Meander.py:31-33):

        if 'Op_Server0' in self.escalated_hosts:
            return Impact(agent='Red', hostname='Op_Server0', session=session)

    i.e. the ATTACKER's win condition is a string literal. On a renamed twin
    that literal never matches, so red can never Impact the crown jewel and the
    twin becomes silently easier for blue.

    Measured (n=10, name-blind blue probe, L0 vs L1_alt_a):

        30 steps    L0  -12.13   L1  -12.22   delta   -0.09   p=0.94
        60 steps    L0  -65.03   L1  -47.02   delta  +18.01
       100 steps    L0 -282.24   L1 -137.93   delta +144.31   <- twin 2x easier

    The 30-step cap HIDES this. The Gate 1 pass at 30 steps is therefore
    accidental, and would silently break the moment anyone lengthened an
    episode. Parameterising the goal removes the confound at every horizon.
    """

    def __init__(self, goal_host="Op_Server0", op_hosts=None, ent_hosts=None):
        super().__init__()
        self.goal_host = goal_host
        # Tier membership, supplied from the scenario instead of inferred from
        # substrings in the hostname. See _process_failed_ip below.
        self.op_hosts = set(op_hosts or ())
        self.ent_hosts = set(ent_hosts or ())

    def _process_failed_ip(self):
        """
        Faithful copy of RedMeanderAgent._process_failed_ip with ONE change:
        tier membership comes from the scenario, not from substring matching.

        The shipped agent contains (Meander.py:97-98):

            hosts_of_type = lambda y: [x for x in self.escalated_hosts if y in x]
            if len(hosts_of_type('Op')) > 0:   ...
            elif len(hosts_of_type('Ent')) > 0: ...

        i.e. red decides which tier a host belongs to by asking whether its NAME
        contains the substring 'Op' or 'Ent'. On a renamed twin nothing matches
        ('PaymentDB_Primary' has no 'Op'; 'AuthGateway' has no 'Ent'), so red's
        entire failure-recovery path goes dead: it never backs off, keeps a
        stale escalated_hosts list, then skips those hosts forever via
        `if hostname in self.escalated_hosts: continue`. Red gets stuck in the
        entry subnet and the twin becomes markedly easier for blue.

        Measured (N=100, 100-step episodes, L0 vs L1_alt_a):
            before fix:  L0 -238.02   L1 -141.89   delta +96.13   p<0.001

        This is also a substring trap in the other direction: 'SecOpsMonitor'
        *does* contain 'Op', so a plausible rename can accidentally re-activate
        the branch for the wrong host.
        """
        self.exploited_ips.remove(self.last_ip)

        def hosts_of_tier(tier):
            return [x for x in self.escalated_hosts if x in tier]

        for tier in (self.op_hosts, self.ent_hosts):
            hits = hosts_of_tier(tier)
            if hits:
                for host in hits:
                    self.escalated_hosts.remove(host)
                    ip = self.host_ip_map[host]
                    self.exploited_ips.remove(ip)
                return

    def get_action(self, observation, action_space):
        """
        Faithful copy of RedMeanderAgent.get_action with ONE change:
        the literal 'Op_Server0' becomes self.goal_host.

        Copied rather than delegated because the parent calls
        _process_success() itself -- calling super() after our own
        _process_success() double-processes the observation and silently
        corrupts the agent's internal state (measured: red stops attacking
        entirely, reward flat 0.00 at every horizon).
        """
        import random as _random

        from CybORG.Shared.Actions import (
            DiscoverNetworkServices, DiscoverRemoteSystems,
            ExploitRemoteService, Impact, PrivilegeEscalate,
        )

        self._process_success(observation)
        session = list(action_space["session"].keys())[0]

        # always impact if able  (was: hardcoded 'Op_Server0')
        if self.goal_host in self.escalated_hosts:
            self.last_host = self.goal_host
            return Impact(agent="Red", hostname=self.goal_host, session=session)

        for subnet in action_space["subnet"]:
            if not action_space["subnet"][subnet] or subnet in self.scanned_subnets:
                continue
            self.scanned_subnets.append(subnet)
            return DiscoverRemoteSystems(subnet=subnet, agent="Red", session=session)

        addresses = [i for i in action_space["ip_address"]]
        _random.shuffle(addresses)
        for address in addresses:
            if not action_space["ip_address"][address] or address in self.scanned_ips:
                continue
            self.scanned_ips.append(address)
            return DiscoverNetworkServices(ip_address=address, agent="Red", session=session)

        hostnames = [x for x in action_space["hostname"].keys()]
        _random.shuffle(hostnames)
        for hostname in hostnames:
            if not action_space["hostname"][hostname]:
                continue
            if hostname in self.escalated_hosts:
                continue
            if hostname in self.host_ip_map and self.host_ip_map[hostname] not in self.exploited_ips:
                continue
            self.escalated_hosts.append(hostname)
            self.last_host = hostname
            return PrivilegeEscalate(hostname=hostname, agent="Red", session=session)

        for address in addresses:
            if not action_space["ip_address"][address] or address in self.exploited_ips:
                continue
            self.exploited_ips.append(address)
            self.last_ip = address
            return ExploitRemoteService(ip_address=address, agent="Red", session=session)

        return Sleep()


def _blue_home(scenario):
    """
    The defender's own box, derived structurally (it hosts the Velociraptor
    SERVER session). Needed because the shipped agents' substring test
    `'Enterprise' in x` matches Enterprise0-2 but NOT 'Defender' -- even though
    Defender sits in that same subnet. A tier set derived purely from subnet
    membership would wrongly include it and diverge from the original policy.
    """
    for sess in scenario["Agents"]["Blue"].get("starting_sessions", []):
        if sess.get("type") == "VelociraptorServer":
            return sess.get("hostname")
    return None


def _tiers(scenario):
    """
    Structural tier derivation -- never from names.

      op tier    : the subnet holding the crown jewel
      entry tier : the subnet holding red's starting session
      ent tier   : the remaining (pivot) subnet, minus the defender's own box
    """
    from .twins import crown_jewel, red_start

    goal, start = crown_jewel(scenario), red_start(scenario)
    host2sub = {}
    for sub, sv in scenario["Subnets"].items():
        for h in sv["Hosts"]:
            host2sub[h] = sub

    op_sub, entry_sub = host2sub[goal], host2sub[start]
    home = _blue_home(scenario)

    op_hosts = set(scenario["Subnets"][op_sub]["Hosts"])
    ent_hosts = set()
    for s in scenario["Subnets"]:
        if s in (op_sub, entry_sub):
            continue
        ent_hosts |= set(scenario["Subnets"][s]["Hosts"])
    ent_hosts.discard(home)          # 'Enterprise' never matched 'Defender'
    return goal, start, op_hosts, ent_hosts


class AgnosticBlineAgent(B_lineAgent):
    """
    B_lineAgent with its five hardcoded names parameterised.

    B_line beelines for the crown jewel instead of wandering, which is why it
    matters here: with RedMeanderAgent, red reaches Op_Server0 in 0/60 episodes
    at the 30-step horizon, so the single most retrievable fact in CAGE-2
    ("Op_Server0 is the crown jewel") is INERT and renaming it cannot possibly
    change anything. B_line reaches it in ~12% of 30-step episodes, putting a
    live signal inside the horizon where the twins are already verified matched.

    The shipped agent cannot run on a renamed twin at all -- it does
    `observation['User0'][...]` and raises KeyError. Literals replaced:

        action 0   observation['User0']            -> self.entry_host
        action 4   [x for x in obs if 'Enterprise' in x][0]
                                                   -> first obs key in self.ent_hosts
        action 11  observation['Op_Server0']       -> self.goal_host
        action 13  PrivilegeEscalate('Op_Server0') -> self.goal_host
        action 14  Impact('Op_Server0')            -> self.goal_host

    Policy, jump table and RNG consumption are otherwise byte-for-byte identical.
    """

    def __init__(self, entry_host="User0", ent_hosts=(), goal_host="Op_Server0"):
        super().__init__()
        self.entry_host = entry_host
        self.ent_hosts = set(ent_hosts)
        self.goal_host = goal_host

    def get_action(self, observation, action_space):
        import random as _random

        from CybORG.Shared.Actions import (
            DiscoverNetworkServices, DiscoverRemoteSystems,
            ExploitRemoteService, Impact, PrivilegeEscalate,
        )

        session = 0
        while True:
            if observation["success"] == True:
                self.action += 1 if self.action < 14 else 0
            else:
                self.action = self.jumps[self.action]

            if self.action in self.action_history:
                action = self.action_history[self.action]

            elif self.action == 0:
                self.initial_ip = observation[self.entry_host]["Interface"][0]["IP Address"]
                self.last_subnet = observation[self.entry_host]["Interface"][0]["Subnet"]
                action = DiscoverRemoteSystems(session=session, agent="Red", subnet=self.last_subnet)

            elif self.action == 1:
                hosts = [v for k, v in observation.items() if k != "success"]
                get_ip = lambda x: x["Interface"][0]["IP Address"]
                interfaces = [get_ip(x) for x in hosts if get_ip(x) != self.initial_ip]
                self.last_ip_address = _random.choice(interfaces)
                action = DiscoverNetworkServices(session=session, agent="Red",
                                                 ip_address=self.last_ip_address)

            elif self.action == 2:
                action = ExploitRemoteService(session=session, agent="Red",
                                              ip_address=self.last_ip_address)

            elif self.action == 3:
                hostname = [v for k, v in observation.items()
                            if k != "success" and "System info" in v][0]["System info"]["Hostname"]
                action = PrivilegeEscalate(agent="Red", hostname=hostname, session=session)

            elif self.action == 4:
                self.enterprise_host = [x for x in observation if x in self.ent_hosts][0]
                self.last_ip_address = observation[self.enterprise_host]["Interface"][0]["IP Address"]
                action = DiscoverNetworkServices(session=session, agent="Red",
                                                 ip_address=self.last_ip_address)

            elif self.action == 5:
                self.target_ip_address = [v for k, v in observation.items()
                                          if k != "success"][0]["Interface"][0]["IP Address"]
                action = ExploitRemoteService(session=session, agent="Red",
                                              ip_address=self.target_ip_address)

            elif self.action == 6:
                hostname = [v for k, v in observation.items()
                            if k != "success" and "System info" in v][0]["System info"]["Hostname"]
                action = PrivilegeEscalate(agent="Red", hostname=hostname, session=session)

            elif self.action == 7:
                self.last_subnet = observation[self.enterprise_host]["Interface"][0]["Subnet"]
                action = DiscoverRemoteSystems(subnet=self.last_subnet, agent="Red", session=session)

            elif self.action == 8:
                self.target_ip_address = [v for k, v in observation.items()
                                          if k != "success"][2]["Interface"][0]["IP Address"]
                action = DiscoverNetworkServices(session=session, agent="Red",
                                                 ip_address=self.target_ip_address)

            elif self.action == 9:
                self.target_ip_address = [v for k, v in observation.items()
                                          if k != "success"][0]["Interface"][0]["IP Address"]
                action = ExploitRemoteService(session=session, agent="Red",
                                              ip_address=self.target_ip_address)

            elif self.action == 10:
                hostname = [v for k, v in observation.items()
                            if k != "success" and "System info" in v][0]["System info"]["Hostname"]
                action = PrivilegeEscalate(agent="Red", hostname=hostname, session=session)

            elif self.action == 11:
                action = DiscoverNetworkServices(
                    session=session, agent="Red",
                    ip_address=observation[self.goal_host]["Interface"][0]["IP Address"])

            elif self.action == 12:
                info = [v for k, v in observation.items() if k != "success"]
                if len(info) > 0:
                    action = ExploitRemoteService(agent="Red", session=session,
                                                  ip_address=info[0]["Interface"][0]["IP Address"])
                else:
                    self.action = 0
                    continue

            elif self.action == 13:
                action = PrivilegeEscalate(agent="Red", hostname=self.goal_host, session=session)

            elif self.action == 14:
                action = Impact(agent="Red", session=session, hostname=self.goal_host)

            if self.action not in self.action_history:
                self.action_history[self.action] = action
            return action


def patch_simulator_enterprise_check(scenario):
    """
    Patch CybORG's SIMULATOR-level access control to be name-blind.

    THE ROOT CAUSE OF EVERY TWIN DIVERGENCE FOUND SO FAR.

    CybORG/Shared/Actions/ConcreteActions/ConcreteAction.py:75-82

        def check_for_enterprise_sessions(self):
            permission = False
            for session_id in self.state.sessions[self.agent]:
                session = self.state.sessions[self.agent][session_id]
                if 'Enterprise' in session.host:      # <-- SUBSTRING MATCH
                    permission = True
            return permission

    called from check_routable() at line 40:

        elif 'None' in to_subnet.nacls[from_subnet.name]['in']:
            # If you don't have access to Enteprise network, you can't act on Operational Host
            # TODO refactor this hacky fix
            permission = self.check_for_enterprise_sessions()

    Scenario2's defining structural constraint -- "red must pivot through the
    Enterprise subnet to reach Operational" -- is implemented as a substring
    test on hostnames, inside the ENVIRONMENT, flagged by CAGE-2's own authors
    as a hacky fix. It is not in any agent; no amount of fixing agents helps.

    Consequence for renamed twins: 'Enterprise' does not appear in
    'AuthGateway', so red is PERMANENTLY LOCKED OUT of the Operational subnet
    and can never reach the crown jewel. Measured, before this patch:

        Meander,  30 steps : reaches crown jewel   0/60  (L0 also 0 -- too slow)
        Meander, 100 steps : L0 -238.02  L1 -141.89   (+96, p<0.001)
        B_line,   30 steps : L0 reaches goal 16.5%, L1 0.0%   (+11.65, p<0.001)

    That single line explains all of it: every configuration where the
    crown-jewel signal was live, the twin diverged.

    This patch restores the INTENDED semantics (pivot through the middle tier)
    using structural tier membership instead of spelling. It is global and
    per-process, so it must be re-applied whenever the scenario changes -- pass
    the scenario you are about to run.
    """
    from CybORG.Shared.Actions.ConcreteActions.ConcreteAction import ConcreteAction

    _, _, _op, ent_hosts = _tiers(scenario)

    def check_for_enterprise_sessions(self):
        for session_id in self.state.sessions[self.agent]:
            if self.state.sessions[self.agent][session_id].host in ent_hosts:
                return True
        return False

    ConcreteAction.check_for_enterprise_sessions = check_for_enterprise_sessions
    return ent_hosts


def make_bline_probe_class(scenario):
    """Name-blind B_line probe wired to THIS scenario's structure. Returns a CLASS."""
    goal, start, _op, ent = _tiers(scenario)

    def __init__(self):
        AgnosticBlineAgent.__init__(self, entry_host=start, ent_hosts=ent, goal_host=goal)

    return type(f"AgnosticBline_{goal}", (AgnosticBlineAgent,), {"__init__": __init__})


def make_red_probe_class(scenario):
    """
    Name-blind RED probe (Meander) wired to THIS scenario's structure.

    Returns a CLASS, not an instance: CybORG instantiates the agent itself
    (`agents={'Red': SomeAgentClass}`) with no arguments, so everything must be
    closed over at class-creation time.
    """
    goal, _start, op_hosts, ent_hosts = _tiers(scenario)

    def __init__(self):
        AgnosticMeanderAgent.__init__(self, goal_host=goal,
                                      op_hosts=op_hosts, ent_hosts=ent_hosts)

    return type(f"AgnosticMeander_{goal}", (AgnosticMeanderAgent,), {"__init__": __init__})
