# environments/environment.py
import numpy as np
from abc import abstractmethod
from typing import Optional, List, Dict, Any


class Environment:
    """
    Simulator-agnostic base class for RL environments.

    Removes all traci/SUMO coupling from the original. Subclasses implement
    the abstract methods for their specific simulator backend.

    What's retained from the original:
    - _make_state / _add_to_state / _process_state logic (unchanged)
    - _process_action binary encoding for single-agent mode (unchanged)
    - Abstract interface: get_state, get_reward, _execute_action, _open_connection

    What's removed:
    - traci.main._connections management
    - self.connection (traci connection object)
    - get_intersections / get_intersection_neighborhoods (SUMO net scrape)
    - _generate_configfile / _generate_routefile / _generate_addfile
    - step() default impl (too simulator-specific to share meaningfully)
    """

    def __init__(
        self,
        constants: dict,
        device,
        agent_ID,
        eval_agent: bool,
        scenario_path,
        vis: bool = False,
    ):
        self.constants = constants
        self.device = device
        self.agent_ID = agent_ID
        self.eval_agent = eval_agent
        self.vis = vis
        self.scenario_path = scenario_path

        self.agent_type = constants['agent']['agent_type']
        self.single_agent = constants['agent']['single_agent']

        # Subclasses are responsible for populating these after __init__
        # since intersection discovery is simulator-specific.
        # Set here as empty so _make_state/_add_to_state don't fail on access
        # before the subclass __init__ runs its discovery.
        self.phases: Optional[Any] = None
        self.intersections: List[str] = []
        self.intersections_index: Dict[str, int] = {}
        self.neighborhoods: Dict[str, List[str]] = {}
        self.max_num_neighbors: int = 1

        # Derived state/action sizes — set by subclass after intersection
        # discovery, used by run_agent_parallel._make_nn()
        self.state_size: Optional[int] = None
        self.action_size: Optional[int] = None

    # ---- State construction (unchanged from original) -----------------------

    def _make_state(self):
        if self.agent_type == 'rule':
            return {}
        if self.single_agent:
            return []
        if self.constants['multiagent']['state_interpolation'] == 0:
            return [[] for _ in range(len(self.intersections))]
        return {}  # dict when state interpolation is active

    def _add_to_state(self, state, value, key, intersection):
        if self.agent_type == 'rule':
            if intersection:
                if intersection not in state:
                    state[intersection] = {}
                state[intersection][key] = value
            else:
                state[key] = value
        else:
            if self.single_agent:
                if isinstance(value, list):
                    state.extend(value)
                else:
                    state.append(value)
            else:
                if self.constants['multiagent']['state_interpolation'] == 0:
                    if isinstance(value, list):
                        state[self.intersections_index[intersection]].extend(value)
                    else:
                        state[self.intersections_index[intersection]].append(value)
                else:
                    if intersection not in state:
                        state[intersection] = []
                    if isinstance(value, list):
                        state[intersection].extend(value)
                    else:
                        state[intersection].append(value)

    def _process_state(self, state):
        if self.agent_type != 'rule':
            if self.single_agent:
                return np.expand_dims(np.array(state), axis=0)
            return np.array(state)
        return state

    # ---- Action processing (unchanged from original) ------------------------

    def _process_action(self, a):
        """
        Converts PPO output to per-intersection action dict.
        Single-agent: binary-encodes the integer action across intersections.
        Multi-agent: maps action array directly to intersection IDs.
        """
        action = a.copy()
        if self.single_agent and self.agent_type != 'rule':
            # Encode single integer as binary string across all intersections
            action = '{0:0b}'.format(int(a[0]))
            action = action.zfill(len(self.intersections))
            action = [int(c) for c in action]
        return {
            intersection: action[i]
            for i, intersection in enumerate(self.intersections)
        }

    # ---- Connection lifecycle -----------------------------------------------

    @abstractmethod
    def _open_connection(self):
        """
        Initializes or resets the simulator backend for a new episode.
        For A/B Street: POST /sim/load with scenario path.
        For SUMO: start traci connection.
        """
        raise NotImplementedError

    def _close_connection(self):
        """
        Optional teardown. Default is a no-op — stateless HTTP simulators
        (A/B Street) don't need explicit teardown. SUMO subclasses override.
        """
        pass

    def reset(self):
        """
        Resets the environment for a new episode.
        Calls _open_connection() then returns initial state.
        Subclasses can override if reset needs more than just reopening.
        """
        self._open_connection()
        return self.get_state()

    # ---- Abstract simulator interface ---------------------------------------

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, get_global: bool):
        raise NotImplementedError

    @abstractmethod
    def _execute_action(self, action: dict):
        raise NotImplementedError

    @abstractmethod
    def step(self, a, ep_step: int, get_global_reward: bool, def_agent: bool = False):
        """
        Step is abstract here because the done condition is fundamentally
        different across simulators:
        - SUMO: done when getMinExpectedNumber() <= 0 or max_ep_steps reached
        - A/B Street: done when max_ep_steps reached (episode-scoped sim)
        Subclasses implement their own done logic rather than sharing a broken default.
        """
        raise NotImplementedError
