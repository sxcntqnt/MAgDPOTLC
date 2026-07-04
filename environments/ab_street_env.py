# environments/ab_street_env.py
"""
A/B Street RL environment.
Speaks only to ABStreetClient — zero direct HTTP calls here.
All RL logic (state construction, reward shaping, action execution) lives here.
"""
import numpy as np
from collections import OrderedDict
from environments.environment import Environment
from environments.abstreet_client import ABStreetClient

PER_AGENT_STATE_SIZE = 6
GLOBAL_STATE_SIZE = 1
ACTION_SIZE = 2


class ABStreetIntersectionsEnv(Environment):

    def __init__(
        self,
        constants,
        device,
        worker_index: int,
        worker_name: str,
        eval_agent,
        scenario_path,
        vis=False,
    ):
        super().__init__(
            constants,
            device,
            worker_name,
            eval_agent,
            scenario_path,
            vis,
        )

        base_port = constants["parallel"]["base_port"]

        self.client = ABStreetClient(
            host="http://localhost",
            port=base_port + worker_index,
        )


        self.env_name = (
            f"{constants['environment']['shape'][0]}_"
            f"{constants['environment']['shape'][1]}_intersections"
        )

        self.sim_time_seconds = 0
        self.step_interval_seconds = 30

        self.intersections = self._discover_intersections()
        
        if self.single_agent:
            self.state_size = PER_AGENT_STATE_SIZE * len(self.intersections) + GLOBAL_STATE_SIZE
        else:
           interp = constants['multiagent']['state_interpolation']
           neighbor_factor = self.max_num_neighbors if interp != 0 else 1
           self.state_size = (PER_AGENT_STATE_SIZE + GLOBAL_STATE_SIZE) * neighbor_factor

        self.action_size = ACTION_SIZE

        self.intersections_index = {
            intersection: i
            for i, intersection in enumerate(self.intersections)
        }
        self.distances = self._get_intersection_distances()

    # ---- Setup --------------------------------------------------------------

    def _discover_intersections(self):
        """
        Discovers live intersection IDs from the running server.
        Falls back to placeholder IDs if the server isn't up yet at init time.
        """
        try:
            response = self.client.get_all_signal_states()
            return list(response.keys())
        except Exception:
            n = (
                self.constants['environment']['shape'][0]
                * self.constants['environment']['shape'][1]
            )
            return [f"intersection_{i}" for i in range(n)]

    def _get_intersection_distances(self):
        """
        Placeholder distance matrix. Replace with real geometry from
        client.get_intersection_geometry() once map metadata is available.
        """
        distances = {}
        for i in self.intersections:
            distances[i] = {
                j: (0.0 if i == j else 1.0)
                for j in self.intersections
            }
        return distances

    def _open_connection(self):
        self.client.sim_reset()
        self.sim_time_seconds = 0

    # ---- State --------------------------------------------------------------

    def _get_sim_step(self, normalize: bool) -> float:
        sim_step = float(self.sim_time_seconds)
        if normalize:
            sim_step /= (self.constants['episode']['max_ep_steps'] / 10.0)
        return sim_step

    def get_state(self):
        state = self._make_state()
        normalize = self.agent_type != 'rule'
        sim_step = self._get_sim_step(normalize)

        all_signals = self.client.get_all_signal_states()

        for intersection in self.intersections:
            signal_data = all_signals.get(intersection, {"waiting": [], "stage": 0})

            waiting_count = len(signal_data.get("waiting", []))
            jam_length = [waiting_count] * 4
            self._add_to_state(state, jam_length, key='jam_length', intersection=intersection)

            curr_phase = signal_data.get("stage", 0)
            self._add_to_state(state, curr_phase, key='curr_phase', intersection=intersection)

            elapsed_phase_time = 0.0
            self._add_to_state(state, elapsed_phase_time, key='elapsed_phase_time', intersection=intersection)

            if not self.single_agent and self.agent_type != 'rule':
                self._add_to_state(state, sim_step, key='sim_step', intersection=intersection)

        if self.single_agent or self.agent_type == 'rule':
            self._add_to_state(state, sim_step, key='sim_step', intersection=None)

        if (
            self.single_agent
            or self.agent_type == 'rule'
            or self.constants['multiagent']['state_interpolation'] == 0
        ):
            return self._process_state(state)

        # Multi-agent neighborhood pooling
        state_size = PER_AGENT_STATE_SIZE + GLOBAL_STATE_SIZE
        final_state = []
        for intersection in self.intersections:
            neighborhood = self.neighborhoods.get(intersection, [])
            intersection_state = state[intersection]
            row = np.zeros(shape=(state_size * self.max_num_neighbors,))
            row[:state_size] = np.array(intersection_state)[:state_size]

            for n, neighbor in enumerate(neighborhood):
                if neighbor in state:
                    scale = self.constants['multiagent']['state_interpolation']
                    extension = scale * np.array(state[neighbor])[:state_size]
                    start = (n + 1) * state_size
                    row[start:start + state_size] = extension

            final_state.append(row)

        return self._process_state(final_state)

    # ---- Reward -------------------------------------------------------------

    def get_reward(self, get_global: bool):
        reward_interpolation = self.constants['multiagent']['reward_interpolation']
        all_signals = self.client.get_all_signal_states()

        local_rewards = {}
        for intersection in self.intersections:
            signal_data = all_signals.get(intersection, {"waiting": []})
            waiting_cars = len(signal_data.get("waiting", []))
            local_rewards[intersection] = -float(waiting_cars) / 10.0

        if get_global:
            return sum(local_rewards.values())

        if len(self.intersections) == 1 or self.single_agent:
            return np.array([sum(local_rewards.values())])

        if reward_interpolation == 0.0:
            return np.array(list(local_rewards.values()))

        if reward_interpolation == 1.0:
            global_r = sum(local_rewards.values())
            return np.array([global_r] * len(self.intersections))

        # Distance-decay spatial interpolation
        arr = []
        for intersection in self.intersections:
            dists = self.distances[intersection]
            max_dist = max(dists.values()) if dists else 1.0
            local_rew = 0.0
            for inner in self.intersections:
                d = dists.get(inner, 1.0)
                r = local_rewards[inner]
                local_rew += pow(reward_interpolation, d / (max_dist or 1.0)) * r
            arr.append(local_rew)
        return np.array(arr)

    # ---- Action -------------------------------------------------------------

    def _execute_action(self, action: dict):
        """
        Translates decoded action dict to A/B Street signal configuration.
        action: {intersection_id: 0 (hold) or 1 (advance phase)}
        """
        for intersection, value in action.items():
            if value == 0:
                continue

            try:
                signal_config = self.client.get_signal(intersection)
                total_stages = len(signal_config.get("stages", []))
                if total_stages > 0:
                    current = signal_config.get("current_stage_index", 0)
                    signal_config["current_stage_index"] = (current + 1) % total_stages
                    self.client.set_signal(signal_config)
            except Exception:
                # Individual intersection failure shouldn't kill the episode
                pass

    # ---- Step / Reset -------------------------------------------------------

    def step(self, a, ep_step: int, get_global_reward: bool, def_agent: bool = False):
        action = self._process_action(a)
        if not def_agent:
            self._execute_action(action)

        self.sim_time_seconds += self.step_interval_seconds
        time_str = ABStreetClient.seconds_to_hhmmss(self.sim_time_seconds)
        self.client.sim_goto_time(time_str)

        s_ = self.get_state()
        r = self.get_reward(get_global_reward)

        done = False
        if ep_step >= self.constants['episode']['max_ep_steps']:
            if not self.eval_agent:
                s_ = self.reset()
            done = True

        return s_, r, done

    def reset(self):
        self._open_connection()
        return self.get_state()
