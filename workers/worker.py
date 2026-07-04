# workers/worker.py
import numpy as np
from collections import defaultdict
from copy import deepcopy
import requests


class Worker:
    def __init__(self, constants, device, env, worker_id, data_collector):
        self.constants = constants
        self.device = device
        self.env = env
        self.id = worker_id
        self.data_collector = data_collector

    def _reset(self):
        raise NotImplementedError

    def _get_prediction(self, states, actions=None, ep_step=None):
        raise NotImplementedError

    def _get_action(self, prediction):
        raise NotImplementedError

    def _copy_shared_model_to_local(self):
        raise NotImplementedError

    def _read_episode_results(self, results: dict) -> dict:
        """
        Replaces _read_edge_results (SUMO edgeData XML).
        Pulls equivalent metrics from A/B Street's HTTP API instead.

        Metrics collected per episode:
          - mean_trip_duration:  average finished trip duration in seconds
          - cancelled_trips:     count of trips that didn't finish
          - mean_congestion:     (actual - lower_bound) / lower_bound averaged
                                 across finished trips — equivalent to per-edge
                                 delay the SUMO version was computing
          - road_throughput:     total vehicle-hours across all roads this hour
        """
        host = self.env.client.base_url

        try:
            finished = requests.get(
                f"{host}/data/get-finished-trips", timeout=30
            ).json()

            lower_bounds_raw = requests.get(
                f"{host}/data/all-trip-time-lower-bounds", timeout=30
            ).json()
            lower_bounds = {str(e[0]): float(e[1]) for e in lower_bounds_raw}

            durations = []
            congestion_ratios = []
            n_cancelled = 0

            for trip in finished:
                if trip.get('duration') is None:
                    n_cancelled += 1
                    continue
                dur = float(trip['duration'])
                durations.append(dur)
                lb = lower_bounds.get(str(trip['id']))
                if lb and lb > 0:
                    congestion_ratios.append((dur - lb) / lb)

            if durations:
                results['mean_trip_duration'].append(
                    sum(durations) / len(durations)
                )
            results['cancelled_trips'].append(n_cancelled)
            if congestion_ratios:
                results['mean_congestion'].append(
                    sum(congestion_ratios) / len(congestion_ratios)
                )

            road_thruput = requests.get(
                f"{host}/data/get-road-thruput", timeout=30
            ).json()
            # road_thruput is [[road, agent_type, hour, count], ...]
            total_thruput = sum(row[3] for row in road_thruput)
            results['road_throughput'].append(float(total_thruput))

        except Exception as e:
            # Don't let metric collection kill an episode
            print(f"[Worker {self.id}] _read_episode_results failed: {e}")

        return results

    def eval_episode(self, results: dict) -> dict:
        ep_rew = 0.0
        step = 0
        state = self.env.reset()
        self._reset()

        while True:
            prediction = self._get_prediction(state, ep_step=step)
            action = self._get_action(prediction)
            next_state, reward, done = self.env.step(
                action, step, get_global_reward=True
            )
            ep_rew += reward
            if done:
                break
            if not isinstance(state, dict):
                state = np.copy(next_state)
            else:
                state = deepcopy(next_state)
            step += 1

        results = self._read_episode_results(results)
        results['rew'].append(ep_rew)
        return results

    def eval_episodes(self, current_rollout, model_state=None, ep_count=None):
        self._copy_shared_model_to_local()
        results = defaultdict(list)

        for _ in range(self.constants['episode']['eval_num_eps']):
            results = self.eval_episode(results)

        if current_rollout is not None:
            results['rollout'] = [current_rollout]

        # Average across episodes before logging — same smoothing as original
        results = {k: sum(v) / len(v) for k, v in results.items()}

        if self.data_collector:
            self.data_collector.collect_ep(
                results,
                model_state,
                ep_count + self.constants['episode']['eval_num_eps']
                if ep_count is not None else None,
            )
