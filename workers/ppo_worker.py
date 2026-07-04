# workers/ppo_worker.py
import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from workers.worker import Worker
from workers.matatu_ppo_worker import SharedWeightBuffer, clip_grad_norm, EpisodeTransition


# ---------------------------------------------------------------------------
# Tinygrad replacements for PyTorch utils
# (Storage, tensor, random_sample, ensure_shared_grads)
# ---------------------------------------------------------------------------

def tg_tensor(x, device=None) -> Tensor:
    """
    Replaces utils.tensor() — converts numpy array or scalar to Tensor.
    device param retained for API compatibility; tinygrad manages device
    via environment variables, not per-tensor assignment.
    """
    if isinstance(x, Tensor):
        return x
    if isinstance(x, np.ndarray):
        return Tensor(x.astype(np.float32))
    return Tensor(np.array(x, dtype=np.float32))


def random_sample(indices: np.ndarray, minibatch_size: int):
    """
    Replaces utils.random_sample — yields shuffled minibatch index arrays.
    Original used torch.randperm; numpy equivalent is identical behaviour.
    """
    indices = np.random.permutation(indices)
    n = len(indices)
    for start in range(0, n, minibatch_size):
        yield indices[start:start + minibatch_size]


class Storage:
    """
    Replaces utils.Storage — rolling buffer for PPO rollout data.
    Original stored torch tensors; this stores tinygrad Tensors.

    Keys stored per step: a, log_pi_a, ent, v, r, m, s, adv, ret
    placeholder() adds two extra slots at the end for bootstrap value.
    """

    def __init__(self, rollout_length: int):
        self.rollout_length = rollout_length
        self._data = {}     # key -> list of Tensors, one per step
        self._size = 0

    def add(self, d: dict):
        for k, v in d.items():
            if k not in self._data:
                self._data[k] = []
            # Store as Tensor, detached — we don't want grad through storage
            if isinstance(v, Tensor):
                self._data[k].append(Tensor(v.numpy()))
            else:
                self._data[k].append(v)
        self._size += 1

    def placeholder(self):
        """
        Adds None-padded slots so GAE indexing [i+1] doesn't go out of bounds.
        Original Storage.placeholder() did the same for torch.
        """
        for k in self._data:
            self._data[k].append(None)

    def __getattr__(self, key: str):
        """Allows storage.r[i], storage.v[i], etc."""
        if key.startswith('_'):
            raise AttributeError(key)
        return self._data.get(key, [])

    def __setattr__(self, key: str, value):
        if key.startswith('_') or key in ('rollout_length',):
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def cat(self, keys: list):
        """
        Concatenates stored tensors along batch dim for a list of keys.
        Replaces storage.cat() from original — filters out None placeholders.
        Returns one Tensor per key, shape (rollout_length * num_agents, ...).
        """
        result = []
        for k in keys:
            valid = [v for v in self._data[k] if v is not None]
            stacked = np.concatenate([v.numpy() for v in valid], axis=0)
            result.append(Tensor(stacked.astype(np.float32)))
        return result


# ---------------------------------------------------------------------------
# PPOWorker — tinygrad port
# ---------------------------------------------------------------------------

class PPOWorker(Worker):
    """
    Tinygrad port of the original PPOWorker.

    Key differences from original:
    - shared_NN (PyTorch module) → shared_buffer (SharedWeightBuffer)
    - _copy_shared_model_to_local: load_state_dict → set_flat_params from buffer
    - ensure_shared_grads → _push_grads_to_shared via buffer after each epoch
    - tensor() → tg_tensor()
    - torch.min → Tensor.minimum()
    - .detach() → Tensor(x.numpy()) — explicit copy with no grad
    - states.size(0) → states.shape[0]
    - NN.eval() / NN.train() — tinygrad has no train/eval mode distinction,
      dropout/batchnorm not used here so this is a no-op; kept as comments
      for structural clarity
    """

    def __init__(
        self,
        constants: dict,
        device,
        env,
        data_collector,
        shared_buffer: SharedWeightBuffer,
        local_NN,
        optimizer,           # Adam instance or None for eval workers
        worker_id: str,
        dont_reset: bool = False,
    ):
        super(PPOWorker, self).__init__(constants, device, env, worker_id, data_collector)
        self.NN = local_NN
        self.shared_buffer = shared_buffer
        self.opt = optimizer

        print(worker_id, "before reset", flush=True)

        if not dont_reset:
            self.state = self.env.reset()

        print(worker_id, "after reset", flush=True)

        self.ep_step = 0
        self.num_agents = (
            len(env.intersections)
            if not self.constants['agent']['single_agent']
            else 1
        )

    def _reset(self):
        pass

    def _get_prediction(self, states, actions=None, ep_step=None):
        """
        Replaces original _get_prediction.
        Converts states to Tensor, calls NN forward, returns prediction dict.
        actions passed during training epochs for log_prob recomputation.
        """
        states_t = tg_tensor(states)

        actions_t = tg_tensor(actions) if actions is not None else None
   
        return self.NN(states_t, actions_t)

    def _get_action(self, prediction) -> np.ndarray:
        """Extracts action from prediction dict as numpy array."""
        return prediction['a'].numpy()

    def _copy_shared_model_to_local(self):
        """Replaces load_state_dict — reads flat params from shared buffer."""
        self.NN.set_flat_params(self.shared_buffer.read())

    def _push_grads_to_shared(self):
        """
        Replaces ensure_shared_grads.
        After local backward pass, push updated local weights to shared buffer.
        Using full weight replacement rather than gradient injection —
        cleaner for tinygrad since there's no param._grad assignment.
        """
        self.shared_buffer.write(self.NN.get_flat_params())

    def _stack(self, val) -> np.ndarray:
        assert not isinstance(val, list)
        return np.stack([val] * self.num_agents)

    def _detach(self, t: Tensor) -> Tensor:
        """Explicit detach: copy data into new Tensor with no grad history."""
        return Tensor(t.numpy())

    def train_rollout(self, total_step: int) -> int:

        storage = Storage(self.constants['episode']['rollout_length'])


        state = np.copy(self.state)


        step_times = []

        self._copy_shared_model_to_local()


        rollout_amt = 0

        # ---- Rollout collection --------------------------------------------
        while rollout_amt < self.constants['episode']['rollout_length']:
            start = time.time()

            prediction = self._get_prediction(state)
            action = self._get_action(prediction)
            next_state, reward, done = self.env.step(
                action, self.ep_step, get_global_reward=False
            )
            self.ep_step += 1

            if done:
                self._copy_shared_model_to_local()
                self.ep_step = 0

            if self.ep_step > self.constants['episode']['warmup_ep_steps']:
                storage.add(prediction)
                storage.add({
                    'r': tg_tensor(reward).reshape(-1, 1),
                    'm': tg_tensor(self._stack(1 - done)).reshape(-1, 1),
                    's': tg_tensor(state),
                })
                rollout_amt += 1

            state = np.copy(next_state)
            total_step += 1
            step_times.append(time.time() - start)

        self.state = np.copy(state)

        # Bootstrap value from final state
        final_prediction = self._get_prediction(state)
        storage.add(final_prediction)
        storage.placeholder()

        # ---- GAE computation -----------------------------------------------
        advantages = Tensor(
            np.zeros((self.num_agents, 1), dtype=np.float32)
        )
        returns = self._detach(final_prediction['v'])

        for i in reversed(range(self.constants['episode']['rollout_length'])):
            r = storage.r[i]
            m = storage.m[i]
            v_next = storage.v[i + 1]
            v_curr = storage.v[i]

            discount = self.constants['ppo']['discount']
            gae_tau = self.constants['ppo']['gae_tau']

            # Discounted return
            returns = r + discount * m * returns

            # GAE td-error
            td_error = r + discount * m * v_next - v_curr
            advantages = (
                advantages * gae_tau * discount * m + td_error
            )

            storage.adv[i] = self._detach(advantages)
            storage.ret[i] = self._detach(returns)

        # ---- Prepare training batch ----------------------------------------
        states, actions, log_probs_old, returns, advantages = storage.cat(
            ['s', 'a', 'log_pi_a', 'ret', 'adv']
        )

        # Detach: these are fixed targets during optimization epochs
        actions = self._detach(actions)
        log_probs_old = self._detach(log_probs_old)

        adv_np = advantages.numpy()
        advantages = Tensor(
            ((adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)).astype(np.float32)
        )

        n_samples = states.shape[0]

        # ---- Optimization epochs -------------------------------------------
        for _ in range(self.constants['ppo']['optimization_epochs']):
            # Sync at start of each epoch — same as original
            self._copy_shared_model_to_local()

            sampler = random_sample(
                np.arange(n_samples),
                self.constants['ppo']['minibatch_size'],
            )

            for batch_idx in sampler:
                batch_idx = batch_idx.astype(np.int32)

                # Index tensors via numpy — tinygrad supports integer array indexing
                s_b = Tensor(states.numpy()[batch_idx])
                a_b = Tensor(actions.numpy()[batch_idx])
                lp_b = Tensor(log_probs_old.numpy()[batch_idx])
                ret_b = Tensor(returns.numpy()[batch_idx])
                adv_b = Tensor(advantages.numpy()[batch_idx])

                prediction = self._get_prediction(s_b, a_b)

                # PPO clipped surrogate
                ratio = (prediction['log_pi_a'] - lp_b).exp()

                obj = ratio * adv_b
                obj_clipped = (
                    ratio.clip(
                        1.0 - self.constants['ppo']['ppo_ratio_clip'],
                        1.0 + self.constants['ppo']['ppo_ratio_clip'],
                    ) * adv_b
                )

                policy_loss = (
                    -obj.minimum(obj_clipped).mean()
                    - self.constants['ppo']['entropy_weight']
                    * prediction['ent'].mean()
                )

                value_loss = (
                    self.constants['ppo']['value_loss_coef']
                    * (ret_b - prediction['v']).pow(2).mean()
                )

                loss = policy_loss + value_loss

                self.opt.zero_grad()
                loss.backward()

                if self.constants['ppo']['clip_grads']:
                    clip_grad_norm(
                        self.NN.parameters(),
                        self.constants['ppo']['gradient_clip'],
                    )

                # Push weights to shared buffer after each minibatch update
                # replaces ensure_shared_grads + opt.step() on shared params
                self.opt.step()
                self._push_grads_to_shared()

        return total_step
