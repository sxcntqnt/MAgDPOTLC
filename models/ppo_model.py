# models/ppo_model.py
import numpy as np
import math
from tinygrad.tensor import Tensor
import tinygrad.nn as nn
from typing import List, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Weight initialization — replaces layer_init_filter from utils.utils
# Orthogonal initialization is standard for PPO; tinygrad has no built-in
# ---------------------------------------------------------------------------

def orthogonal_init(weight: Tensor, gain: float = 1.0) -> Tensor:
    """
    Orthogonal initialization for Linear layer weights.
    Replaces layer_init_filter — tinygrad has no apply() or init utilities.
    """
    shape = weight.shape
    flat_shape = (shape[0], int(np.prod(shape[1:])))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return Tensor(gain * q.astype(np.float32))


def init_linear(layer: nn.Linear, gain: float = 1.0, bias_const: float = 0.0):
    """Applies orthogonal init to a tinygrad Linear layer in place."""
    layer.weight.assign(orthogonal_init(layer.weight, gain))
    layer.bias.assign(
        Tensor(np.full(layer.bias.shape, bias_const, dtype=np.float32))
    )


# ---------------------------------------------------------------------------
# ModelBody — shared trunk, direct tinygrad port
# nn.Sequential and nn.ReLU don't exist in tinygrad; inline them
# ---------------------------------------------------------------------------

class ModelBody:
    def __init__(self, input_size: int, hidden_size: int):
        self.name = 'model_body'
        self.fc = nn.Linear(input_size, hidden_size)
        init_linear(self.fc, gain=math.sqrt(2))

    def __call__(self, states: Tensor) -> Tensor:
        return self.fc(states).relu()

    def parameters(self) -> List[Tensor]:
        return [self.fc.weight, self.fc.bias]


# ---------------------------------------------------------------------------
# ActorModel — discrete action head (retained for existing intersection env)
# Outputs raw logits; distribution sampling happens in NN_Model.forward()
# ---------------------------------------------------------------------------

class ActorModel:
    def __init__(self, hidden_size: int, action_size: int):
        self.name = 'actor'
        self.fc = nn.Linear(hidden_size, action_size)
        init_linear(self.fc, gain=0.01)  # small gain keeps initial policy near-uniform

    def __call__(self, hidden_states: Tensor) -> Tensor:
        return self.fc(hidden_states)

    def parameters(self) -> List[Tensor]:
        return [self.fc.weight, self.fc.bias]


# ---------------------------------------------------------------------------
# CriticModel — value head, unchanged structure
# ---------------------------------------------------------------------------

class CriticModel:
    def __init__(self, hidden_size: int):
        self.name = 'critic'
        self.fc = nn.Linear(hidden_size, 1)
        init_linear(self.fc, gain=1.0)

    def __call__(self, hidden_states: Tensor) -> Tensor:
        return self.fc(hidden_states)

    def parameters(self) -> List[Tensor]:
        return [self.fc.weight, self.fc.bias]


# ---------------------------------------------------------------------------
# Categorical distribution — replaces torch.distributions.Categorical
# ---------------------------------------------------------------------------

class Categorical:
    """
    Discrete action distribution over logits.
    Replaces torch.distributions.Categorical for the existing intersection env.
    """

    def __init__(self, logits: Tensor):
        # Subtract max for numerical stability before softmax
        logits_np = logits.numpy()
        logits_np = logits_np - logits_np.max(axis=-1, keepdims=True)
        self._logits = Tensor(logits_np)
        self._probs = self._logits.softmax(axis=-1)

    def sample(self) -> Tensor:
        probs_np = self._probs.numpy()
        batch = probs_np.shape[0] if len(probs_np.shape) > 1 else 1
        n_actions = probs_np.shape[-1]
        if batch == 1:
            action = np.random.choice(n_actions, p=probs_np.flatten())
            return Tensor([action])
        actions = np.array([
            np.random.choice(n_actions, p=probs_np[i])
            for i in range(batch)
        ])
        return Tensor(actions.astype(np.float32))

    def log_prob(self, actions: Tensor) -> Tensor:
        actions_np = actions.numpy().astype(int).flatten()
        log_probs_np = np.log(self._probs.numpy() + 1e-8)
        selected = np.array([
            log_probs_np[i, actions_np[i]] if len(log_probs_np.shape) > 1
            else log_probs_np[actions_np[i]]
            for i in range(len(actions_np))
        ], dtype=np.float32)
        return Tensor(selected)

    def entropy(self) -> Tensor:
        probs_np = self._probs.numpy()
        ent = -np.sum(probs_np * np.log(probs_np + 1e-8), axis=-1)
        return Tensor(ent.astype(np.float32))


# ---------------------------------------------------------------------------
# Gaussian distribution — for continuous matatu action space
# ---------------------------------------------------------------------------

class Gaussian:
    """
    Continuous action distribution N(mean, std²).
    Used by MatatuActorModel for vehicle_count, departure_spread, branch_weights.
    """

    def __init__(self, mean: Tensor, log_std: Tensor):
        self.mean = mean
        # Clamp log_std: prevents std collapsing to zero or growing unbounded
        log_std_np = log_std.numpy().clip(-3, 1)
        self.log_std = Tensor(log_std_np)
        self.std = self.log_std.exp()

    def sample(self) -> Tensor:
        noise = Tensor(np.random.randn(*self.mean.shape).astype(np.float32))
        return self.mean + self.std * noise

    def log_prob(self, x: Tensor) -> Tensor:
        """Log probability — shape (batch, action_dims)"""
        var = self.std ** 2
        return (
            -0.5 * ((x - self.mean) ** 2 / (var + 1e-8))
            - self.log_std
            - Tensor([math.log(math.sqrt(2 * math.pi))])
        )

    def entropy(self) -> Tensor:
        """Entropy — shape (batch, action_dims)"""
        return 0.5 * (
            Tensor([math.log(2 * math.pi * math.e)]) + 2 * self.log_std
        )


# ---------------------------------------------------------------------------
# MatatuActorModel — continuous head for matatu action space
# Outputs (mean, log_std) for Gaussian policy over the full action vector
# ---------------------------------------------------------------------------

class MatatuActorModel:
    """
    Continuous actor head for the matatu demand calibration action space.
    Raw outputs are unbounded floats; decoding to vehicle_count/spread/weights
    is handled by MatatuActionSpace.decode() in the scenario generator,
    keeping the network itself simple (no baked-in sigmoid/softmax).
    """

    def __init__(self, hidden_size: int, action_size: int):
        self.name = 'matatu_actor'
        self.fc_mean = nn.Linear(hidden_size, action_size)
        # log_std as a separate learned parameter vector (not input-dependent)
        # This is simpler and more stable than a second linear head for log_std
        self.log_std = Tensor(
            np.zeros(action_size, dtype=np.float32)
        )
        init_linear(self.fc_mean, gain=0.01)

    def __call__(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.fc_mean(hidden_states)
        # Expand log_std to match batch dimension
        batch = hidden_states.shape[0]
        log_std_expanded = self.log_std.reshape(1, -1).expand(batch, -1)
        return mean, log_std_expanded

    def parameters(self) -> List[Tensor]:
        return [self.fc_mean.weight, self.fc_mean.bias, self.log_std]


# ---------------------------------------------------------------------------
# NN_Model — original discrete model, direct tinygrad port
# Retained for existing IntersectionsEnv compatibility
# ---------------------------------------------------------------------------

class NN_Model:
    """
    Direct tinygrad port of the original PyTorch NN_Model.
    Used by the existing intersection signal-control environment.
    Returns the same dict structure as the original forward() for
    drop-in compatibility with PPOWorker.
    """

    def __init__(self, state_size: int, action_size: int, hidden_layer_size: int, device=None):
        # device param retained for API compatibility — tinygrad handles device differently
        self.body_model = ModelBody(state_size, hidden_layer_size)
        self.actor_model = ActorModel(hidden_layer_size, action_size)
        self.critic_model = CriticModel(hidden_layer_size)
        self.models = [self.body_model, self.actor_model, self.critic_model]

    def parameters(self) -> List[Tensor]:
        params = []
        for m in self.models:
            params += m.parameters()
        return params

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.numpy().flatten() for p in self.parameters()])

    def set_flat_params(self, flat: np.ndarray):
        offset = 0
        for p in self.parameters():
            size = int(np.prod(p.shape))
            new_data = flat[offset:offset + size].reshape(p.shape)
            p.assign(Tensor(new_data.astype(np.float32)))
            offset += size

    def param_count(self) -> int:
        return sum(int(np.prod(p.shape)) for p in self.parameters())

    def __call__(self, states: Tensor, actions: Optional[Tensor] = None) -> Dict:
        hidden = self.body_model(states)
        v = self.critic_model(hidden)
        logits = self.actor_model(hidden)
        dist = Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions).reshape(-1, 1)
        entropy = dist.entropy().reshape(-1, 1)
        return {
            'a': actions,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }


# ---------------------------------------------------------------------------
# MatatuNN_Model — continuous model for matatu demand calibration
# Shares body/critic structure with NN_Model; swaps actor for MatatuActorModel
# ---------------------------------------------------------------------------

class MatatuNN_Model:
    """
    Actor-Critic for the matatu PPO training loop.
    Action space is continuous (vehicle counts, departure spreads, branch weights)
    so uses Gaussian policy rather than Categorical.

    forward() returns the same dict structure as NN_Model for compatibility
    with MatatuPPOWorker, with log_pi_a summed across action dims.
    """

    def __init__(self, state_size: int, action_size: int, hidden_layer_size: int, device=None):
        self.body_model = ModelBody(state_size, hidden_layer_size)
        self.actor_model = MatatuActorModel(hidden_layer_size, action_size)
        self.critic_model = CriticModel(hidden_layer_size)
        self.models = [self.body_model, self.actor_model, self.critic_model]

    def parameters(self) -> List[Tensor]:
        params = []
        for m in self.models:
            params += m.parameters()
        return params

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.numpy().flatten() for p in self.parameters()])

    def set_flat_params(self, flat: np.ndarray):
        offset = 0
        for p in self.parameters():
            size = int(np.prod(p.shape))
            new_data = flat[offset:offset + size].reshape(p.shape)
            p.assign(Tensor(new_data.astype(np.float32)))
            offset += size

    def param_count(self) -> int:
        return sum(int(np.prod(p.shape)) for p in self.parameters())

    def __call__(self, states: Tensor, actions: Optional[Tensor] = None) -> Dict:
        hidden = self.body_model(states)
        v = self.critic_model(hidden)
        mean, log_std = self.actor_model(hidden)
        dist = Gaussian(mean=mean, log_std=log_std)

        if actions is None:
            actions = dist.sample()

        # Sum log_prob across action dims — joint log probability of full action vector
        log_prob = dist.log_prob(actions).sum(axis=-1, keepdim=True)
        entropy = dist.entropy().sum(axis=-1, keepdim=True)

        return {
            'a': actions,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v,
        }
