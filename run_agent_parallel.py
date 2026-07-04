# run_agent_parallel.py
import multiprocessing as mp
from copy import deepcopy

from models.ppo_model import MatatuNN_Model, NN_Model
from utils.utils import Counter, get_rule_set_class
from environments.ab_street_env import (
    ABStreetIntersectionsEnv,
    PER_AGENT_STATE_SIZE,
    GLOBAL_STATE_SIZE,
    ACTION_SIZE,
)
from workers.ppo_worker import PPOWorker
from workers.rule_worker import RuleBasedWorker
from workers.matatu_ppo_worker import MatatuPPOWorker, SharedWeightBuffer
from environments.abstreet_client import ABStreetClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(constants, device, worker_id: str, eval_agent: bool) -> ABStreetIntersectionsEnv:
    scenario_path = constants["environment"]["scenario_path"]

    worker_index = int(worker_id.split("_")[-1])

    return ABStreetIntersectionsEnv(
        constants=constants,
        device=device,
        worker_index=worker_index,
        worker_name=worker_id,
        eval_agent=eval_agent,
        scenario_path=scenario_path,
    )

def _make_nn(constants, device, env: ABStreetIntersectionsEnv) -> NN_Model:
    """
    Builds an NN_Model sized to the environment's state/action dimensions.
    Replaces get_state_action_size() + NN_Model() call from the original —
    ABStreetIntersectionsEnv exposes these directly rather than needing
    the neighborhood-scrape path that IntersectionsEnv required.
    """
    state_size = env.state_size     # set by ABStreetIntersectionsEnv / Environment base
    action_size = env.action_size
    hidden = constants['ppo']['hidden_layer_size']
    return NN_Model(state_size, action_size, hidden, device)




# ---------------------------------------------------------------------------
# Process targets
# ---------------------------------------------------------------------------

def train_worker(
    worker_id: str,
    shared_buffer: SharedWeightBuffer,
    data_collector,
    rollout_counter: Counter,
    constants: dict,
    device,
):
    env = _make_env(constants, device, worker_id, eval_agent=False)
    local_NN = _make_nn(constants, device, env)
    local_NN.set_flat_params(shared_buffer.read())  # start from shared weights

    optimizer_cls = __import__('tinygrad.nn.optim', fromlist=['Adam']).Adam
    optimizer = optimizer_cls(
        local_NN.parameters(),
        lr=constants['ppo']['learning_rate']
    )

    worker = PPOWorker(
        constants=constants,
        device=device,
        env=env,
        data_collector=None,
        shared_buffer=shared_buffer,   # replaces shared_NN
        local_NN=local_NN,
        optimizer=optimizer,
        worker_id=worker_id,
    )

    print(worker_id, "created PPOWorker", flush=True)

    train_step = 0
 
    while rollout_counter.get() < constants['episode']['num_train_rollouts'] + 1:
        print(worker_id, "calling train_rollout", flush=True)

        worker.train_rollout(train_step)

        print(worker_id, "returned from train_rollout", flush=True)

        rollout_counter.increment()

        print(worker_id, "incremented counter", flush=True)

        train_step += 1


    # No connection.close() — ABStreet uses stateless HTTP, nothing to tear down


def eval_worker(
    worker_id: str,
    shared_buffer: SharedWeightBuffer,
    data_collector,
    rollout_counter: Counter,
    constants: dict,
    device,
):
    # Eval worker gets a port above the training worker range
    n_train = constants['parallel']['num_workers']
    eval_idx = int(worker_id.split('_')[-1])
    eval_port = 1234 + n_train + eval_idx

    env = ABStreetIntersectionsEnv(
        constants=constants,
        device=device,
        worker_index=n_train + eval_idx,
        worker_name=worker_id,
        eval_agent=True,
        scenario_path=constants["environment"]["scenario_path"],
    )

    local_NN = _make_nn(constants, device, env)

    worker = PPOWorker(
        constants=constants,
        device=device,
        env=env,
        data_collector=data_collector,
        shared_buffer=shared_buffer,
        local_NN=local_NN,
        optimizer=None,             # eval worker never updates weights
        worker_id=worker_id,
    )

    last_eval = 0
    while True:
        curr_r = rollout_counter.get()
        if curr_r % constants['episode']['eval_freq'] == 0 and last_eval != curr_r:
            last_eval = curr_r
            # Sync to latest shared weights before eval
            local_NN.set_flat_params(shared_buffer.read())
            worker.eval_episodes(curr_r)
        if curr_r >= constants['episode']['num_train_rollouts'] + 1:
            break

    # Final eval at training end
    local_NN.set_flat_params(shared_buffer.read())
    worker.eval_episodes(curr_r)


def test_worker(
    worker_id: str,
    ep_counter: Counter,
    constants: dict,
    device,
    worker=None,
    data_collector=None,
    shared_buffer: SharedWeightBuffer = None,
):
    if not worker:
        env = _make_env(constants, device, worker_id, eval_agent=True)
        local_NN = _make_nn(constants, device, env)
        if shared_buffer is not None:
            local_NN.set_flat_params(shared_buffer.read())

        worker = PPOWorker(
            constants=constants,
            device=device,
            env=env,
            data_collector=data_collector,
            shared_buffer=shared_buffer,
            local_NN=local_NN,
            optimizer=None,
            worker_id=worker_id,
        )

    while ep_counter.get() < constants['episode']['test_num_eps']:
        worker.eval_episodes(ep_count=ep_counter.get())
        ep_counter.increment(constants['episode']['eval_num_eps'])


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def train_PPO(constants: dict, device, data_collector):
    """
    Drop-in replacement for the original train_PPO.
    SharedWeightBuffer replaces shared_NN.share_memory() + shared optimizer.
    Each training worker owns its own Adam; weight sync goes through the buffer.
    """
    # Build a throwaway env just to get state/action sizes for buffer sizing
    probe_env = _make_env(constants, device, 'probe_0', eval_agent=False)
    probe_NN = _make_nn(constants, device, probe_env)
    shared_buffer = SharedWeightBuffer(probe_NN.param_count())
    shared_buffer.write(probe_NN.get_flat_params())  # seed with fresh random weights

    rollout_counter = Counter()
    processes = []

    # Eval worker — always eval_0, port above training range
    p = mp.Process(
        target=eval_worker,
        args=('eval_0', shared_buffer, data_collector, rollout_counter, constants, device),
    )
    p.start()
    processes.append(p)

    # Training workers
    for i in range(constants['parallel']['num_workers']):
        p = mp.Process(
            target=train_worker,
            args=(f'train_{i}', shared_buffer, data_collector, rollout_counter, constants, device),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        print(
            p.pid,
            p.exitcode
        )


def test_PPO(constants: dict, device, data_collector, loaded_flat_params: 'np.ndarray'):
    """
    loaded_flat_params: numpy array from a previous shared_buffer.read() checkpoint.
    Replaces loaded_model (PyTorch state_dict) from the original.
    """
    probe_env = _make_env(constants, device, 'probe_0', eval_agent=False)
    probe_NN = _make_nn(constants, device, probe_env)
    shared_buffer = SharedWeightBuffer(probe_NN.param_count())
    shared_buffer.write(loaded_flat_params)

    ep_counter = Counter()
    processes = []

    for i in range(constants['parallel']['num_workers']):
        p = mp.Process(
            target=test_worker,
            args=(f'test_{i}', ep_counter, constants, device, None, data_collector, shared_buffer),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def test_rule_based(constants: dict, device, data_collector):
    """
    Rule-based baseline — unchanged topology, just swaps IntersectionsEnv
    for ABStreetIntersectionsEnv. RuleBasedWorker is env-agnostic.
    """
    rule_set_class = get_rule_set_class(constants['rule']['rule_set'])
    ep_counter = Counter()
    processes = []

    for i in range(constants['parallel']['num_workers']):
        worker_id = f'test_{i}'
        env = _make_env(constants, device, worker_id, eval_agent=True)
        rule_set_params = deepcopy(constants['rule']['rule_set_params'])
        # phases comes from the env now, not from a net scrape
        rule_set_params['phases'] = env.phases

        worker = RuleBasedWorker(
            constants=constants,
            device=device,
            env=env,
            rule_set=rule_set_class(
                rule_set_params,
                constants['environment']['scenario_path'],
                constants,
            ),
            data_collector=data_collector,
            worker_id=worker_id,
        )

        p = mp.Process(
            target=test_worker,
            args=(worker_id, ep_counter, constants, device, worker, None, None),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
