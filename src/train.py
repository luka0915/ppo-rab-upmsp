import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import types
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import torch

from src.config import Config
from src.data import resolve_train_instance
from src.env import PMSPEnvironment
from src.ppo import PPO
from src.metrics import MetricsTracker
from src.utils import save_config, set_seed


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _dataset_dict(dataset):
    return {
        'machines': dataset.MACHINES,
        'job_types': list(dataset.JOB_TYPES),
        'demand': dict(dataset.DEMAND),
        'processing_times': dataset.PROCESSING_TIMES,
        'due_dates': dataset.DUE_DATES,
        'setup_times': dataset.SETUP_TIMES,
        'weighted_tardiness': dataset.WEIGHTED_TARDINESS,
    }


def _make_env(ds):
    return PMSPEnvironment(
        machines=ds['machines'], job_types=ds['job_types'], demand=ds['demand'],
        processing_times=ds['processing_times'], due_dates=ds['due_dates'],
        setup_times=ds['setup_times'], weighted_tardiness=ds['weighted_tardiness'],
        initial_setup=[ds['job_types'][0]] * ds['machines'])


def _run_episode(env, agent, job_types, machines):
    """Run one episode with the current policy (Algorithm 2 inner loop) and
    return its transitions plus episode statistics. The initial setup status of
    every machine is randomized at the start of the episode (paper, Section 5.2).
    Critic-based adaptive batching (Algorithm 1) is applied at every step.
    """
    env.initial_setup = [random.choice(job_types) for _ in range(machines)]
    state = env.reset()
    transitions = []
    batch_sizes = []
    ep_reward = 0.0
    done = False
    while not done:
        valid = env.get_valid_actions()
        action, logp, value, cand, idx = agent.select_action(state, valid, env)
        m = action // len(job_types)
        f = job_types[action % len(job_types)]
        ns, r, done = env.step(m, f, ppo_agent=agent, batching='critic')
        transitions.append((state, cand, idx, r, value, logp, done))
        batch_sizes.append(env.batch_allocations[-1]['batch_size'])
        state = ns
        ep_reward += r
    return transitions, ep_reward, env._get_mean_weighted_tardiness(), env._get_makespan(), batch_sizes


def _resolve_workers(cfg):
    nw = getattr(cfg, 'NUM_WORKERS', None)
    return nw if nw else max(1, cpu_count() - 1)


def _cfg_snapshot(cfg):
    """Capture the runtime hyperparameter VALUES the workers need into a picklable
    namespace. Passing the Config class itself would pickle by reference, so the
    spawned workers would re-import config.py and miss any runtime mutations."""
    keys = ['ACTOR_LR', 'CRITIC_LR', 'TEMPERATURE', 'GAMMA', 'GAE_LAMBDA',
            'NUM_MINI_BATCHES', 'UPDATE_EPOCHS', 'EPSILON', 'MAX_GRAD_NORM', 'ETA']
    return types.SimpleNamespace(**{k: getattr(cfg, k, None) for k in keys})


# ---------------------------------------------------------------------------
# parallel trajectory collection (multiprocessing ON)
# ---------------------------------------------------------------------------
def _train_worker_init():
    torch.set_num_threads(1)


def _collect_worker(args):
    """Module-level worker (picklable for Windows spawn): rebuild the policy from
    the broadcast weights and collect `n_episodes` on-policy episodes."""
    actor_state, critic_state, cfg, ds, seed, n_episodes = args
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(ds)
    agent = PPO(state_dim=4, feature_dim=5, cfg=cfg)
    agent.actor.load_state_dict(actor_state)
    agent.critic.load_state_dict(critic_state)

    out = []
    for _ in range(n_episodes):
        tr, rew, mwt, mk, bsz = _run_episode(env, agent, ds['job_types'], ds['machines'])
        out.append((tr, rew, mwt, mk, bsz))
    return out


# ---------------------------------------------------------------------------
# training entry point
# ---------------------------------------------------------------------------
def train(cfg, dataset):
    """Algorithm 2: training procedure of actor and critic for PPO-RAB.

    Critic-based adaptive batching (Algorithm 1) is applied during training, so
    each environment step is one MDP transition assigning a whole same-family
    batch. Set cfg.USE_MULTIPROCESSING to choose between:
      - False (default): a single environment collects trajectories sequentially
        (faithful to the paper's Algorithm 2 loop).
      - True: parallel actors collect on-policy trajectories each round, then the
        main process performs the PPO update (standard parallel-PPO acceleration;
        training dynamics differ from the single-env loop).
    """
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    set_seed(cfg.TRAIN_SEED)
    save_config(cfg, dataset)

    ppo_agent = PPO(state_dim=4, feature_dim=5, cfg=cfg)
    metrics = MetricsTracker(cfg)

    use_mp = getattr(cfg, 'USE_MULTIPROCESSING', False)
    mode = f"parallel ({_resolve_workers(cfg)} workers)" if use_mp else "single-process"
    print("Starting PPO-RAB training")
    print(f"Episodes: {cfg.NUM_EPISODES} | Machines: {dataset.MACHINES} | Families: {len(dataset.JOB_TYPES)} | {mode}")
    print(f"N_B: {cfg.BATCH_SIZE} | mini-batches: {cfg.NUM_MINI_BATCHES} | N_U: {cfg.UPDATE_EPOCHS} | "
          f"gamma {cfg.GAMMA} | lambda {cfg.GAE_LAMBDA} | eps {cfg.EPSILON} | "
          f"tau {getattr(cfg, 'TEMPERATURE', 0.1)} | eta {cfg.ETA}")

    start_time = time.time()
    if use_mp:
        _train_parallel(cfg, dataset, ppo_agent, metrics, start_time)
    else:
        _train_sequential(cfg, dataset, ppo_agent, metrics, start_time)

    ppo_agent.save(f"{cfg.MODEL_DIR}final_model.pt")
    return ppo_agent, metrics


def _train_sequential(cfg, dataset, ppo_agent, metrics, start_time):
    env = _make_env(_dataset_dict(dataset))
    best_tardiness = float('inf')
    avg_batch_hist = []

    for episode in range(cfg.NUM_EPISODES):
        env.initial_setup = [random.choice(dataset.JOB_TYPES) for _ in range(dataset.MACHINES)]
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_batch_sizes = []
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action, log_prob, value, cand_features, chosen_idx = ppo_agent.select_action(
                state, valid_actions, env)
            machine = action // len(env.job_types)
            job_type = env.job_types[action % len(env.job_types)]

            next_state, reward, done = env.step(machine, job_type, ppo_agent=ppo_agent,
                                                batching='critic')

            batch_info = env.batch_allocations[-1]
            episode_batch_sizes.append(batch_info['batch_size'])

            ppo_agent.memory.add(state, cand_features, chosen_idx, reward, value, log_prob, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            if len(ppo_agent.memory) >= cfg.BATCH_SIZE:
                ppo_agent.update()

        mean_tardiness = env._get_mean_weighted_tardiness()
        makespan = env._get_makespan()
        metrics.update(episode, episode_reward, mean_tardiness, makespan)
        avg_batch = float(np.mean(episode_batch_sizes)) if episode_batch_sizes else 1.0
        avg_batch_hist.append(avg_batch)

        if mean_tardiness < best_tardiness:
            best_tardiness = mean_tardiness
            ppo_agent.save(f"{cfg.MODEL_DIR}best_model.pt")
            print(f"\nNew best at episode {episode + 1}: MWT={mean_tardiness:.2f}")

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{cfg.NUM_EPISODES} | steps {episode_steps} | "
                  f"reward {episode_reward:.2f} | MWT {mean_tardiness:.2f} | "
                  f"best {best_tardiness:.2f} | "
                  f"{time.time() - start_time:.1f}s")

        if (episode + 1) % cfg.SAVE_INTERVAL == 0:
            ppo_agent.save(f"{cfg.MODEL_DIR}checkpoint_{episode + 1}.pt")


def _train_parallel(cfg, dataset, ppo_agent, metrics, start_time):
    workers = _resolve_workers(cfg)
    ds = _dataset_dict(dataset)
    snap = _cfg_snapshot(cfg)
    best_tardiness = float('inf')
    episodes_done = 0
    round_idx = 0

    with Pool(processes=workers, initializer=_train_worker_init) as pool:
        while episodes_done < cfg.NUM_EPISODES:
            eps_this_round = min(workers, cfg.NUM_EPISODES - episodes_done)
            actor_state = {k: v.detach().cpu() for k, v in ppo_agent.actor.state_dict().items()}
            critic_state = {k: v.detach().cpu() for k, v in ppo_agent.critic.state_dict().items()}
            args = [(actor_state, critic_state, snap, ds,
                     cfg.TRAIN_SEED + round_idx * 1000 + w, 1)
                    for w in range(eps_this_round)]

            results = pool.map(_collect_worker, args)

            round_batches = []
            for worker_out in results:
                for tr, rew, mwt, mk, bsz in worker_out:
                    for t in tr:
                        ppo_agent.memory.add(*t)
                    metrics.update(episodes_done, rew, mwt, mk)
                    round_batches.extend(bsz)
                    episodes_done += 1
                    if mwt < best_tardiness:
                        best_tardiness = mwt
                        ppo_agent.save(f"{cfg.MODEL_DIR}best_model.pt")
                        print(f"  new best (parallel) ~ep {episodes_done}: MWT={mwt:.2f}")
                    if episodes_done % cfg.SAVE_INTERVAL == 0:
                        ppo_agent.save(f"{cfg.MODEL_DIR}checkpoint_{episodes_done}.pt")

            if len(ppo_agent.memory) >= cfg.BATCH_SIZE:
                ppo_agent.update()

            round_idx += 1
            avg_b = float(np.mean(round_batches)) if round_batches else 1.0
            print(f"[parallel] episodes {episodes_done}/{cfg.NUM_EPISODES} | "
                  f"best MWT {best_tardiness:.2f} | buffer {len(ppo_agent.memory)} | "
                  f"{time.time() - start_time:.1f}s")


if __name__ == "__main__":
    dataset = resolve_train_instance(getattr(Config, 'TRAIN_DATA_PATH', None))
    print(f"Training instance: {dataset.name} | machines={dataset.MACHINES} | "
          f"families={len(dataset.JOB_TYPES)} | jobs={sum(dataset.DEMAND.values())}")
    train(Config, dataset)
