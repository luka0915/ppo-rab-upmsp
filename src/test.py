import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import csv
import time
import random
from copy import deepcopy
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import Config
from src.data import resolve_test_instances
from src.env import PMSPEnvironment
from src.ppo import PPO
from src.utils import set_seed

# =============================================================================
# Scheduling process of PPO-RAB (Algorithm 4) using the rollout procedure
# (Algorithm 3) and critic-based adaptive batching (Algorithm 1).
# =============================================================================


class LightEnv(PMSPEnvironment):
    """Thin wrapper that also carries the config object, kept for compatibility
    with the evaluation drivers (ppo_checkpoint_vaildation.py, compare_*)."""
    def __init__(self, machines, job_types, demand, processing_times, due_dates,
                 setup_times, weighted_tardiness, config=None, initial_setup=None):
        super().__init__(machines, job_types, demand, processing_times, due_dates,
                         setup_times, weighted_tardiness, initial_setup)
        self.config = config


def _decode(action, env):
    n_fam = len(env.job_types)
    return action // n_fam, env.job_types[action % n_fam]


def above_average_candidates(ppo_agent, state, valid_actions, env):
    """Phi' = { a in Phi_t : q_a >= mean(q) }   (Eq. 28)."""
    n_fam = len(env.job_types)
    cand = np.stack([env.get_action_features(a // n_fam, env.job_types[a % n_fam])
                     for a in valid_actions]).astype(np.float32)
    omega = ppo_agent._omega(state)
    q = cand @ omega
    threshold = q.mean()
    phi = [valid_actions[i] for i in range(len(valid_actions)) if q[i] >= threshold]
    if not phi:
        phi = [valid_actions[int(np.argmax(q))]]
    return phi


def rollout(base_env, first_action, ppo_agent):
    """Algorithm 3: Rollout(a; pi_theta, U).

    Generate a complete schedule by forcing the candidate action `first_action`
    at the current step and then following the softmax policy until all jobs are
    assigned. Adaptive batching is NOT used inside the rollout (one job per step,
    selecting the min-weighted-tardiness job of the chosen family, which is
    trivial under family-level identical jobs).
    Returns (MWT, schedule) where schedule is [[(completion, family), ...], ...].
    """
    env = deepcopy(base_env)

    machine, family = _decode(first_action, env)
    env.step(machine, family)                       # k == t: a_k = a (single job)

    while any(v > 0 for v in env.remaining_jobs.values()):
        state = env.get_size_independent_state()
        valid = env.get_valid_actions()
        action, _, _, _, _ = ppo_agent.select_action(state, valid, env)   # Eq.21-22
        machine, family = _decode(action, env)
        env.step(machine, family)                   # single job, no batching

    mwt = env._get_mean_weighted_tardiness()
    schedule = [[(j['completion'], j['job']) for j in env.machine_schedule[i]]
                for i in range(env.machines)]
    return mwt, schedule


def _seed_rollout(seed):
    """Seed the RNGs so a given candidate rollout is reproducible. This makes the
    multiprocessing and sequential paths produce identical per-candidate results,
    so the multiprocessing toggle affects only throughput, not schedule quality."""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def _worker_init():
    """Keep each parallel worker on a single torch thread for CPU-fair scaling."""
    torch.set_num_threads(1)


def _rollout_candidate(args):
    """Module-level worker (picklable for the Windows spawn start method)."""
    base_env, action, ppo_agent, seed = args
    _seed_rollout(seed)
    mwt, sched = rollout(base_env, action, ppo_agent)
    return action, mwt, sched


def test_with_model(ppo_agent, env, time_limit=None, eta=None,
                    use_multiprocessing=None, num_workers=None):
    """Algorithm 4: scheduling process with PPO-RAB.

    Returns (best_schedule, allocation_sequence, convergence_data). The returned
    schedule U* is the best COMPLETE schedule found by the rollout procedure,
    guaranteeing a feasible schedule even if the time limit is reached before all
    jobs are committed.
    """
    cfg = getattr(env, 'config', None) or Config
    if time_limit is None:
        time_limit = getattr(cfg, 'TIME_LIMIT', 300)
    if eta is None:
        eta = getattr(cfg, 'ETA', 0.015)
    if use_multiprocessing is None:
        use_multiprocessing = getattr(cfg, 'USE_MULTIPROCESSING', False)
    if num_workers is None:
        num_workers = getattr(cfg, 'NUM_WORKERS', None)
    workers = num_workers if num_workers else max(1, cpu_count() - 1)
    base_seed = (getattr(cfg, 'TEST_SEED', 0) + 1) * 9999   # per-candidate rollout seeding

    best_mwt = float('inf')
    best_schedule = None
    allocation_sequence = []
    convergence = []
    job_counter = 0
    step_idx = 0

    start = time.time()
    mode = f"multiprocessing ({workers} workers)" if use_multiprocessing else "single-process"
    print(f"\nPPO-RAB scheduling | time limit {time_limit}s | eta {eta} | {mode}")

    while any(v > 0 for v in env.remaining_jobs.values()):
        if time.time() - start >= time_limit:
            print(f"=== time limit reached ({time_limit}s) ===")
            break

        state = env.get_size_independent_state()
        valid_actions = env.get_valid_actions()
        phi_prime = above_average_candidates(ppo_agent, state, valid_actions, env)

        # per-candidate seeds: identical work in either execution mode
        seeds = [base_seed + step_idx * 9999 + i for i in range(len(phi_prime))]

        # --- rollout every above-average candidate, keep the best (lines 9-16) ---
        # Time guard: stop launching/collecting rollouts once the limit is hit,
        # but always let at least one rollout complete (best_schedule None on the
        # first step) so a feasible complete schedule U* is guaranteed.
        best_a = None
        best_a_mwt = float('inf')
        best_a_sched = None

        if use_multiprocessing and len(phi_prime) > 1:
            args = [(env, a, ppo_agent, seeds[i]) for i, a in enumerate(phi_prime)]
            with Pool(processes=workers, initializer=_worker_init) as pool:
                for a, mwt, sched in pool.imap_unordered(_rollout_candidate, args):
                    if mwt < best_a_mwt:
                        best_a_mwt, best_a, best_a_sched = mwt, a, sched
                    if best_schedule is not None and time.time() - start >= time_limit:
                        break   # remaining workers are terminated on context exit
        else:
            for i, a in enumerate(phi_prime):
                if best_schedule is not None and time.time() - start >= time_limit:
                    break
                _seed_rollout(seeds[i])
                mwt, sched = rollout(env, a, ppo_agent)
                if mwt < best_a_mwt:
                    best_a_mwt, best_a, best_a_sched = mwt, a, sched

        step_idx += 1
        if best_a is None:          # out of time before any rollout finished this step
            break

        if best_a_mwt < best_mwt:
            best_mwt = best_a_mwt
            best_schedule = best_a_sched
            convergence.append({'time': time.time() - start, 'mwt': best_mwt})
            print(f"  new best MWT {best_mwt:.2f} @ {time.time() - start:.1f}s")

        # stop before committing another batch once the time limit is exceeded
        if time.time() - start >= time_limit:
            print(f"=== time limit reached ({time_limit}s) ===")
            break

        # --- commit a* with critic-based adaptive batching (lines 16-20) ---
        machine, family = _decode(best_a, env)
        env.step(machine, family, ppo_agent=ppo_agent, eta=eta)
        binfo = env.batch_allocations[-1]
        for i in range(binfo['batch_size']):
            ct = env.machine_schedule[machine][-1 - i]['completion']
            allocation_sequence.append((job_counter, machine, family, ct))
            job_counter += 1

    if not np.isinf(best_mwt):
        convergence.append({'time': time.time() - start, 'mwt': best_mwt})
    print(f"Final best MWT {best_mwt:.2f} | elapsed {time.time() - start:.1f}s")
    return best_schedule, allocation_sequence, convergence


def _schedule_mwt(schedule, env):
    total_t, total_n = 0.0, 0
    for m_sched in schedule:
        for ct, f in m_sched:
            total_t += max(0.0, ct - env.due_dates[f]) * env.weighted_tardiness[f]
            total_n += 1
    return total_t / total_n if total_n else 0.0


def evaluate_instance(ppo_agent, inst, cfg):
    """Build an env from a loaded instance and run PPO-RAB once. The initial
    setup status comes from the instance file (fixed during testing)."""
    env = LightEnv(
        machines=inst.MACHINES, job_types=inst.JOB_TYPES, demand=inst.DEMAND,
        processing_times=inst.PROCESSING_TIMES, due_dates=inst.DUE_DATES,
        setup_times=inst.SETUP_TIMES, weighted_tardiness=inst.WEIGHTED_TARDINESS,
        config=cfg, initial_setup=inst.INITIAL_SETUP)

    start = time.time()
    schedule, _, convergence = test_with_model(ppo_agent, env)
    elapsed = time.time() - start
    mwt = _schedule_mwt(schedule, env) if schedule else float('inf')
    makespan = max((ct for ms in schedule for ct, _ in ms), default=0.0) if schedule else 0.0
    res = {'instance': inst.name, 'machines': inst.MACHINES,
           'families': len(inst.JOB_TYPES), 'jobs': inst.total_jobs,
           'mwt': mwt, 'makespan': makespan, 'time': elapsed}
    return res, schedule, convergence, env


def _save_instance_outputs(env, schedule, convergence, res, out_dir):
    """Write per-instance convergence CSV + plot."""
    os.makedirs(out_dir, exist_ok=True)

    # convergence CSV + plot (best MWT over wall-clock time)
    with open(os.path.join(out_dir, 'convergence.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time', 'mwt'])
        for d in convergence:
            w.writerow([round(d['time'], 4), round(d['mwt'], 4)])
    if convergence:
        plt.figure(figsize=(8, 5))
        plt.plot([d['time'] for d in convergence], [d['mwt'] for d in convergence], 'b-o', ms=4)
        plt.xlabel('Time (s)')
        plt.ylabel('Best Mean Weighted Tardiness')
        plt.title(f"{res['instance']} - PPO-RAB convergence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'convergence.png'), dpi=200)
        plt.close()


def _dataset_name(path):
    """Folder name used for the per-dataset results subfolder (e.g. 'D2')."""
    if not path:
        return 'dataset'
    p = os.path.normpath(str(path))
    if os.path.isfile(p):
        return os.path.basename(os.path.dirname(p)) or 'dataset'
    return os.path.basename(p) or 'dataset'


def run_test_suite(ppo_agent, cfg):
    """Evaluate every dataset folder in cfg.TEST_DATA_PATH (str, list, or None).

    Layout under cfg.RESULTS_DIR:
      <dataset>/<instance>/  -> convergence.csv (time, mwt at each update),
                                convergence.png
    """
    paths = getattr(cfg, 'TEST_DATA_PATH', None)
    if paths is None or isinstance(paths, str):
        paths = [paths]
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    all_results = []
    for ds_path in paths:
        ds_name = _dataset_name(ds_path)
        ds_dir = os.path.join(cfg.RESULTS_DIR, ds_name)
        instances = resolve_test_instances(ds_path)
        print(f"\n========== Dataset {ds_name}: {len(instances)} instance(s) ==========")
        ds_results = []
        for inst in instances:
            print(f"\n##### {ds_name}/{inst.name} | machines={inst.MACHINES} "
                  f"families={len(inst.JOB_TYPES)} jobs={inst.total_jobs} #####")
            res, schedule, convergence, env = evaluate_instance(ppo_agent, inst, cfg)
            res['dataset'] = ds_name
            if schedule:
                _save_instance_outputs(env, schedule, convergence, res,
                                       os.path.join(ds_dir, inst.name))
            ds_results.append(res)
            print(f"  -> MWT={res['mwt']:.2f} | time={res['time']:.1f}s")
        all_results.extend(ds_results)

    print("\n===== Test summary =====")
    for r in all_results:
        print(f"  {r.get('dataset', '?')}/{r['instance']}: MWT={r['mwt']:.2f}  time={r['time']:.1f}s")
    print(f"Saved per-dataset results under {cfg.RESULTS_DIR}")
    return all_results


def main():
    set_seed(Config.TEST_SEED)
    model_path = getattr(Config, 'TEST_MODEL_PATH', None) or f"{Config.MODEL_DIR}best_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Train first or set Config.TEST_MODEL_PATH.")
    ppo_agent = PPO(state_dim=4, feature_dim=5, cfg=Config)
    ppo_agent.load(model_path)
    print(f"Loaded model: {model_path}")
    return run_test_suite(ppo_agent, Config)


if __name__ == "__main__":
    main()
