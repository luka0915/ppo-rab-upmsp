import numpy as np

# =============================================================================
# PMSPEnvironment
#
# Faithful implementation of the UPMSP-SDFS MDP described in the paper
# "Scheduling Unrelated Parallel Machines with Sequence-Dependent Family
#  Setups via DRL-Driven Rollout and Adaptive Batching" (PPO-RAB).
#
# Data model: a "job_type" is a job FAMILY. All jobs of a family share the
# same weight w_f, due date d_f and per-machine processing time p_{i,f}, with
# `demand[f]` identical jobs. Under this (paper-consistent) instance design the
# general per-job formulas of the paper reduce to family-level computations:
#   - a2 = (1/|J_f|) sum p_{i,j} = p_{i,f}
#   - a5 = min_{J_j in R_f} d_j - C_t(i) = d_f - C_t(i)
#   - Algorithm 1, line 4 (argmin w_j T_hat_j over R_f) is trivial.
#
# State   s_t = [s1, s2, s3, s4]            (Eq. 11-14)
# Action  phi_a = [a1, a2, a3, a4, a5]      (Eq. 16-20)
# Reward  r_t = sum_{j in B_t}(-w_j T_j) - 1{t=T-1} * MWT   (Eq. 23)
# All state components and action features are 0-1 normalized [50].
# =============================================================================


class PMSPEnvironment:
    def __init__(self, machines, job_types, demand, processing_times, due_dates,
                 setup_times, weighted_tardiness, initial_setup=None):
        self.machines = machines
        self.job_types = list(job_types)
        self.demand = demand.copy()
        self.processing_times = processing_times      # family -> [p over machines]
        self.due_dates = due_dates                    # family -> d_f
        self.setup_times = setup_times                # family(from) -> {family(to): sigma}
        self.weighted_tardiness = weighted_tardiness  # family -> w_f

        self.initial_setup = initial_setup.copy() if initial_setup is not None else [None] * machines

        self.feature_dim = 5   # action feature dimension (a1..a5)
        self.state_dim = 4     # state dimension (s1..s4)

        # ----- instance-static normalization bounds (no look-ahead / test info) -----
        self._nm = machines
        self._nj = sum(self.demand.values())
        self._dmax = max(self.due_dates.values())
        self._wmax = max(self.weighted_tardiness.values())
        self._pmax = max(max(times) for times in self.processing_times.values())
        self._setupmax = max(max(row.values()) for row in self.setup_times.values())
        total_proc = sum(self.demand[f] * np.mean(self.processing_times[f]) for f in self.job_types)
        # loose makespan upper bound used as the time scale for 0-1 normalization
        self._ub_mk = total_proc / machines + (self._nj / machines) * self._setupmax
        self._ub_mk = max(self._ub_mk, 1e-6)

        self.batch_allocations = []
        self.current_step = 0
        self.reset()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _mm(value, lo, hi):
        """0-1 min-max normalization with clipping (score normalization [50])."""
        if hi - lo <= 1e-12:
            return 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def reset(self):
        self.remaining_jobs = self.demand.copy()
        self.machine_setup = self.initial_setup.copy()
        self.machine_schedule = [[] for _ in range(self.machines)]
        self.machine_completion_times = [0.0] * self.machines
        self.current_step = 0
        self.batch_allocations = []
        return self.get_size_independent_state()

    # ------------------------------------------------------------------ state
    def _assigned_weighted_tardiness(self):
        """s3 raw: sum of weighted tardiness over jobs assigned so far."""
        total = 0.0
        for schedule in self.machine_schedule:
            for job in schedule:
                t = max(0.0, job['completion'] - self.due_dates[job['job']])
                total += t * self.weighted_tardiness[job['job']]
        return total

    def _state_vector(self, completion_times, remaining, s3_raw):
        """Build normalized [s1,s2,s3,s4] from raw scalars (Eq. 11-14)."""
        s1 = max(completion_times) if completion_times else 0.0          # Eq.11
        s2 = sum(1 for c in completion_times if c == 0)                  # Eq.12 (count)
        s3 = s3_raw                                                      # Eq.13 (sum)
        s4 = sum(remaining.values())                                    # Eq.14 (count)
        state = [
            self._mm(s1, 0.0, self._ub_mk),
            self._mm(s2, 0.0, self._nm),
            self._mm(s3, 0.0, self._wmax * self._ub_mk * self._nj),
            self._mm(s4, 0.0, self._nj),
        ]
        return np.array(state, dtype=np.float32)

    def get_size_independent_state(self):
        return self._state_vector(self.machine_completion_times,
                                  self.remaining_jobs,
                                  self._assigned_weighted_tardiness())

    def _get_makespan(self):
        return max(
            max((job['completion'] for job in schedule), default=0.0)
            for schedule in self.machine_schedule
        )

    def _get_mean_weighted_tardiness(self):
        total_t = 0.0
        total_n = 0
        for schedule in self.machine_schedule:
            for job in schedule:
                t = max(0.0, job['completion'] - self.due_dates[job['job']])
                total_t += t * self.weighted_tardiness[job['job']]
                total_n += 1
        return total_t / total_n if total_n > 0 else 0.0

    # --------------------------------------------------------------- actions
    def get_valid_actions(self):
        """Action candidate set Phi_t: family-machine pairs with R_f != empty (Eq. 15)."""
        valid = []
        for m in range(self.machines):
            for j, f in enumerate(self.job_types):
                if self.remaining_jobs[f] > 0:
                    valid.append(m * len(self.job_types) + j)
        return valid

    def get_action_features(self, machine, family):
        """phi_a = [a1,a2,a3,a4,a5] for assigning family `family` to `machine` (Eq. 16-20)."""
        Gi = self.machine_setup[machine]                                # current setup status
        a1 = self.setup_times[Gi][family] if Gi is not None else 0.0    # Eq.16  sigma_{Gi,f}
        a2 = self.processing_times[family][machine]                     # Eq.17  p_bar_{i,f}
        a3 = self.machine_completion_times[machine]                     # Eq.18  C_t(i)
        a4 = a3 + a1 + a2                                               # Eq.19 estimated completion
        a5 = self.due_dates[family] - a3                               # Eq.20 slack (min_{R_f} d_j - C_t(i))

        features = [
            self._mm(a1, 0.0, self._setupmax),
            self._mm(a2, 0.0, self._pmax),
            self._mm(a3, 0.0, self._ub_mk),
            self._mm(a4, 0.0, self._ub_mk + self._setupmax + self._pmax),
            self._mm(a5, -self._ub_mk, self._dmax),
        ]
        return np.array(features, dtype=np.float32)

    # ----------------------------------------------- adaptive batching (Alg.1)
    def adaptive_batching(self, machine, family, ppo_agent, eta):
        """Critic-based Adaptive Batching (Algorithm 1).

        Returns the batch size |B_t| (number of consecutive same-family jobs to
        assign to `machine`). Uses the critic V_psi to decide how far to expand
        the batch, terminating when the relative drop in the estimated value
        exceeds the tolerance eta. The completion estimate excludes setup, as in
        the paper (line 1: C_hat(i) <- C_t(i); line 7: C_hat(i) += p_{i,j'}).
        """
        rem = self.remaining_jobs[family]
        if rem <= 0:
            return 0

        p = self.processing_times[family][machine]
        d = self.due_dates[family]
        w = self.weighted_tardiness[family]

        base_completions = list(self.machine_completion_times)
        base_remaining = dict(self.remaining_jobs)
        base_s3 = self._assigned_weighted_tardiness()

        C_hat = self.machine_completion_times[machine]   # line 1
        s3 = base_s3
        v_star = -np.inf                                 # line 1
        batch = 0

        while batch < rem:                               # line 2: while R_f != empty
            # lines 3-5: pick job with min w*T_hat (trivial: next same-family job), add to B_t
            C_hat = C_hat + p                            # completion of the newly added job (line 7)
            t_new = max(0.0, C_hat - d)                  # T_hat_j (line 3)
            s3 = s3 + w * t_new
            batch += 1

            # line 6: obtain s' by assigning jobs in B_t to M_i
            completions = list(base_completions)
            completions[machine] = C_hat
            remaining = dict(base_remaining)
            remaining[family] = rem - batch
            s_prime = self._state_vector(completions, remaining, s3)

            v_prime = ppo_agent.value_of(s_prime)        # line 8: V' = V_psi(s')

            if v_prime > v_star:                         # lines 9-10
                v_star = v_prime
            elif abs(v_star) > 1e-8 and abs(v_star - v_prime) / abs(v_star) > eta:  # line 11
                break                                    # lines 12-13 (B_t includes this job)
        return batch

    # ------------------------------------------------------------------- step
    def _assign_one(self, machine, family, apply_setup):
        """Assign a single job of `family` to `machine`; return its completion time."""
        if apply_setup:
            Gi = self.machine_setup[machine]
            setup = self.setup_times[Gi][family] if Gi is not None else 0.0
        else:
            setup = 0.0
        start = self.machine_completion_times[machine]
        proc = self.processing_times[family][machine]
        completion = start + setup + proc
        self.machine_schedule[machine].append({
            'job': family,
            'start': start,
            'setup': setup,
            'processing': proc,
            'completion': completion,
        })
        self.machine_completion_times[machine] = completion
        self.machine_setup[machine] = family
        self.remaining_jobs[family] -= 1
        self.current_step += 1
        return completion

    def step(self, machine, family, ppo_agent=None, eta=None, batching=None):
        """Apply one MDP action (a family-machine pair) plus its job batch.

        `batching` selects how many same-family jobs are assigned:
          - 'critic'    : critic-based adaptive batching (Algorithm 1) -- used at
                          test time (Algorithm 4) and during training (Algorithm 2,
                          line 10).
          - 'none'      : a single job -- used by the rollout procedure (Alg. 3).
        If `batching` is None it defaults to 'critic' when (ppo_agent, eta) are
        given, else 'none'. Returns (next_state, reward, done); reward is Eq. 23.
        """
        if self.remaining_jobs[family] <= 0:
            return self.get_size_independent_state(), 0.0, False

        # ----- determine batch size -----
        if batching is None:
            batching = 'critic' if (ppo_agent is not None and eta is not None) else 'none'

        if batching == 'critic':
            batch_size = self.adaptive_batching(machine, family, ppo_agent,
                                                eta if eta is not None else 0.015)
        else:  # 'none'
            batch_size = 1
        batch_size = max(1, min(batch_size, self.remaining_jobs[family]))

        # ----- assign the batch (first job incurs the family setup) -----
        reward = 0.0
        batch_rewards = []
        w = self.weighted_tardiness[family]
        d = self.due_dates[family]
        for k in range(batch_size):
            if self.remaining_jobs[family] <= 0:
                break
            completion = self._assign_one(machine, family, apply_setup=(k == 0))
            t_j = max(0.0, completion - d)
            r_j = -w * t_j                                # Eq.23 first term per job
            reward += r_j
            batch_rewards.append(r_j)

        actual_batch = len(batch_rewards)
        self.batch_allocations.append({
            'machine': machine,
            'job_type': family,
            'batch_size': actual_batch,
            'rewards': batch_rewards,
            'total_reward': reward,
        })

        # ----- terminal reward (Eq. 23: -1{t=T-1} MWT) -----
        done = not any(v > 0 for v in self.remaining_jobs.values())
        if done:
            reward += -self._get_mean_weighted_tardiness()

        return self.get_size_independent_state(), reward, done
