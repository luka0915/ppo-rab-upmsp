import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# PPO agent for PPO-RAB.
#
# Actor and critic are SEPARATE fully-connected networks, each with three
# hidden layers of 64, 32, 16 nodes and ReLU on every hidden layer (paper 5.2).
#   - Actor:  state(4) -> 64 -> 32 -> 16 -> omega(5)     (weight vector, Eq.12)
#   - Critic: state(4) -> 64 -> 32 -> 16 -> 1            (state value V_psi)
#
# Policy (Eq. 12-13):  q_a = omega . phi_a ;  p_a = softmax(q_a / tau).
# Training (Algorithm 2): GAE (Eq.16), clipped surrogate (Eq.17), value loss
# (Eq.18). No entropy regularization and no reward scaling (not in the paper).
# =============================================================================

HIDDEN = (64, 32, 16)


def _mlp(in_dim, out_dim):
    layers = []
    prev = in_dim
    for h in HIDDEN:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))   # last layer: no activation
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Outputs the 5-dim action-feature weight vector omega (Eq. 12)."""
    def __init__(self, state_dim=4, feature_dim=5):
        super().__init__()
        self.net = _mlp(state_dim, feature_dim)

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """Outputs the scalar state value V_psi(s)."""
    def __init__(self, state_dim=4):
        super().__init__()
        self.net = _mlp(state_dim, 1)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class Memory:
    """Trajectory buffer D (Algorithm 2). Stores, per transition, the candidate
    feature matrix and the chosen-candidate index so that the policy ratio can
    be recomputed exactly under the current policy during updates."""
    def __init__(self):
        self.clear()

    def add(self, state, cand_features, chosen_idx, reward, value, log_prob, done):
        self.states.append(state)
        self.cand_features.append(cand_features)   # np.ndarray [k, 5]
        self.chosen_idx.append(chosen_idx)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states = []
        self.cand_features = []
        self.chosen_idx = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []


class PPO:
    def __init__(self, state_dim=4, feature_dim=5, cfg=None):
        self.cfg = cfg
        self.state_dim = state_dim
        self.feature_dim = feature_dim

        self.actor = Actor(state_dim, feature_dim)
        self.critic = Critic(state_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.ACTOR_LR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.CRITIC_LR)

        self.memory = Memory()
        self.temperature = getattr(cfg, 'TEMPERATURE', 0.1)   # tau (Eq.13)

    # ----------------------------------------------------------- policy utils
    def _omega(self, state_np):
        with torch.no_grad():
            return self.actor(torch.as_tensor(state_np, dtype=torch.float32)).numpy()

    def value_of(self, state_np):
        """V_psi(s) as a python float (used by Algorithm 1 and rollout)."""
        with torch.no_grad():
            return float(self.critic(torch.as_tensor(state_np, dtype=torch.float32)).item())

    def policy_distribution(self, state_np, cand_features):
        """Return (probs, omega) for the given candidates (Eq. 12-13)."""
        omega = self._omega(state_np)
        scores = cand_features @ omega                      # q_a = omega . phi_a
        scaled = scores / self.temperature
        scaled = scaled - scaled.max()                      # numerical stability
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        return probs, omega

    def select_action(self, state, valid_actions, env, greedy=False):
        """Sample a family-machine action via the softmax policy (Eq. 12-13).

        Returns: action, log_prob, value, cand_features [k,5], chosen_idx.
        """
        n_fam = len(env.job_types)
        cand_features = np.stack([
            env.get_action_features(a // n_fam, env.job_types[a % n_fam])
            for a in valid_actions
        ]).astype(np.float32)

        probs, _ = self.policy_distribution(state, cand_features)

        if greedy:
            idx = int(np.argmax(probs))
        else:
            idx = int(np.random.choice(len(valid_actions), p=probs))

        action = valid_actions[idx]
        log_prob = float(np.log(probs[idx] + 1e-12))
        value = self.value_of(state)
        return action, log_prob, value, cand_features, idx

    # --------------------------------------------------------------- training
    def compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation (Eq. 16)."""
        gae = 0.0
        advantages = [0.0] * len(rewards)
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.cfg.GAMMA * self.cfg.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def _log_probs_for(self, states, cand_features, chosen_idx):
        """Recompute log pi_theta(a|s) for a batch under the CURRENT policy."""
        omegas = self.actor(states)                          # [B, 5]
        logps = []
        for i in range(len(chosen_idx)):
            cf = torch.as_tensor(cand_features[i], dtype=torch.float32)   # [k,5]
            scores = cf @ omegas[i]                           # [k]
            logp_all = F.log_softmax(scores / self.temperature, dim=0)
            logps.append(logp_all[chosen_idx[i]])
        return torch.stack(logps)

    def update(self):
        states = torch.as_tensor(np.array(self.memory.states), dtype=torch.float32)
        old_log_probs = torch.as_tensor(self.memory.log_probs, dtype=torch.float32)
        values = self.memory.values
        rewards = self.memory.rewards
        dones = self.memory.dones
        cand_features = self.memory.cand_features
        chosen_idx = self.memory.chosen_idx

        advantages = torch.as_tensor(self.compute_gae(rewards, values, dones), dtype=torch.float32)
        returns = advantages + torch.as_tensor(values, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(self.memory)
        mb_size = max(1, n // self.cfg.NUM_MINI_BATCHES)

        for _ in range(self.cfg.UPDATE_EPOCHS):              # N_U epochs
            perm = torch.randperm(n)
            for start in range(0, n, mb_size):
                idx = perm[start:start + mb_size]
                idx_list = idx.tolist()

                mb_states = states[idx]
                mb_old_logp = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                mb_cand = [cand_features[i] for i in idx_list]
                mb_chosen = [chosen_idx[i] for i in idx_list]

                # ----- actor update: clipped surrogate (Eq. 17) -----
                new_logp = self._log_probs_for(mb_states, mb_cand, mb_chosen)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.EPSILON, 1 + self.cfg.EPSILON) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if getattr(self.cfg, 'MAX_GRAD_NORM', None):
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.MAX_GRAD_NORM)
                self.actor_optimizer.step()

                # ----- critic update: value loss (Eq. 18) -----
                value_pred = self.critic(mb_states)
                value_loss = F.mse_loss(value_pred, mb_ret)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                if getattr(self.cfg, 'MAX_GRAD_NORM', None):
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.MAX_GRAD_NORM)
                self.critic_optimizer.step()

        self.memory.clear()

    # ------------------------------------------------------------- persistence
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'temperature': self.temperature,
            'feature_dim': self.feature_dim,
            'state_dim': self.state_dim,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        if 'actor_optimizer_state_dict' in ckpt:
            self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])
        self.temperature = ckpt.get('temperature', self.temperature)
        self.feature_dim = ckpt.get('feature_dim', self.feature_dim)
