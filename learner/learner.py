# learner/learner.py
"""
Simple PPO Learner for Guandan project.

Requirements:
- PyTorch
- storage/replay_buffer.ReplayBuffer exists and actors push transitions there.

Notes / assumptions:
- Buffer stores transitions with keys:
    'obs'    : np.array shape (N, state_dim)
    'action' : np.array shape (N,) int
    'reward' : np.array shape (N,) float
    'done'   : np.array shape (N,) bool/int
    'logp'   : np.array shape (N,) float  (optional but recommended)
    'value'  : np.array shape (N,) float  (optional)
    'mask'   : np.array shape (N, max_actions)  (optional but recommended)
- If 'logp' or 'value' is missing, learner will compute forward with current policy as fallback (but this reduces correctness).
"""

import os
import time
import math
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from storage.replay_buffer import ReplayBuffer


# ---------------------------
# Policy-Value network
# ---------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, state_dim)
        returns logits (B, action_dim), values (B,)
        """
        h = self.net(x)
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        return logits, values


# ---------------------------
# Helper functions (masked logprob / entropy)
# ---------------------------
def masked_logits_to_probs(logits: torch.Tensor, mask: Optional[torch.Tensor]):
    """
    Convert logits to probabilities with mask.
    logits: (B, A)
    mask: None or (B, A) with 0/1 values (or bool)
    Returns: probs (B,A), log_probs (B,A)
    """
    if mask is None:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        return probs, log_probs
    # mask: convert to bool
    mask_bool = mask.bool()
    inf = -1e9
    # set illegal logits to large negative
    masked_logits = logits.masked_fill(~mask_bool, inf)
    # handle rows where mask is all False -> set uniform small probabilities across all (avoid NaN)
    # create denom mask to detect all-false rows
    row_has_valid = mask_bool.any(dim=-1)
    if not row_has_valid.all().item():
        # for rows without any valid action, make mask all True (fallback)
        masked_logits[~row_has_valid] = logits[~row_has_valid]
        mask_bool[~row_has_valid] = True

    probs = torch.softmax(masked_logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    return probs, log_probs


def gather_log_probs(log_probs: torch.Tensor, actions: torch.Tensor):
    """
    log_probs: (B, A) log probability matrix
    actions: (B,) long tensor of indices
    returns: (B,) selected log probs
    """
    return log_probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)


def masked_entropy(probs: torch.Tensor, mask: Optional[torch.Tensor]):
    """
    compute entropy of masked categorical distribution, treating masked probs as zero.
    probs: (B,A)
    mask: (B,A) optional
    returns: (B,) entropy per example
    """
    # clamp
    p = torch.clamp(probs, 1e-12, 1.0)
    ent = - (p * torch.log(p)).sum(dim=-1)
    return ent


# ---------------------------
# PPO Learner class
# ---------------------------
class PPOLearner:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 buffer: ReplayBuffer,
                 device: str = 'cpu',
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_eps: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 epochs: int = 4,
                 minibatch_size: int = 64,
                 max_grad_norm: float = 0.5,
                 save_dir: str = "checkpoints",
                 save_every_updates: int = 50):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = buffer

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_every_updates = save_every_updates

        # model & optimizer
        self.model = PolicyValueNet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # bookkeeping
        self.updates = 0

    def prepare_batch(self, data: Dict[str, Any]):
        """
        Accepts the dict returned by ReplayBuffer.get_all()
        Ensures presence and correct shapes for necessary arrays.
        Expected keys: obs, action, reward, done, mask (optional), logp (optional), value (optional)
        Returns a dict with torch tensors on device.
        """
        if not data:
            return None

        # required
        obs = np.asarray(data.get('obs'))
        actions = np.asarray(data.get('action'))
        rewards = np.asarray(data.get('reward'))
        dones = np.asarray(data.get('done')).astype(np.float32)

        # optional
        masks = data.get('mask', None)  # shape (N, A)
        old_logp = np.asarray(data.get('logp')) if data.get('logp') is not None else None
        old_value = np.asarray(data.get('value')) if data.get('value') is not None else None

        # convert to tensors
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        if masks is not None:
            masks_t = torch.tensor(np.asarray(masks), dtype=torch.float32, device=self.device)
            if masks_t.ndim == 1:
                masks_t = masks_t.unsqueeze(0)
        else:
            masks_t = None

        if old_logp is not None:
            old_logp_t = torch.tensor(old_logp, dtype=torch.float32, device=self.device)
        else:
            old_logp_t = None

        if old_value is not None:
            old_value_t = torch.tensor(old_value, dtype=torch.float32, device=self.device)
        else:
            old_value_t = None

        return {
            'obs': obs_t,
            'actions': actions_t,
            'rewards': rewards_t,
            'dones': dones_t,
            'masks': masks_t,
            'old_logp': old_logp_t,
            'old_value': old_value_t
        }

    def compute_gae_returns(self, rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor, last_value: float = 0.0):
        """
        rewards: (N,)
        dones: (N,)
        values: (N,)
        returns GAE advantages and returns (both numpy arrays)
        This implementation assumes the data is a single long trajectory or concatenated trajectories.
        """
        device = rewards.device
        N = rewards.shape[0]
        advantages = torch.zeros_like(rewards, device=device)
        last_gae = 0.0
        # iterate backwards
        for t in reversed(range(N)):
            mask = 1.0 - dones[t]
            next_value = last_value if t == N - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_gae = delta + self.gamma * self.lam * mask * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def ppo_update(self, batch: Dict[str, torch.Tensor]):
        """
        Do PPO update for one epoch over the provided batch (which is a dict with tensors).
        We'll use multiple minibatches outside this function (in train).
        batch contains tensors: obs, actions, rewards, dones, masks, old_logp, old_value
        """
        obs = batch['obs']
        actions = batch['actions']
        masks = batch.get('masks', None)
        old_logp = batch.get('old_logp', None)
        old_value = batch.get('old_value', None)
        rewards = batch['rewards']
        dones = batch['dones']

        # compute values and logp under current policy
        with torch.no_grad():
            logits, values = self.model(obs)
            probs, log_probs_all = masked_logits_to_probs(logits, masks)
            curr_logp = gather_log_probs(log_probs_all, actions)  # (N,)
            curr_values = values

        # If old_logp/old_value not provided, fallback to current (warning)
        if old_logp is None:
            old_logp = curr_logp.detach()
        if old_value is None:
            old_value = curr_values.detach()

        # compute advantages & returns (GAE)
        advantages, returns = self.compute_gae_returns(rewards, dones, old_value, last_value=old_value[-1].item() if old_value.shape[0] > 0 else 0.0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Now perform multiple epochs of minibatch SGD
        N = obs.shape[0]
        idxs = np.arange(N)
        for epoch in range(self.epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, self.minibatch_size):
                mb_idx = idxs[start:start + self.minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_obs = obs[mb_idx_t]
                mb_actions = actions[mb_idx_t]
                mb_returns = returns[mb_idx_t].detach()
                mb_adv = advantages[mb_idx_t].detach()
                mb_masks = masks[mb_idx_t] if masks is not None else None
                mb_old_logp = old_logp[mb_idx_t].detach()
                mb_old_value = old_value[mb_idx_t].detach()

                logits, values = self.model(mb_obs)
                probs, log_probs_all = masked_logits_to_probs(logits, mb_masks)
                new_logp = gather_log_probs(log_probs_all, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # value loss
                value_loss = torch.mean((values - mb_returns) ** 2)

                # entropy
                entropy = torch.mean(masked_entropy(probs, mb_masks))

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # bookkeeping
        self.updates += 1
        return {
            'policy_loss': float(policy_loss.detach().cpu().item()),
            'value_loss': float(value_loss.detach().cpu().item()),
            'entropy': float(entropy.detach().cpu().item())
        }

    def save(self, prefix: str = "ppo"):
        fname = os.path.join(self.save_dir, f"{prefix}_step{self.updates}.pth")
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'updates': self.updates
        }, fname)
        return fname

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception:
                pass
        self.updates = ckpt.get('updates', 0)
        return self

    def train(self,
              total_updates: int = 1000,
              fetch_interval: float = 1.0,
              samples_per_update: int = 2048,
              save_every: int = 50):
        """
        Main training loop.
        - total_updates: number of learner update iterations
        - fetch_interval: seconds to wait when buffer is empty before retrying
        - samples_per_update: target number of samples to accumulate before an update
        """
        for upd in range(total_updates):
            # wait until enough data in buffer
            waited = 0.0
            while len(self.buffer) < samples_per_update:
                time.sleep(fetch_interval)
                waited += fetch_interval
                if waited % 30 == 0:
                    print(f"[Learner] waiting for data... currently buffer size {len(self.buffer)}")

            # pull data
            data = self.buffer.pop_all()
            # data is list of transitions; convert to dict-of-arrays
            if not data:
                continue
            # Build dict arrays for prepare_batch
            keys = list(data[0].keys())
            batch = {k: [] for k in keys}
            for item in data:
                for k in keys:
                    batch[k].append(item.get(k, None))
            # convert lists to numpy arrays where possible
            for k in list(batch.keys()):
                try:
                    batch[k] = np.asarray(batch[k])
                except Exception:
                    batch[k] = np.asarray(batch[k], dtype=object)

            # prepare tensors
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue

            # if prepared missing old_logp or old_value, compute via current policy
            if prepared['old_logp'] is None or prepared['old_value'] is None:
                with torch.no_grad():
                    logits, values = self.model(prepared['obs'])
                    probs, log_probs_all = masked_logits_to_probs(logits, prepared['masks'])
                    curr_logp = gather_log_probs(log_probs_all, prepared['actions'])
                if prepared['old_logp'] is None:
                    prepared['old_logp'] = curr_logp.detach()
                if prepared['old_value'] is None:
                    prepared['old_value'] = values.detach()

            # call update (this function does multiple epochs internally)
            stats = self.ppo_update(prepared)

            print(f"[Learner] Update {self.updates}: policy_loss={stats['policy_loss']:.4f}, "
                  f"value_loss={stats['value_loss']:.4f}, entropy={stats['entropy']:.4f}, samples={len(prepared['actions'])}")

            # save periodically
            if self.updates % save_every == 0:
                p = self.save(prefix="ppo")
                print(f"[Learner] Saved checkpoint: {p}")

        print("[Learner] Training finished.")
