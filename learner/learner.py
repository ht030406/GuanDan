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
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from storage.replay_buffer import ReplayBuffer
from Agent.message2state import sparse_to_dense

ACTION_DIM = 500
# ---------------------------
# Policy-Value network
# ---------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: tuple = (256, 256, 128, 64)):
        super().__init__()
        hs = list(hidden_sizes)
        assert len(hs) == 4, "hidden_sizes must be 4-tuple"
        self.net = nn.Sequential(
            nn.Linear(state_dim, hs[0]),
            nn.ReLU(),
            nn.Linear(hs[0], hs[1]),
            nn.ReLU(),
            nn.Linear(hs[1], hs[2]),
            nn.ReLU(),
            nn.Linear(hs[2], hs[3]),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hs[3], action_dim)
        self.value_head = nn.Linear(hs[3], 1)

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
                 epochs: int = 6,
                 minibatch_size: int = 256,
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

        Expected required keys:  obs, action, reward, done
        Supported optional keys:
            - masks or mask:        (B, A) float add-mask (legal=0, illegal=-1e9), or (B, A) bool
                                    Also accepts (A,) and will broadcast to (B, A).
            - action_feats:         (B, A, F) float features per action index; or (A, F) and will
                                    broadcast to (B, A, F).
            - logp (old_logp):      previous log-prob
            - value (old_value):    previous value

        Returns a dict of torch tensors on device with keys:
            obs, actions, rewards, dones, masks, action_feats, old_logp, old_value
        """
        if not data:
            return None

        illegal_value = -1e9  # 用于把非法位钉到极小

        # ---------- required ----------

        obs_np = np.asarray(data.get('obs'))
        actions_np = np.asarray(data.get('action'))
        rewards_np = np.asarray(data.get('reward'))
        dones_np = np.asarray(data.get('done')).astype(np.float32)


        B = len(actions_np)
        device = getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        action_dim_attr = getattr(self, "action_dim", None)  # 可用于校验 A

        # ---------- optional: masks / mask ----------
        masks_in = data.get('masks', None)
        if masks_in is None:
            masks_in = data.get('mask', None)

        masks_t = None
        A_from_masks = None
        if masks_in is not None:
            try:
                masks_arr = np.asarray(masks_in)
                # 单样本/一维情况：统一扩成 (B, A)
                if masks_arr.ndim == 1:
                    A_from_masks = masks_arr.shape[0]
                    masks_arr = masks_arr.reshape(1, -1).repeat(B, axis=0)
                elif masks_arr.ndim == 2:
                    if masks_arr.shape[0] == 1 and B > 1:
                        A_from_masks = masks_arr.shape[1]
                        masks_arr = masks_arr.repeat(B, axis=0)
                    else:
                        A_from_masks = masks_arr.shape[1]
                        if masks_arr.shape[0] != B:
                            raise ValueError(f"masks batch dim {masks_arr.shape[0]} != B {B}")
                else:
                    raise ValueError(f"masks must have ndim 1 or 2, got {masks_arr.ndim}")

                # bool 掩码 -> 加法掩码；float 掩码直接转 float32
                if masks_arr.dtype == np.bool_:
                    add_mask = np.full_like(masks_arr, fill_value=illegal_value, dtype=np.float32)
                    add_mask[masks_arr] = 0.0
                    masks_arr = add_mask
                else:
                    masks_arr = masks_arr.astype(np.float32)

                masks_t = torch.tensor(masks_arr, dtype=torch.float32, device=device)
            except Exception as e:
                self.logger.error(f"[prepare_batch] failed to parse masks: {e}")
                masks_t = None

        # ---------- optional: action_feats ----------
        feats_in = data.get('action_feats', None)
        action_feats_t = None
        A_from_feats = None
        F_from_feats = None
        if feats_in is not None:
            try:
                feats_arr = np.asarray(feats_in)
                if feats_arr.ndim == 2:
                    # (A, F) -> 广播到 (B, A, F)
                    A_from_feats, F_from_feats = feats_arr.shape
                    feats_arr = feats_arr.reshape(1, A_from_feats, F_from_feats).repeat(B, axis=0)
                elif feats_arr.ndim == 3:
                    if feats_arr.shape[0] == 1 and B > 1:
                        A_from_feats, F_from_feats = feats_arr.shape[1], feats_arr.shape[2]
                        feats_arr = feats_arr.repeat(B, 1, 1)
                    else:
                        if feats_arr.shape[0] != B:
                            raise ValueError(f"action_feats batch dim {feats_arr.shape[0]} != B {B}")
                        A_from_feats, F_from_feats = feats_arr.shape[1], feats_arr.shape[2]
                else:
                    raise ValueError(f"action_feats must have ndim 2 or 3, got {feats_arr.ndim}")

                feats_arr = feats_arr.astype(np.float32)
                action_feats_t = torch.tensor(feats_arr, dtype=torch.float32, device=device)
            except Exception as e:
                self.logger.error(f"[prepare_batch] failed to parse action_feats: {e}")
                action_feats_t = None

        # ---------- infer / validate action_dim (A) ----------
        inferred_A = None
        if A_from_masks is not None and A_from_feats is not None:
            if A_from_masks != A_from_feats:
                self.logger.error(
                    f"[prepare_batch] A mismatch: masks A={A_from_masks} vs action_feats A={A_from_feats}")
            inferred_A = A_from_masks
        elif A_from_masks is not None:
            inferred_A = A_from_masks
        elif A_from_feats is not None:
            inferred_A = A_from_feats

        if action_dim_attr is not None and inferred_A is not None and action_dim_attr != inferred_A:
            self.logger.warning(
                f"[prepare_batch] action_dim attr ({action_dim_attr}) != inferred A ({inferred_A}); using tensors as-is.")

        # ---------- convert required tensors ----------
        # obs：尽量转 float32，支持 (B, D) 或单样本 (D,)
        try:
            obs_t = torch.tensor(np.asarray(obs_np, dtype=np.float32), dtype=torch.float32, device=device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            if obs_t.shape[0] != B:
                # 尝试广播/重复
                if obs_t.shape[0] == 1:
                    obs_t = obs_t.repeat(B, 1)
                else:
                    raise ValueError(f"obs batch dim {obs_t.shape[0]} != B {B}")
        except Exception as e:
            self.logger.error(f"[prepare_batch] failed to convert obs to float32 tensor: {e}")
            raise

        try:
            actions_t = torch.tensor(np.asarray(actions_np, dtype=np.int64), dtype=torch.long, device=device).view(-1)
        except Exception as e:
            self.logger.error(f"[prepare_batch] failed to convert action to long tensor: {e}")
            raise

        try:
            rewards_t = torch.tensor(np.asarray(rewards_np, dtype=np.float32), dtype=torch.float32, device=device).view(
                -1)
        except Exception as e:
            self.logger.error(f"[prepare_batch] failed to convert reward to float32 tensor: {e}")
            raise

        try:
            dones_t = torch.tensor(np.asarray(dones_np, dtype=np.float32), dtype=torch.float32, device=device).view(-1)
        except Exception as e:
            self.logger.error(f"[prepare_batch] failed to convert done to float32 tensor: {e}")
            raise

        # ---------- optional: old_logp / old_value ----------
        def _to_float_tensor(candidate, name, fill_value=0.0):
            if candidate is None:
                return None
            try:
                arr = np.asarray(candidate)
            except Exception:
                try:
                    arr = np.array([x for x in candidate])
                except Exception:
                    self.logger.error(f"[prepare_batch] cannot convert {name} to numpy array; returning None")
                    return None
            if arr.dtype == np.dtype('O'):
                flat = []
                bad_idx = []
                for i, v in enumerate(arr):
                    try:
                        if v is None:
                            flat.append(float(fill_value))
                        else:
                            flat.append(float(v))
                    except Exception:
                        try:
                            flat.append(float(np.asarray(v).item()))
                        except Exception:
                            flat.append(float(fill_value));
                            bad_idx.append(i)
                if bad_idx:
                    self.logger.warning(
                        f"[prepare_batch] {name} contains {len(bad_idx)} non-numeric entries; sample idx: {bad_idx[:5]}")
                arr = np.asarray(flat, dtype=np.float32)
            else:
                try:
                    arr = arr.astype(np.float32)
                except Exception:
                    arr = np.array([float(x) if x is not None else fill_value for x in arr], dtype=np.float32)
            try:
                t = torch.tensor(arr, dtype=torch.float32, device=device).view(-1)
                if len(t) != B:
                    if len(t) == 1:
                        t = t.repeat(B)
                    else:
                        self.logger.warning(f"[prepare_batch] {name} batch dim {len(t)} != B {B}; keeping as-is")
                return t
            except Exception as e:
                self.logger.error(f"[prepare_batch] failed to convert {name} to tensor: {e}")
                return None

        old_logp_t = _to_float_tensor(data.get('logp', data.get('old_logp', None)), "old_logp", 0.0)
        old_value_t = _to_float_tensor(data.get('value', data.get('old_value', None)), "old_value", 0.0)

        # ---------- final dict ----------
        return {
            'obs': obs_t,  # (B, D)
            'actions': actions_t,  # (B,)
            'rewards': rewards_t,  # (B,)
            'dones': dones_t,  # (B,)
            'masks': masks_t,  # (B, A) or None
            'action_feats': action_feats_t,  # (B, A, F) or None
            'old_logp': old_logp_t,  # (B,) or None
            'old_value': old_value_t,  # (B,) or None
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
        batch contains tensors:
            obs, actions, rewards, dones,
            masks: (B,A) float add-mask (legal=0, illegal=-1e9) or None
            action_feats: (B,A,F) float (仅牌型+点数 one-hot) or None
            old_logp, old_value (optional)
        """
        device = self.device
        obs = batch['obs']  # (B, D)
        actions = batch['actions']  # (B,)
        rewards = batch['rewards']  # (B,)
        dones = batch['dones']  # (B,)
        masks = batch.get('masks', None)  # (B, A) or None
        action_feats = batch.get('action_feats', None)  # (B, A, F) or None
        old_logp = batch.get('old_logp', None)  # (B,) or None
        old_value = batch.get('old_value', None)  # (B,) or None

        # ---- 懒加载 “动作语义 → bias” 的小网络头（作用：对每个 index 生成加分/减分）----
        # 这样不用改你的 PolicyValueNet；并把参数加进已有 optimizer。
        if action_feats is not None and not hasattr(self, "action_feat_head"):
            self.action_feat_head = torch.nn.Sequential(
                torch.nn.Linear(25, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)  # 每个 index 输出一个标量 bias
            ).to(device)
            # 把新头的参数加入优化器
            try:
                self.optimizer.add_param_group({'params': self.action_feat_head.parameters()})
            except Exception:
                # 某些优化器不支持 add_param_group 时，至少给个提示
                print("[ppo_update] Warning: optimizer.add_param_group failed; "
                      "ensure action_feat_head params are optimized.")

        # ---- 一个小工具：把 logits、masks、action_feats 合成为 masked_logits ----
        def _compose_masked_logits(logits: torch.Tensor,
                                   masks: Optional[torch.Tensor],
                                   action_feats: Optional[torch.Tensor]) -> torch.Tensor:
            """
            logits: (B, A) 来自 policy_head
            masks:  (B, A) 加法掩码（legal=0, illegal=-1e9）；None 表示全合法
            action_feats: (B, A, F) 仅“牌型+点数”特征；None 表示不加语义 bias
            return: masked_logits: (B, A)
            """
            if masks is None:
                masks = torch.zeros_like(logits)
            if action_feats is not None and hasattr(self, "action_feat_head"):
                # 对每个 index 的 (F,) 生成一个 bias 标量
                bias = self.action_feat_head(action_feats).squeeze(-1)  # (B, A)
                # 只在“合法位”加 bias（非法位反正会被 -1e9 掩掉，这里再乘一次保险）
                legal = (masks == 0.0).to(logits.dtype)  # (B, A) in {0,1}
                return logits + masks + bias * legal
            else:
                return logits + masks

        # ===================== 计算当前策略的 logp / value（no_grad） =====================
        with torch.no_grad():
            logits_all, values_all = self.model(obs)  # (B, A), (B,)
            masked_logits_all = _compose_masked_logits(logits_all, masks, action_feats)  # (B, A)
            log_probs_all = torch.log_softmax(masked_logits_all, dim=-1)  # (B, A)
            # 选中动作的 logp
            curr_logp = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
            curr_values = values_all  # (B,)

        # ---- old_logp / old_value 缺省时回退到当前 ----
        if old_logp is None:
            old_logp = curr_logp.detach()
        if old_value is None:
            old_value = curr_values.detach()

        # ===================== GAE 优势 / 回报 =====================
        advantages, returns = self.compute_gae_returns(
            rewards, dones, old_value,
            last_value=old_value[-1].item() if old_value.shape[0] > 0 else 0.0
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ===================== 多 epoch / mini-batch 训练 =====================
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
                mb_old_logp = old_logp[mb_idx_t].detach()
                mb_old_val = old_value[mb_idx_t].detach()

                mb_masks = masks[mb_idx_t] if masks is not None else None
                mb_feats = action_feats[mb_idx_t] if action_feats is not None else None

                # 前向：当前策略
                logits, values = self.model(mb_obs)  # (Mb, A),(Mb,)
                masked_logits = _compose_masked_logits(logits, mb_masks, mb_feats)  # (Mb, A)

                log_probs_all = torch.log_softmax(masked_logits, dim=-1)  # (Mb, A)
                new_logp = log_probs_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)  # (Mb,)

                # PPO-clip
                ratio = torch.exp(new_logp - mb_old_logp)  # (Mb,)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.min(surr1, surr2))

                # Value loss
                value_loss = torch.mean((values - mb_returns) ** 2)

                # Entropy（基于 masked_logits 的分布）
                probs = torch.softmax(masked_logits, dim=-1)  # (Mb, A)
                entropy = -torch.sum(probs * log_probs_all, dim=-1).mean()

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                # 如果我们懒加载了 action_feat_head，也需要裁剪/优化它的梯度
                if hasattr(self, "action_feat_head"):
                    torch.nn.utils.clip_grad_norm_(self.action_feat_head.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # ===================== 事后统计 =====================
        self.updates += 1
        with torch.no_grad():
            y_pred = curr_values.detach().cpu().numpy()
            y_true = returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"[Learner] Explained Variance: {explained_var:.4f}")

        return {
            'policy_loss': float(policy_loss.detach().cpu().item()),
            'value_loss': float(value_loss.detach().cpu().item()),
            'entropy': float(entropy.detach().cpu().item()),
            'explained_var': float(explained_var)
        }

    def save(self, prefix: str = "ppo"):
        """
        Save both a full checkpoint and a model-only state_dict.
        Returns path to the 'latest' model-only file.
        """
        # full checkpoint (for resume)
        ckpt_fname = os.path.join(self.save_dir, f"{prefix}_step{self.updates}.pth")
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'updates': self.updates
        }, ckpt_fname)

        # model-only file (overwrite latest)
        model_only_fname = os.path.join(self.save_dir, f"{prefix}_step{self.updates}_model.pth")
        torch.save(self.model.state_dict(), model_only_fname)

        latest_target = os.path.join(self.save_dir, f"{prefix}_latest_model.pth")
        try:
            # atomic-ish replace (copy then move)
            shutil.copyfile(model_only_fname, latest_target)
        except Exception:
            try:
                shutil.move(model_only_fname, latest_target)
            except Exception:
                # last resort: leave model_only_fname as is
                pass

        return latest_target

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

    # --- DEBUG SNIPPET: 放在 prepare_batch(prepared) 之后，ppo_update 之前 ---
    def _debug_batch(self, prepared):
        import numpy as _np, torch as _torch
        print("=== DEBUG BATCH ===")
        print("num samples:", prepared['obs'].shape[0])
        if prepared['rewards'] is not None:
            r = prepared['rewards'].cpu().numpy()
            print("rewards: mean %.6f std %.6f min %.6f max %.6f" % (r.mean(), r.std(), r.min(), r.max()))
        if prepared['dones'] is not None:
            d = prepared['dones'].cpu().numpy()
            print("dones: unique", _np.unique(d))
        if prepared.get('old_logp') is not None:
            ol = prepared['old_logp'].cpu().numpy()
            print("old_logp: mean %.6e std %.6e" % (ol.mean(), ol.std()))
        if prepared.get('old_value') is not None:
            ov = prepared['old_value'].cpu().numpy()
            print("old_value: mean %.6e std %.6e" % (ov.mean(), ov.std()))
        print("===================")

    def train(self,
              total_updates: int = 30000,
              fetch_interval: float = 1.0,
              samples_per_update: int = 100,
              save_every: int = 20):
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
            # === 新增：把 action_space_sparse -> 稠密 masks / action_feats ===
            sparse_list = list(batch['action_space_sparse'])  # 长度=B
            masks_np, feats_np = sparse_to_dense(sparse_list)
            # 塞回 batch，让 prepare_batch 把它们转成 tensor
            batch['masks'] = masks_np  # shape (B, A), float32
            batch['action_feats'] = feats_np  # shape (B, A, FEAT_DIM), float32
            # prepare tensors
            prepared = self.prepare_batch(batch)
            if prepared is None:
                continue

            # if prepared missing old_logp or old_value, compute via current policy
            # if prepared['old_logp'] is None or prepared['old_value'] is None:
            #     with torch.no_grad():
            #         logits, values = self.model(prepared['obs'])
            #         probs, log_probs_all = masked_logits_to_probs(logits, prepared['masks'])
            #         curr_logp = gather_log_probs(log_probs_all, prepared['actions'])
            #     if prepared['old_logp'] is None:
            #         prepared['old_logp'] = curr_logp.detach()
            #     if prepared['old_value'] is None:
            #         prepared['old_value'] = values.detach()

            # call update (this function does multiple epochs internally)
            stats = self.ppo_update(prepared)
            self._debug_batch(prepared)

            print(f"[Learner] Update {self.updates}: policy_loss={stats['policy_loss']:.4f}, "
                  f"value_loss={stats['value_loss']:.4f}, entropy={stats['entropy']:.4f}, "
                  f"explained_var={stats['explained_var']:.4f}, samples={len(prepared['actions'])}")
            with open("train_result.txt", "a") as file:
                # 写入 policy_loss value_loss 和 entropy 到文件
                file.write(
                    f"TrainResult - Policy_loss: {stats['policy_loss']:.4f}, Value_loss: {stats['value_loss']:.4f}, Entropy: {stats['entropy']:.4f}\n")

            # save periodically
            if self.updates % save_every == 0:
                p = self.save(prefix="ppo")
                print(f"[Learner] Saved checkpoint: {p}")

        print("[Learner] Training finished.")
