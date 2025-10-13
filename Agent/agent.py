import numpy as np
import torch
import torch.nn as nn
from typing import Dict


class SimpleAgent(nn.Module):
    def __init__(self, state_dim=436, max_actions=1000, hidden_size=128, device=None):
        super(SimpleAgent, self).__init__()
        self.state_dim = int(state_dim)
        self.max_actions = int(max_actions)
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        # 建立网络：注意第一层输入维度必须等于 state_dim
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.max_actions)
        ).to(self.device)

    def forward(self, state, action_mask=None, return_logits=False, debug=False):
        """
        state: 1D array-like of length state_dim (e.g. (436,) numpy array)
               or a torch tensor of shape (state_dim,) or (1, state_dim).
        action_mask: 1D array-like of length max_actions (0/1), or torch tensor.
        return_logits: 如果 True 返回 logits 以及选定动作 index
        """
        # --- 归一化输入到 tensor 并确保形状为 (1, state_dim) ---
        if isinstance(state, np.ndarray) or isinstance(state, list):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.to(self.device).float()
        else:
            raise TypeError("state must be numpy array, list or torch.Tensor")

        # 将任意形状标准化为 (1, state_dim)
        if state_tensor.dim() == 0:
            state_tensor = state_tensor.view(1, -1)
        elif state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)        # (1, state_dim)
        elif state_tensor.dim() == 2:
            if state_tensor.shape[1] == 1 and state_tensor.shape[0] == self.state_dim:
                # (state_dim,1) -> 转置为 (1,state_dim)
                state_tensor = state_tensor.t().view(1, -1)
            # else 假定已经 (1, state_dim) 或 (batch, state_dim)
        else:
            raise ValueError(f"Unsupported state tensor ndim={state_tensor.dim()}")

        if state_tensor.shape[1] != self.state_dim:
            raise ValueError(f"State dimension mismatch: got {state_tensor.shape[1]}, expected {self.state_dim}")

        # --- forward through network ---
        logits = self.net(state_tensor)      # shape (1, max_actions)

        # --- 处理 mask ---
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray) or isinstance(action_mask, list):
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).view(1, -1)
            elif isinstance(action_mask, torch.Tensor):
                mask_tensor = action_mask.to(self.device).float().view(1, -1)
            else:
                raise TypeError("action_mask must be numpy array, list or torch.Tensor")

            if mask_tensor.shape[1] != self.max_actions:
                raise ValueError(f"Action mask length mismatch: got {mask_tensor.shape[1]}, expected {self.max_actions}")

            # 给非法动作一个很大的负值，使其概率趋近于0
            logits = logits.masked_fill(mask_tensor == 0, -1e9)

        probs = torch.softmax(logits, dim=-1)   # shape (1, max_actions)
        action_idx = torch.argmax(probs, dim=-1).item()

        if debug:
            print(f"[DEBUG] state.shape={state_tensor.shape}, logits.shape={logits.shape}, action={action_idx}")

        if return_logits:
            return action_idx, logits.detach().cpu().numpy().ravel()

        return action_idx


# agents/ppo_agent.py
"""
PPOAgent: policy-value network for discrete action spaces suitable for PPO.

Usage (inference):
    agent = PPOAgent(state_dim=436, action_dim=50, device='cpu')
    action_idx = agent(state_vector, action_mask)   # returns greedy argmax action (int)

Usage (for training/inference with sampling or to get logp/value):
    a, logp, value = agent.select_action(state_vector, action_mask, deterministic=False)

Save / load:
    agent.save_weights("checkpoints/ppo_latest_model.pth")
    agent.load_weights("checkpoints/ppo_latest_model.pth")
"""

from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPOAgent(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 feat_dim: int = 25,
                 hidden_sizes: Optional[Tuple[int, int, int, int]] = (256, 256, 128, 64),
                 device: Optional[Union[str, torch.device]] = None):
        """
        Args:
            state_dim: input dimension of state vector
            action_dim: number of discrete actions (max_actions)
            hidden_sizes: tuple of 4 hidden layer sizes (defaults shown)
            device: 'cpu' or 'cuda' or torch.device
        """
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        # Build feed-forward backbone with 4 hidden layers
        hs = list(hidden_sizes)
        assert len(hs) == 4, "hidden_sizes must be a 4-tuple"
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hs[0]),
            nn.ReLU(),
            nn.Linear(hs[0], hs[1]),
            nn.ReLU(),
            nn.Linear(hs[1], hs[2]),
            nn.ReLU(),
            nn.Linear(hs[2], hs[3]),
            nn.ReLU()
        )

        # heads
        self.policy_head = nn.Linear(hs[3], self.action_dim)
        self.value_head = nn.Linear(hs[3], 1)
        # 把 (type+rank) 特征映射成一个“对该 index 的额外加分/减分”
        self.action_feat_head = nn.Sequential(
            nn.Linear(feat_dim, hs[3] // 2),
            nn.ReLU(),
            nn.Linear(hs[3] // 2, 1)  # -> 标量 bias
        )

        # move to device
        self.to(self.device)
        self.eval()

    # ---------- helpers ----------
    def _ensure_tensor(self, x):
        if torch.is_tensor(x):
            return x if x.ndim > 1 else x.unsqueeze(0)
        x = torch.as_tensor(x, dtype=torch.float32)
        return x if x.ndim > 1 else x.unsqueeze(0)

    def _mask_logits(self, logits: torch.Tensor, action_mask: Optional[Union[np.ndarray, list, torch.Tensor]]):
        """
        Apply mask to logits: illegal actions set to -1e9 so softmax ~ 0.
        Accepts 1D mask or 2D mask. Returns masked_logits tensor on same device.
        """
        if action_mask is None:
            return logits
        if not isinstance(action_mask, torch.Tensor):
            mask = torch.tensor(np.asarray(action_mask), dtype=torch.bool, device=logits.device)
        else:
            mask = action_mask.to(logits.device).bool()

        # Expand 1D mask to batch if needed
        if mask.dim() == 1:
            if mask.shape[0] != logits.shape[1]:
                raise ValueError(f"Mask length {mask.shape[0]} doesn't match action_dim {logits.shape[1]}")
            mask = mask.unsqueeze(0).expand(logits.shape[0], -1)
        elif mask.dim() == 2:
            if mask.shape[0] != logits.shape[0] or mask.shape[1] != logits.shape[1]:
                # allow (1, A) -> expand
                if mask.shape[0] == 1 and mask.shape[1] == logits.shape[1]:
                    mask = mask.expand(logits.shape[0], -1)
                else:
                    raise ValueError(f"Mask shape {tuple(mask.shape)} incompatible with logits shape {tuple(logits.shape)}")

        # mask is bool (B, A)
        inf = -1e9
        masked = logits.masked_fill(~mask, inf)
        # fallback for rows with all False: keep original logits to avoid NaNs
        row_has_valid = mask.any(dim=-1)
        if not row_has_valid.all().item():
            masked[~row_has_valid] = logits[~row_has_valid]
        return masked

    def _normalize_mask_and_feat(
            self,action_mask: Optional[Union[np.ndarray, list, torch.Tensor, Dict]],
            logits: torch.Tensor,
            illegal_value: float = -1e9,
    ):
        """
        统一输入成：(B,A) 的加法掩码 和 (B,A,F) 的特征（或 None）
         - 支持传入：
            * None：无掩码、无特征
            * 仅掩码：(A,) 或 (B,A)，bool 或 float
            * 语义 bundle：{'mask':(A,), 'feat':(A,F)} 或批量版
        """
        B, A = logits.size(0), logits.size(1)
        device = logits.device

        # 默认（无掩码、无特征）
        add_mask = torch.zeros((B, A), dtype=torch.float32, device=device)
        feat = None

        if action_mask is None:
            return add_mask, feat

        # 语义 bundle
        if isinstance(action_mask, dict):
            m = action_mask.get('mask', None)
            f = action_mask.get('feat', None)

            # 处理 mask
            if m is not None:
                m = torch.as_tensor(m, dtype=torch.float32, device=device)
                if m.ndim == 1:
                    m = m.unsqueeze(0).expand(B, -1)  # (B,A)
                elif m.ndim == 2 and m.size(0) == 1:
                    m = m.expand(B, -1)
                elif m.ndim != 2 or m.size(0) != B or m.size(1) != A:
                    raise ValueError(f"mask shape must be (A,) or (B,A), got {m.shape}")
                add_mask = m

            # 处理 feat
            if f is not None:
                f = torch.as_tensor(f, dtype=torch.float32, device=device)  # (A,F) or (B,A,F)
                if f.ndim == 2:
                    f = f.unsqueeze(0).expand(B, -1, -1)  # (B,A,F)
                elif f.ndim == 3 and f.size(0) == 1:
                    f = f.expand(B, -1, -1)
                elif f.ndim != 3 or f.size(0) != B or f.size(1) != A:
                    raise ValueError(f"feat shape must be (A,F) or (B,A,F), got {f.shape}")
                feat = f
            return add_mask, feat

        # 仅掩码（bool/float）
        m = torch.as_tensor(action_mask, device=device)
        if m.dtype == torch.bool:
            if m.ndim == 1:
                m = m.unsqueeze(0).expand(B, -1)
            elif m.ndim == 2 and m.size(0) == 1:
                m = m.expand(B, -1)
            add_mask = torch.full_like(m, fill_value=illegal_value, dtype=torch.float32)
            add_mask[m] = 0.0
        else:
            if m.ndim == 1:
                m = m.unsqueeze(0).expand(B, -1)
            elif m.ndim == 2 and m.size(0) == 1:
                m = m.expand(B, -1)
            add_mask = m.to(dtype=torch.float32)
        return add_mask, feat
    # ---------- forward --------------
    def forward(self, state: Union[np.ndarray, list, torch.Tensor],
                action_mask: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
                return_logits: bool = False):
        """
        Inference forward pass. Returns greedy action index (int) by default.
        If return_logits=True, returns (action_idx, logits_np) for debugging.

        Compatibility: this method allows the same calling style as your old SimpleAgent:
            action_index = agent(state, action_mask)
        """
        s = self._ensure_tensor(state)  # (1, D) or (B, D)
        h = self.net(s)
        logits = self.policy_head(h)  # (B, action_dim)
        value = self.value_head(h).squeeze(-1)  # (B,)

        # 统一成 (B,A) 的加法掩码 和 (B,A,F) 的语义特征
        add_mask, feat = self._normalize_mask_and_feat(action_mask, logits, illegal_value=-1e9)

        if feat is not None:
            # 将 (B,A,F) 逐 index 做一个小 MLP，得到 (B,A,1) 的 bias，再 squeeze -> (B,A)
            bias = self.action_feat_head(feat).squeeze(-1)
            # 只对合法位生效（非法位即便有 bias 也会被 -1e9 掩掉，这里可选做一层“保险”）
            legal = (add_mask == 0.0).to(bias.dtype)
            masked_logits = logits + add_mask + bias * legal
        else:
            masked_logits = logits + add_mask

        # 兜底：如果某行全是 -inf（极端），回退到未掩码 logits
        finite_any = torch.isfinite(masked_logits).any(dim=-1)
        if not torch.all(finite_any):
            bad = (~finite_any).nonzero(as_tuple=False).flatten()
            masked_logits[bad] = logits[bad]

        # 贪心：谁分高选谁（训练或采样才需要 softmax 概率）
        action_idx = torch.argmax(masked_logits, dim=-1)
        if action_idx.size(0) == 1:
            action_idx = int(action_idx.item())
        else:
            action_idx = action_idx.cpu().numpy()

        if return_logits:
            return action_idx, masked_logits.detach().cpu().numpy()
        return action_idx

    # ---------- sampling interface for training --------------
    def select_action(self, state: Union[np.ndarray, list, torch.Tensor],
                      action_mask: Optional[Union[np.ndarray, list, torch.Tensor]] = None,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Sample action from policy (Categorical) respecting action_mask. Returns:
            action_idx (int), logp (float), value (float)
        If deterministic=True, uses argmax instead of sampling.
        """
        s = self._ensure_tensor(state)  # (1, D) or (B,D) - we assume single sample typically
        h = self.net(s)
        logits = self.policy_head(h)  # (1, A)
        value = self.value_head(h).squeeze(-1)  # (1,)

        masked_logits = self._mask_logits(logits, action_mask)
        probs = F.softmax(masked_logits, dim=-1)
        # sample
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        # log prob
        # if sampled via dist, compute log prob; else compute log prob of selected action
        logp = torch.log(torch.clamp(probs, 1e-12, 1.0)).gather(1, action.unsqueeze(1)).squeeze(1)

        # return scalars
        return int(action.item()), float(logp.item()), float(value.item())

    # ---------- value only --------------
    def value(self, state: Union[np.ndarray, list, torch.Tensor]) -> float:
        s = self._ensure_tensor(state)
        with torch.no_grad():
            h = self.net(s)
            v = self.value_head(h).squeeze(-1)
        return float(v.item())

    # ---------- save / load ----------
    def save_weights(self, path: str):
        """
        Save model state_dict to path (torch.save of state_dict).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: Optional[Union[str, torch.device]] = None):
        """
        Load model weights. Accepts either:
         - a state_dict saved by torch.save(state_dict)
         - a checkpoint dict with key 'model_state'
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location=map_location or self.device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        else:
            state_dict = ckpt
        # load into model (allow partial load with strict=False as a safe option)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            # try non-strict load for compatibility
            self.load_state_dict(state_dict, strict=False)
        self.to(self.device)
        self.eval()
        return self
