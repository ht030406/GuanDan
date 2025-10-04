import numpy as np
import torch
import torch.nn as nn


class SimpleAgent(nn.Module):
    def __init__(self, state_dim=436, max_actions=50, hidden_size=128, device=None):
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
