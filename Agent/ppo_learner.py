# ppo_learner.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import os

# ---------------------------
# 1. Actor-Critic 网络
# ---------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.policy_head = nn.Linear(256, action_dim)  # logits
        self.value_head = nn.Linear(256, 1)           # state value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value.squeeze(-1)

# ---------------------------
# 2. PPO Learner
# ---------------------------
class PPOLearner:
    def __init__(self, state_dim, max_actions, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.model = ActorCritic(state_dim, max_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 经验缓冲
        self.buffer = deque(maxlen=10000)

    # ---------------------------
    # 添加经验
    # ---------------------------
    def add_experience(self, state, action, reward, next_state, done, action_mask):
        self.buffer.append({
            'state': np.array(state, dtype=np.float32),
            'action': action,
            'reward': reward,
            'next_state': np.array(next_state, dtype=np.float32),
            'done': done,
            'action_mask': np.array(action_mask, dtype=np.float32)
        })

    # ---------------------------
    # 计算优势函数 (GAE)
    # ---------------------------
    def compute_gae(self, rewards, values, dones, next_value):
        adv = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[-1]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            adv[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        returns = adv + values
        return adv, returns

    # ---------------------------
    # PPO 更新
    # ---------------------------
    def update(self, batch_size=64, epochs=4):
        if len(self.buffer) < batch_size:
            return

        # 转为tensor
        states = torch.tensor([b['state'] for b in self.buffer], dtype=torch.float32, device=self.device)
        actions = torch.tensor([b['action'] for b in self.buffer], dtype=torch.long, device=self.device)
        rewards = np.array([b['reward'] for b in self.buffer], dtype=np.float32)
        dones = np.array([b['done'] for b in self.buffer], dtype=np.float32)
        action_masks = torch.tensor([b['action_mask'] for b in self.buffer], dtype=torch.float32, device=self.device)

        # 计算 value
        with torch.no_grad():
            _, values = self.model(states)
            values = values.cpu().numpy()
            next_value = 0.0  # 最后一条经验的 next_value 可以近似为0

        adv, returns = self.compute_gae(rewards, values, dones, next_value)
        adv = torch.tensor(adv, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # PPO 多轮更新
        for _ in range(epochs):
            logits, value_pred = self.model(states)
            # mask logits
            logits = logits.masked_fill(action_masks == 0, -1e9)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            probs = dist.probs

            # ratio
            with torch.no_grad():
                old_logits, _ = self.model(states)
                old_logits = old_logits.masked_fill(action_masks == 0, -1e9)
                old_dist = Categorical(logits=old_logits)
                old_log_probs = old_dist.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)

            # policy loss
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # value loss
            value_loss = self.value_coef * (returns - value_pred).pow(2).mean()

            # entropy bonus
            entropy_bonus = self.entropy_coef * dist.entropy().mean()

            # 总 loss
            loss = policy_loss + value_loss - entropy_bonus

            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 清空缓冲
        self.buffer.clear()

    # ---------------------------
    # 保存/加载
    # ---------------------------
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    # ---------------------------
    # forward 推断
    # ---------------------------
    def select_action(self, state, action_mask):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state_tensor)
            logits = logits.masked_fill(mask_tensor == 0, -1e9)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
        return action
