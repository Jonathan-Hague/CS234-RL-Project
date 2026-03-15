"""Deep Q-Network (DQN) agent for context mode selection.

Uses a small MLP Q-function with experience replay and a target network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List

from .utils import NUM_ACTIONS, STATE_DIM


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN with experience replay and target network."""

    name = "DQN"

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 50,
        buffer_size: int = 10000,
        hidden: int = 64,
        seed: int = 42,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cpu")
        self.q_net = QNetwork(STATE_DIM, NUM_ACTIONS, hidden).to(self.device)
        self.target_net = QNetwork(STATE_DIM, NUM_ACTIONS, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.training_rewards: List[float] = []

    def select_action(self, observation: np.ndarray, explore: bool = False) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, NUM_ACTIONS)
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_net(state)
            return int(q_values.argmax(dim=1).item())

    def _update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_on_env(self, env, n_episodes: int = 2000, verbose: bool = True):
        """Train DQN on the environment."""
        episode_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.buffer.push(obs, action, reward, next_obs, float(done))
                self._update()

                obs = next_obs
                total_reward += reward

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if (ep + 1) % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            episode_rewards.append(total_reward)

            if verbose and (ep + 1) % 200 == 0:
                recent = np.mean(episode_rewards[-100:])
                print(
                    f"  Episode {ep + 1:5d}/{n_episodes} | "
                    f"Avg reward (last 100): {recent:+.3f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        self.training_rewards = episode_rewards
        return episode_rewards
