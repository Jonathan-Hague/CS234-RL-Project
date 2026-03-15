"""LinUCB contextual bandit for context mode selection.

Treats each turn independently (no sequential modeling).
Each action (mode) has a linear model: reward ~ theta_a^T x + alpha * uncertainty.
"""

import numpy as np
from typing import Dict, List, Optional

from .utils import NUM_ACTIONS, STATE_DIM, MODE_NAMES


class LinUCBAgent:
    """LinUCB with disjoint linear models per arm.

    At each step, selects the arm (mode) that maximises:
        UCB_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

    where A_a = sum(x_t x_t^T) + I and b_a = sum(r_t x_t).
    """

    name = "LinUCB"

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self.alpha = alpha
        self._rng = np.random.RandomState(seed)
        self.d = STATE_DIM

        self.A = [np.eye(self.d) for _ in range(NUM_ACTIONS)]
        self.b = [np.zeros(self.d) for _ in range(NUM_ACTIONS)]
        self.A_inv = [np.eye(self.d) for _ in range(NUM_ACTIONS)]
        self.training_rewards: List[float] = []

    def select_action(self, observation: np.ndarray, explore: bool = True) -> int:
        x = observation.astype(np.float64)
        ucb_values = np.zeros(NUM_ACTIONS)

        for a in range(NUM_ACTIONS):
            theta = self.A_inv[a] @ self.b[a]
            pred = theta @ x
            uncertainty = np.sqrt(x @ self.A_inv[a] @ x)
            ucb_values[a] = pred + (self.alpha * uncertainty if explore else 0.0)

        return int(np.argmax(ucb_values))

    def update(self, observation: np.ndarray, action: int, reward: float):
        x = observation.astype(np.float64)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
        self.A_inv[action] = np.linalg.inv(self.A[action])

    def train_on_env(self, env, n_episodes: int = 1000, verbose: bool = True):
        """Train the bandit on environment episodes (treating each step independently)."""
        episode_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                self.update(obs, action, reward)
                obs = next_obs
                total_reward += reward

            episode_rewards.append(total_reward)

            if verbose and (ep + 1) % 200 == 0:
                recent = np.mean(episode_rewards[-100:])
                print(
                    f"  Episode {ep + 1:5d}/{n_episodes} | "
                    f"Avg reward (last 100): {recent:+.3f}"
                )

        self.training_rewards = episode_rewards
        return episode_rewards

    def get_policy_stats(self) -> Dict:
        """Show learned theta vectors per arm."""
        stats = {}
        for a in range(NUM_ACTIONS):
            theta = self.A_inv[a] @ self.b[a]
            stats[MODE_NAMES[a]] = {
                "theta_norm": float(np.linalg.norm(theta)),
                "theta": theta.tolist(),
            }
        return stats
