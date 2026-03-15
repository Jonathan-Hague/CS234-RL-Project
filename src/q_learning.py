"""Tabular Q-Learning agent for context mode selection."""

import numpy as np
from collections import defaultdict
from typing import Tuple, Dict, List

from .utils import discretize_state, NUM_ACTIONS


class QLearningAgent:
    """Tabular Q-Learning with epsilon-greedy exploration.

    State is discretized into bins for tabular lookup.
    """

    name = "Q-Learning"

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        n_bins: int = 5,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins
        self._rng = np.random.RandomState(seed)

        self.q_table: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS)
        )
        self.training_rewards: List[float] = []

    def _discretize(self, obs: np.ndarray) -> Tuple:
        return discretize_state(obs, self.n_bins)

    def select_action(self, observation: np.ndarray, explore: bool = False) -> int:
        if explore and self._rng.random() < self.epsilon:
            return self._rng.randint(0, NUM_ACTIONS)
        state = self._discretize(observation)
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)

        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def train_on_env(self, env, n_episodes: int = 1000, verbose: bool = True):
        """Train the agent on the environment for n_episodes."""
        episode_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                self.update(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

            self.decay_epsilon()
            episode_rewards.append(total_reward)

            if verbose and (ep + 1) % 200 == 0:
                recent = np.mean(episode_rewards[-100:])
                print(
                    f"  Episode {ep + 1:5d}/{n_episodes} | "
                    f"Avg reward (last 100): {recent:+.3f} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Q-table size: {len(self.q_table)}"
                )

        self.training_rewards = episode_rewards
        return episode_rewards

    def get_policy_stats(self) -> Dict:
        """Summarise the learned policy."""
        action_counts = defaultdict(int)
        for state, q_vals in self.q_table.items():
            best = int(np.argmax(q_vals))
            action_counts[best] += 1
        total = sum(action_counts.values()) or 1
        return {
            f"pct_{k}": 100 * v / total
            for k, v in sorted(action_counts.items())
        }
