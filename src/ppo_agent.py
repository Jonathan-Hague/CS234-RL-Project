"""PPO agent wrapper using Stable-Baselines3.

Wraps the ContextSelectionEnv with SB3's PPO for policy gradient training.
"""

import numpy as np
from typing import List, Dict, Optional

from .utils import NUM_ACTIONS, STATE_DIM


class PPOAgent:
    """PPO via Stable-Baselines3.

    Wraps the gymnasium env so SB3 handles training. After training,
    select_action calls the learned policy.
    """

    name = "PPO"

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 128,
        batch_size: int = 64,
        n_epochs: int = 10,
        seed: int = 42,
    ):
        self.lr = lr
        self.gamma = gamma
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self.model = None
        self.training_rewards: List[float] = []

    def train_on_env(self, env, n_episodes: int = 2000, verbose: bool = True):
        """Train PPO on the given env.

        SB3 uses total_timesteps rather than episodes, so we estimate
        based on average episode length.
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            print("[PPO] stable-baselines3 not installed. pip install stable-baselines3")
            return []

        avg_steps = self._estimate_avg_episode_length(env, n_sample=50)
        total_timesteps = int(n_episodes * avg_steps)

        vec_env = DummyVecEnv([lambda: env])

        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=self.lr,
            gamma=self.gamma,
            n_steps=min(self.n_steps, total_timesteps),
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            seed=self.seed,
            verbose=0,
        )

        if verbose:
            print(f"  Training PPO for {total_timesteps} timesteps (~{n_episodes} episodes)...")

        self.model.learn(total_timesteps=total_timesteps)

        self.training_rewards = self._evaluate_training_curve(env, n_episodes=200)

        if verbose:
            recent = np.mean(self.training_rewards[-50:]) if self.training_rewards else 0
            print(f"  PPO training complete. Avg reward (last 50 eval): {recent:+.3f}")

        return self.training_rewards

    def select_action(self, observation: np.ndarray, explore: bool = False) -> int:
        if self.model is None:
            return np.random.randint(0, NUM_ACTIONS)
        action, _ = self.model.predict(observation, deterministic=not explore)
        return int(action)

    def _estimate_avg_episode_length(self, env, n_sample: int = 50) -> float:
        lengths = []
        for _ in range(n_sample):
            obs, _ = env.reset()
            steps = 0
            done = False
            while not done:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            lengths.append(steps)
        return np.mean(lengths)

    def _evaluate_training_curve(self, env, n_episodes: int = 200) -> List[float]:
        """Evaluate trained policy to build a rewards list for comparison."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_r = 0.0
            done = False
            while not done:
                action = self.select_action(obs, explore=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_r += reward
            rewards.append(total_r)
        return rewards
