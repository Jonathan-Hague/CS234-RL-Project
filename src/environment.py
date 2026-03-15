"""Custom Gymnasium environment for context selection MDP.

Replays extracted conversation episodes. The agent chooses a context mode
(MINIMAL / BOUNDED / FULL) at each turn, and receives a reward based on
the counterfactual: what would have happened if that mode had been used?

Two quality estimation strategies:
  1. LLM-scored: when episodes contain llm_quality_score, use calibrated
     quality scores per mode (observed score for the actual mode, estimated
     for counterfactuals via a mode-dependent offset model).
  2. Probabilistic fallback: token-cost scaling + logistic quality model.
"""

import json
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pathlib import Path
from typing import List, Dict, Any, Optional

from .utils import (
    MODE_MAP, MODE_NAMES, NUM_ACTIONS, STATE_DIM,
    compute_reward, extract_state_features,
)

TOKEN_MULTIPLIERS = {
    0: 0.15,   # MINIMAL: ~15% of FULL tokens
    1: 0.45,   # BOUNDED: ~45% of FULL tokens
    2: 1.00,   # FULL: 100% baseline
}

QUALITY_PROBS = {
    0: 0.45,   # MINIMAL: lower quality probability
    1: 0.65,   # BOUNDED: moderate
    2: 0.78,   # FULL: highest quality
}

QUALITY_OFFSETS = {
    0: -0.8,
    1:  0.0,
    2: +0.5,
}


class ContextSelectionEnv(gym.Env):
    """Offline replay environment for context mode selection.

    Each reset() picks a conversation episode. Each step() advances one turn.
    The agent's action (mode choice) produces a counterfactual reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, episodes: List[Dict[str, Any]], seed: int = 42):
        super().__init__()
        self.episodes = episodes
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self._rng = np.random.RandomState(seed)
        self._current_episode = None
        self._current_step_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self._rng.randint(0, len(self.episodes))
        self._current_episode = self.episodes[idx]
        self._current_step_idx = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        step_data = self._current_episode["steps"][self._current_step_idx]

        reward = self._counterfactual_reward(step_data, action)

        self._current_step_idx += 1
        terminated = self._current_step_idx >= len(self._current_episode["steps"])
        truncated = False

        obs = self._get_obs() if not terminated else np.zeros(STATE_DIM, dtype=np.float32)

        info = {
            "actual_mode": step_data["diagram_mode"],
            "chosen_mode": MODE_NAMES[action],
            "actual_tokens": step_data["total_input_tokens"],
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        if self._current_step_idx >= len(self._current_episode["steps"]):
            return np.zeros(STATE_DIM, dtype=np.float32)
        step = self._current_episode["steps"][self._current_step_idx]
        return extract_state_features(step)

    def _counterfactual_reward(self, step: Dict, action: int) -> float:
        """Estimate reward for a hypothetical mode choice.

        Uses observed data when the action matches the actual mode.
        For counterfactuals: only diagram tokens scale with mode (user message
        and RAG tokens are mode-independent). Quality is estimated via the
        LLM-calibrated offset model when scores are available.
        """
        actual_action = MODE_MAP.get(step["diagram_mode"], 1)
        actual_tokens = step["total_input_tokens"]
        actual_diag = step.get("diagram_tokens", 0) or 0
        actual_ttft = step.get("ttft") or 0.0
        llm_score = step.get("llm_quality_score")

        if action == actual_action:
            return compute_reward(
                step["feedback"], actual_tokens, actual_ttft,
                llm_quality_score=llm_score,
            )

        diag_ratio = TOKEN_MULTIPLIERS[action] / max(TOKEN_MULTIPLIERS[actual_action], 0.01)
        est_diag = int(actual_diag * diag_ratio)
        non_diag = actual_tokens - actual_diag
        est_tokens = non_diag + est_diag
        est_ttft = actual_ttft * (est_tokens / max(actual_tokens, 1)) if actual_ttft else None

        if llm_score is not None:
            offset = QUALITY_OFFSETS[action] - QUALITY_OFFSETS[actual_action]
            est_score = np.clip(llm_score + offset, 1.0, 5.0)
            noise = self._rng.normal(0, 0.25)
            est_score = np.clip(est_score + noise, 1.0, 5.0)
            return compute_reward(
                step["feedback"], est_tokens, est_ttft,
                llm_quality_score=float(est_score),
            )

        quality_prob = QUALITY_PROBS[action]
        if self._rng.random() < quality_prob:
            feedback = "thumbs_up" if step.get("feedback") == "thumbs_up" else None
        else:
            feedback = "thumbs_down"

        return compute_reward(feedback, est_tokens, est_ttft)


def load_episodes(path: Optional[str] = None) -> List[Dict]:
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "data" / "episodes.json")
    with open(path) as f:
        return json.load(f)


def make_env(episodes=None, seed=42) -> ContextSelectionEnv:
    if episodes is None:
        episodes = load_episodes()
    return ContextSelectionEnv(episodes, seed=seed)
