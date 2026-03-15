"""Baseline policies for context mode selection."""

import numpy as np
from typing import Optional


class BasePolicy:
    """Base class for context selection policies."""

    name: str = "base"

    def select_action(self, observation: np.ndarray) -> int:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class RandomPolicy(BasePolicy):
    """Uniform random over {MINIMAL, BOUNDED, FULL}."""

    name = "Random"

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def select_action(self, observation: np.ndarray) -> int:
        return self._rng.randint(0, 3)


class AlwaysMinimalPolicy(BasePolicy):
    """Always select MINIMAL mode (action=0)."""

    name = "Always-MINIMAL"

    def select_action(self, observation: np.ndarray) -> int:
        return 0


class AlwaysFullPolicy(BasePolicy):
    """Always select FULL mode (action=2)."""

    name = "Always-FULL"

    def select_action(self, observation: np.ndarray) -> int:
        return 2


class HeuristicPolicy(BasePolicy):
    """Rule-based context mode selection policy.

    Observation features (from utils.extract_state_features):
        [0] diagram_token_ratio
        [1] conversation_turn (normalised)
        [2] rag_hits_normalised
        [3] is_minimal (prev mode one-hot)
        [4] is_bounded
        [5] is_full
        [6] has_feedback

    Heuristic rules:
        - If first turn (turn~0) and no previous mode -> FULL (overview)
        - If diagram_token_ratio is low (< 0.1) -> FULL (need more context)
        - Otherwise -> BOUNDED (focused subgraph)
        - MINIMAL is rarely used (only when explicitly requested, which we can't
          detect from these features alone)
    """

    name = "Heuristic"

    def select_action(self, observation: np.ndarray) -> int:
        turn_norm = observation[1]
        diag_ratio = observation[0]
        prev_is_minimal = observation[3]
        prev_is_bounded = observation[4]
        prev_is_full = observation[5]
        has_prev_mode = prev_is_minimal + prev_is_bounded + prev_is_full > 0.5

        if turn_norm < 0.05 and not has_prev_mode:
            return 2  # FULL on first turn

        if diag_ratio < 0.1:
            return 2  # FULL when diagram ratio is low

        return 1  # BOUNDED otherwise
