"""Shared utilities for CS234 RL context selection project."""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


MODE_MAP = {
    "MINIMAL": 0, "BOUNDED": 1, "FULL": 2,
    "minimal": 0, "bounded": 1, "full": 2,
}
MODE_NAMES = ["MINIMAL", "BOUNDED", "FULL"]
NUM_ACTIONS = len(MODE_NAMES)
STATE_DIM = 7


def compute_reward(
    feedback: Optional[str],
    total_tokens: int,
    ttft: Optional[float],
    alpha_token: float = 0.0002,
    alpha_ttft: float = 0.05,
    llm_quality_score: Optional[float] = None,
) -> float:
    """Compute scalar reward from a single interaction step.

    If llm_quality_score (1-5 scale) is provided, it takes precedence over
    binary feedback for the quality component: maps linearly to [-1, +1].
    Otherwise falls back to explicit feedback / weak positive default.
    """
    if llm_quality_score is not None:
        quality = (llm_quality_score - 3.0) / 2.0
    elif feedback == "thumbs_up":
        quality = 1.0
    elif feedback == "thumbs_down":
        quality = -1.0
    else:
        quality = 0.3

    token_penalty = -alpha_token * total_tokens
    ttft_penalty = -alpha_ttft * ttft if ttft and ttft > 0 else 0.0

    return quality + token_penalty + ttft_penalty


def extract_state_features(step: Dict[str, Any]) -> np.ndarray:
    """Extract a fixed-length feature vector from one episode step.

    Features (7-dim):
        0: diagram_token_ratio  (diagram_tokens / max(total_tokens, 1))
        1: conversation_turn    (normalised by 20)
        2: rag_hits_normalised  (rag_hits_count / 20)
        3: is_minimal           (one-hot previous mode)
        4: is_bounded
        5: is_full
        6: has_feedback         (1 if explicit feedback exists)
    """
    total = max(step.get("total_input_tokens", 1), 1)
    diag = step.get("diagram_tokens", 0) or 0
    turn = step.get("conversation_turn", 0) or 0
    rag_hits = step.get("rag_hits_count", 0) or 0
    prev_mode = step.get("previous_mode", None)
    feedback = step.get("feedback", None)

    mode_onehot = [0.0, 0.0, 0.0]
    if prev_mode in MODE_MAP:
        mode_onehot[MODE_MAP[prev_mode]] = 1.0

    return np.array(
        [
            diag / total,
            min(turn / 20.0, 1.0),
            min(rag_hits / 20.0, 1.0),
            mode_onehot[0],
            mode_onehot[1],
            mode_onehot[2],
            1.0 if feedback in ("thumbs_up", "thumbs_down") else 0.0,
        ],
        dtype=np.float32,
    )


def discretize_state(features: np.ndarray, bins: int = 5) -> Tuple:
    """Discretize continuous features into bin indices for tabular Q-Learning."""
    clipped = np.clip(features, 0.0, 1.0)
    indices = np.minimum((clipped * bins).astype(int), bins - 1)
    return tuple(indices.tolist())
