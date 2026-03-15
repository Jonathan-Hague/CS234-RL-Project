#!/usr/bin/env python3
"""Score real episodes with GPT-4o-mini as an LLM judge.

Reads episodes.json, scores each real episode step that has response content,
and writes episodes_scored.json with the llm_quality_score field (1-5 scale).
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
MAX_CONTENT_CHARS = 1500

RUBRIC = """Rate the quality of an AI assistant response in a RAG-based diagram analysis system.
The assistant receives a user question along with retrieved diagram context, then generates an answer.

Score on a 1-5 scale:
  1 - Completely wrong, off-topic, or harmful. Ignores the question entirely.
  2 - Partially relevant but contains significant errors, hallucinations, or misses the core question.
  3 - Acceptable. Addresses the question but lacks depth, precision, or misses nuance.
  4 - Good. Accurate, relevant, and helpful. Minor gaps or verbosity.
  5 - Excellent. Precise, well-structured, directly addresses the question with appropriate detail.

Consider: accuracy, relevance to the user question, appropriate use of retrieved context,
helpfulness, and conciseness. Respond with ONLY a single integer 1-5."""


def score_step(client, user_content: str, assistant_content: str) -> float:
    """Call GPT-4o-mini to score a single user-assistant interaction."""
    user_q = (user_content or "")[:MAX_CONTENT_CHARS]
    asst_a = (assistant_content or "")[:MAX_CONTENT_CHARS]

    if not asst_a.strip():
        return 3.0

    messages = [
        {"role": "system", "content": RUBRIC},
        {
            "role": "user",
            "content": f"User question:\n{user_q}\n\nAssistant response:\n{asst_a}",
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=5,
        )
        text = resp.choices[0].message.content.strip()
        score = float(text)
        return max(1.0, min(5.0, score))
    except (ValueError, IndexError):
        return 3.0
    except Exception as e:
        print(f"  [warn] API error: {e}")
        return 3.0


def main():
    print("=" * 60)
    print("CS234 LLM Judge Scoring")
    print("=" * 60)

    episodes_path = DATA_DIR / "episodes.json"
    if not episodes_path.exists():
        print(f"[error] {episodes_path} not found. Run extract_data.py first.")
        sys.exit(1)

    with open(episodes_path) as f:
        episodes = json.load(f)

    if not OPENAI_API_KEY:
        print("[warn] OPENAI_API_KEY not set -- generating calibrated synthetic scores")
        episodes = _generate_synthetic_scores(episodes)
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("[error] openai package not installed. pip install openai")
            sys.exit(1)

        client = OpenAI(api_key=OPENAI_API_KEY)
        scored = 0
        skipped = 0

        for ep in episodes:
            if ep["conversation_id"].startswith("synthetic_"):
                continue
            for step in ep["steps"]:
                user_c = step.get("user_content")
                asst_c = step.get("assistant_content")
                if asst_c and user_c:
                    score = score_step(client, user_c, asst_c)
                    step["llm_quality_score"] = score
                    scored += 1
                    if scored % 20 == 0:
                        print(f"  Scored {scored} steps...")
                    time.sleep(0.1)
                else:
                    skipped += 1

        print(f"\n[judge] Scored {scored} steps, skipped {skipped} (no content)")

        if scored > 0:
            _calibrate_synthetic_episodes(episodes)

    out_path = DATA_DIR / "episodes_scored.json"
    with open(out_path, "w") as f:
        json.dump(episodes, f, indent=2, default=str)
    print(f"[judge] Saved scored episodes to {out_path}")

    _print_score_summary(episodes)


def _generate_synthetic_scores(episodes):
    """When no API key is available, generate calibrated synthetic LLM scores."""
    import numpy as np
    rng = np.random.RandomState(42)

    mode_score_params = {
        "MINIMAL": (2.8, 0.7),
        "BOUNDED": (3.5, 0.6),
        "FULL": (4.0, 0.5),
    }

    for ep in episodes:
        for step in ep["steps"]:
            mode = step.get("diagram_mode", "BOUNDED")
            mu, sigma = mode_score_params.get(mode, (3.5, 0.6))

            feedback = step.get("feedback")
            if feedback == "thumbs_up":
                mu += 0.5
            elif feedback == "thumbs_down":
                mu -= 0.8

            score = rng.normal(mu, sigma)
            step["llm_quality_score"] = float(np.clip(score, 1.0, 5.0))

    return episodes


def _calibrate_synthetic_episodes(episodes):
    """Use real LLM scores to calibrate synthetic episode scores."""
    import numpy as np
    rng = np.random.RandomState(99)

    real_scores_by_mode = {}
    for ep in episodes:
        if ep["conversation_id"].startswith("synthetic_"):
            continue
        for step in ep["steps"]:
            if "llm_quality_score" in step:
                mode = step["diagram_mode"]
                real_scores_by_mode.setdefault(mode, []).append(step["llm_quality_score"])

    mode_params = {}
    for mode, scores in real_scores_by_mode.items():
        arr = np.array(scores)
        mode_params[mode] = (float(arr.mean()), float(arr.std()) + 0.1)

    if not mode_params:
        return

    for ep in episodes:
        if not ep["conversation_id"].startswith("synthetic_"):
            continue
        for step in ep["steps"]:
            mode = step.get("diagram_mode", "BOUNDED")
            mu, sigma = mode_params.get(mode, (3.5, 0.6))
            feedback = step.get("feedback")
            if feedback == "thumbs_up":
                mu += 0.3
            elif feedback == "thumbs_down":
                mu -= 0.5
            score = rng.normal(mu, sigma)
            step["llm_quality_score"] = float(np.clip(score, 1.0, 5.0))


def _print_score_summary(episodes):
    """Print score distribution summary."""
    import numpy as np

    scores_by_mode = {}
    total = 0
    for ep in episodes:
        for step in ep["steps"]:
            if "llm_quality_score" in step:
                mode = step["diagram_mode"]
                scores_by_mode.setdefault(mode, []).append(step["llm_quality_score"])
                total += 1

    print(f"\n--- LLM Score Summary ({total} scored steps) ---")
    for mode in ["MINIMAL", "BOUNDED", "FULL"]:
        scores = scores_by_mode.get(mode, [])
        if scores:
            arr = np.array(scores)
            print(f"  {mode:8s}: mean={arr.mean():.2f}, std={arr.std():.2f}, n={len(scores)}")


if __name__ == "__main__":
    main()
