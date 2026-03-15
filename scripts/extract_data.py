#!/usr/bin/env python3
"""Extract conversation episode data from a PostgreSQL database.

Produces CS234/data/episodes.json with structured episode data for RL training.
Falls back to synthetic data generation if the database is unavailable.
"""

import json
import os
import sys
import random
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/dbname",
)


def extract_from_postgres() -> list:
    """Pull conversation data from the prompt_logs + messages tables."""
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("[extract] psycopg2 not installed -- falling back to synthetic data")
        return []

    try:
        conn = psycopg2.connect(DB_URL)
    except Exception as e:
        print(f"[extract] Cannot connect to PostgreSQL: {e}")
        return []

    query = """
    SELECT
        pl.id                   AS log_id,
        pl.conversation_id,
        pl.message_id,
        pl.diagram_mode,
        pl.total_input_tokens,
        pl.diagram_tokens,
        pl.sources_tokens,
        pl.user_message_tokens,
        pl.persona,
        pl.bypass_rag,
        pl.rag_hits_count,
        m.ttft,
        m.feedback,
        m.role,
        m.content               AS assistant_content,
        (
            SELECT m2.content
            FROM messages m2
            WHERE m2.chat_id = m.chat_id
              AND m2.session_id = m.session_id
              AND m2.role = 'user'
              AND m2.id < m.id
            ORDER BY m2.id DESC
            LIMIT 1
        )                       AS user_content,
        c.flow_id,
        c.chat_id,
        pl.created_at
    FROM prompt_logs pl
    LEFT JOIN messages m ON m.id = pl.message_id
    LEFT JOIN chats c ON c.chat_id = pl.conversation_id
    WHERE pl.diagram_mode IS NOT NULL
    ORDER BY pl.conversation_id, pl.created_at;
    """

    rows = []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            rows = [dict(r) for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        print(f"[extract] Query failed: {e}")
        conn.close()
        return []

    if not rows:
        print("[extract] No rows with diagram_mode found.")
        return []

    episodes = _group_into_episodes(rows)
    print(f"[extract] Extracted {len(episodes)} episodes ({sum(len(e['steps']) for e in episodes)} steps) from PostgreSQL")
    return episodes


def _group_into_episodes(rows: list) -> list:
    """Group flat rows into episodes keyed by conversation_id."""
    from collections import OrderedDict

    convos = OrderedDict()
    for r in rows:
        cid = r["conversation_id"] or r.get("chat_id") or f"unknown_{r['log_id']}"
        convos.setdefault(cid, []).append(r)

    episodes = []
    for cid, steps_raw in convos.items():
        steps = []
        prev_mode = None
        for i, s in enumerate(steps_raw):
            mode = (s["diagram_mode"] or "").upper()
            step = {
                "conversation_turn": i,
                "diagram_mode": mode,
                "total_input_tokens": s["total_input_tokens"] or 0,
                "diagram_tokens": s["diagram_tokens"] or 0,
                "sources_tokens": s["sources_tokens"] or 0,
                "user_message_tokens": s["user_message_tokens"] or 0,
                "rag_hits_count": s["rag_hits_count"] or 0,
                "ttft": s["ttft"],
                "feedback": s["feedback"],
                "persona": s["persona"],
                "previous_mode": prev_mode.upper() if prev_mode else None,
                "flow_id": s["flow_id"],
                "user_content": s.get("user_content"),
                "assistant_content": s.get("assistant_content"),
            }
            steps.append(step)
            prev_mode = mode
        episodes.append({"conversation_id": cid, "steps": steps})

    return episodes


def generate_synthetic_episodes(n_episodes: int = 500, seed: int = 42) -> list:
    """Generate realistic synthetic episodes for development / fallback.

    Models a representative distribution of context mode selections:
    - ~60% BOUNDED, ~30% FULL, ~10% MINIMAL (matching a typical heuristic)
    - Token counts correlate with mode
    - TTFT correlates with token count
    - Feedback is sparse (~15% explicit)
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    mode_probs = [0.10, 0.60, 0.30]  # MINIMAL, BOUNDED, FULL
    mode_token_ranges = {
        "MINIMAL": (800, 2000),
        "BOUNDED": (1500, 4500),
        "FULL":    (3000, 13000),
    }
    mode_names = ["MINIMAL", "BOUNDED", "FULL"]

    episodes = []
    for ep_idx in range(n_episodes):
        n_turns = rng.randint(1, 8)
        steps = []
        prev_mode = None
        flow_id = rng.randint(1, 50)

        for turn in range(n_turns):
            mode = rng.choice(mode_names, p=mode_probs)
            lo, hi = mode_token_ranges[mode]
            diag_tokens = int(rng.uniform(lo, hi))
            sources_tokens = int(rng.uniform(200, 2000))
            user_tokens = int(rng.uniform(20, 200))
            total_tokens = diag_tokens + sources_tokens + user_tokens + int(rng.uniform(500, 1500))

            base_ttft = 0.5 + total_tokens * 0.0005
            ttft = max(0.2, base_ttft + rng.normal(0, 0.3))

            feedback = None
            if rng.random() < 0.15:
                if mode == "FULL":
                    feedback = rng.choice(["thumbs_up", "thumbs_down"], p=[0.75, 0.25])
                elif mode == "BOUNDED":
                    feedback = rng.choice(["thumbs_up", "thumbs_down"], p=[0.65, 0.35])
                else:
                    feedback = rng.choice(["thumbs_up", "thumbs_down"], p=[0.45, 0.55])

            steps.append({
                "conversation_turn": turn,
                "diagram_mode": mode,
                "total_input_tokens": total_tokens,
                "diagram_tokens": diag_tokens,
                "sources_tokens": sources_tokens,
                "user_message_tokens": user_tokens,
                "rag_hits_count": int(rng.uniform(0, 15)),
                "ttft": round(ttft, 3),
                "feedback": feedback,
                "persona": rng.choice(["technical", "business"]),
                "previous_mode": prev_mode,
                "flow_id": flow_id,
            })
            prev_mode = mode

        episodes.append({
            "conversation_id": f"synthetic_{ep_idx:04d}",
            "steps": steps,
        })

    total_steps = sum(len(e["steps"]) for e in episodes)
    print(f"[extract] Generated {len(episodes)} synthetic episodes ({total_steps} steps)")
    return episodes


def main():
    print("=" * 60)
    print("CS234 Data Extraction")
    print("=" * 60)

    real_episodes = extract_from_postgres()

    if not real_episodes:
        print("[extract] Using synthetic data generation as fallback...")
        episodes = generate_synthetic_episodes(n_episodes=500)
    elif len(real_episodes) < 300:
        n_synthetic = 500 - len(real_episodes)
        print(f"[extract] {len(real_episodes)} real episodes -- augmenting with {n_synthetic} synthetic for training")
        synthetic = generate_synthetic_episodes(n_episodes=n_synthetic, seed=42)
        episodes = real_episodes + synthetic
    else:
        episodes = real_episodes

    out_path = DATA_DIR / "episodes.json"
    with open(out_path, "w") as f:
        json.dump(episodes, f, indent=2, default=str)

    print(f"[extract] Saved {len(episodes)} episodes to {out_path}")

    modes = {}
    total_steps = 0
    for ep in episodes:
        for s in ep["steps"]:
            modes[s["diagram_mode"]] = modes.get(s["diagram_mode"], 0) + 1
            total_steps += 1

    print(f"\n--- Dataset Summary ---")
    print(f"Episodes: {len(episodes)}")
    print(f"Total steps: {total_steps}")
    print(f"Avg turns/episode: {total_steps / len(episodes):.1f}")
    print(f"Mode distribution:")
    for m in ["MINIMAL", "BOUNDED", "FULL"]:
        count = modes.get(m, 0)
        print(f"  {m:8s}: {count:5d} ({100 * count / total_steps:.1f}%)")


if __name__ == "__main__":
    main()
