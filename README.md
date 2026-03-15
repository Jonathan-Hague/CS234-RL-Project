# CS234: RL-Based Context Selection for RAG Systems

**Course**: CS234: Reinforcement Learning, Stanford University, Winter 2026

## Overview

This project applies reinforcement learning to optimize context selection in a
retrieval-augmented generation (RAG) system. The system must choose between three
context modes — MINIMAL, BOUNDED, and FULL — when answering user questions about
diagram-based knowledge graphs. Each mode trades off response quality against token
cost and latency.

The agent is trained offline on real conversation episodes and evaluated against
several baselines including rule-based heuristics, tabular Q-Learning, DQN, PPO,
and contextual bandits (LinUCB).

### Key Results (from `data/final_results.json`)

| Policy | Mean Reward | Mean Tokens/Turn |
|---|---|---|
| Always-FULL | -5.19 | 6,107 |
| Heuristic | -4.03 | 5,917 |
| Random | -3.60 | 6,006 |
| Q-Learning | -3.17 | 5,871 |
| Always-MINIMAL | -3.10 | 5,744 |
| LinUCB | -3.04 | 6,080 |
| **DQN** | **-2.82** | **5,362** |

DQN achieves the best reward while also reducing token usage by ~12% compared to
Always-FULL, demonstrating that RL can learn a more efficient context selection
policy than the baseline heuristic.

---

## Repository Structure

```
CS234/
├── src/                        # Core RL environment and agent code
│   ├── environment.py          # Gymnasium environment (offline episode replay)
│   ├── utils.py                # Reward function, feature extraction, MDP constants
│   ├── baselines.py            # Baseline policies (Random, Heuristic, Always-*)
│   ├── q_learning.py           # Tabular Q-Learning agent
│   ├── dqn.py                  # Deep Q-Network agent
│   ├── ppo_agent.py            # PPO agent (via Stable-Baselines3)
│   ├── contextual_bandits.py   # LinUCB contextual bandit
│   └── __init__.py
│
├── scripts/                    # Data pipeline and experiment runners
│   ├── extract_data.py         # Pull episodes from PostgreSQL or generate synthetic data
│   ├── llm_judge.py            # Score episodes with GPT-4o-mini (produces llm_quality_score)
│   ├── run_experiments.py      # Run all policies and save results
│   ├── run_final_experiments.py # Full experiment suite with ablations and figures
│   └── sanitize_data.py        # Sanitize data files for public release (see below)
│
├── data/
│   ├── episodes.json           # Sanitized training episodes (500 episodes, 1,632 steps)
│   ├── episodes_scored.json    # Same episodes with LLM quality scores added
│   ├── results.json            # Intermediate experiment results
│   └── final_results.json      # Final policy comparison + ablation results
│
├── figures/                    # Generated plots (produced by run_final_experiments.py)
│   ├── learning_curves_all.png
│   ├── reward_comparison_final.png
│   ├── mode_distributions_final.png
│   ├── ablation_features.png
│   ├── ablation_reward_coefficients.png
│   └── ablation_data_source.png
│
├── CS234_FINAL.tex/.pdf        # Final report (ICML format, 8 pages)
├── CS234_MILESTONE.tex/.pdf    # Milestone report
├── CS234_SLIDES.tex/.pdf       # Presentation slides
├── CS234_POSTER.tex/.pdf       # Poster session
├── CS234_POSTER_V2.tex/.pdf    # Revised poster
│
├── algorithm.sty               # LaTeX style files (ICML)
├── algorithmic.sty
├── fancyhdr.sty
├── icml2018.bst
├── icml2018.sty
│
├── .gitignore
└── README.md                   # This file
```

---

## MDP Formulation

| Component | Definition |
|---|---|
| **State** | 7-dim feature vector: diagram token ratio, conversation turn (normalized), RAG hit count (normalized), previous mode (one-hot), has explicit feedback |
| **Actions** | {0: MINIMAL, 1: BOUNDED, 2: FULL} |
| **Reward** | `quality − α_token · tokens − α_ttft · ttft` where quality comes from LLM score (1–5 → −1 to +1) or explicit feedback |
| **Episodes** | Real conversation sessions; each turn is one step |

The environment (`src/environment.py`) uses offline replay: each episode is a
recorded conversation session. Counterfactual rewards are estimated for unchosen
actions using a token-scaling model and the LLM quality offset model.

---

## Data

### `data/episodes.json`
500 conversation episodes (1,632 steps total). Each step contains:
- `diagram_mode` — the context mode actually used (MINIMAL / BOUNDED / FULL)
- `total_input_tokens`, `diagram_tokens`, `sources_tokens`, `user_message_tokens`
- `rag_hits_count` — number of RAG document hits
- `ttft` — time to first token (seconds)
- `feedback` — explicit user feedback (`thumbs_up` / `thumbs_down` / `null`)
- `persona` — user persona setting (`technical` / `business`)
- `conversation_turn`, `previous_mode`, `flow_id`
- `user_content`, `assistant_content` — **redacted** (see Data Sanitization below)

### `data/episodes_scored.json`
Same as `episodes.json` with an additional `llm_quality_score` field (1–5 scale)
produced by `scripts/llm_judge.py` using GPT-4o-mini as an automated evaluator.

### `data/final_results.json`
Aggregated policy evaluation results and ablation study outputs. No raw episode
content — safe to share as-is.

### Data Sanitization

> **Note for reproducibility**: The `user_content` and `assistant_content` fields
> in both episode files have been set to `null` in this public release. These fields
> contained the raw text of user queries and AI-generated responses from a production
> system, including proprietary domain-specific content. Removing them has **no
> effect on any results in the paper** — the RL training and evaluation pipeline
> uses only the numeric fields listed above.
>
> Additionally, `conversation_id` values have been re-hashed with SHA-256 to prevent
> reverse lookup of the original session IDs.
>
> If you need to reproduce the LLM scoring step (`llm_judge.py`), you will need to
> provide your own episode data with text content, or use the synthetic data
> generation fallback in `extract_data.py`.

---

## Setup

```bash
# Install dependencies
pip install numpy gymnasium torch stable-baselines3 openai psycopg2-binary

# Run experiments (uses data/episodes_scored.json by default)
cd CS234
python scripts/run_final_experiments.py

# Or reproduce from scratch with synthetic data
python scripts/extract_data.py          # generates data/episodes.json
python scripts/llm_judge.py             # adds llm_quality_score
python scripts/run_final_experiments.py # trains all agents, saves figures
```

### Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string for data extraction | No (falls back to synthetic) |
| `OPENAI_API_KEY` | GPT-4o-mini for LLM judge scoring | No (falls back to synthetic scores) |

Both steps fall back to synthetic data/scores if the credentials are unavailable,
so the full pipeline runs without any external dependencies.

---

## Reproducing Results

All figures in the paper can be reproduced by running:

```bash
python scripts/run_final_experiments.py
```

This trains all agents on the provided `episodes_scored.json`, evaluates them, runs
the three ablation studies (feature importance, reward coefficients, data source),
and writes all plots to `figures/`.

Expected runtime: ~5–10 minutes on CPU.

---

## Citation

If you use this code or data in your work, please cite the accompanying report:

```
CS234: Reinforcement Learning, Stanford University, Winter 2026
RL-Based Diagram Context Selection for RAG Systems
```
