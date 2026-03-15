#!/usr/bin/env python3
"""Run all experiments: evaluate baselines + train Q-Learning, produce results."""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import make_env, load_episodes
from src.baselines import (
    RandomPolicy,
    AlwaysMinimalPolicy,
    AlwaysFullPolicy,
    HeuristicPolicy,
)
from src.q_learning import QLearningAgent
from src.utils import MODE_NAMES, compute_reward

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR.mkdir(exist_ok=True)


def evaluate_policy(policy, episodes, n_eval_episodes=200, seed=123):
    """Evaluate a policy on the environment, return per-episode metrics."""
    env = make_env(episodes, seed=seed)
    all_rewards = []
    all_tokens = []
    all_ttft = []
    all_modes = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_tokens = 0
        ep_ttft = 0.0
        ep_steps = 0
        done = False

        while not done:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_tokens += info.get("actual_tokens", 0)
            all_modes.append(info.get("chosen_mode", "?"))
            ep_steps += 1

        all_rewards.append(ep_reward)
        if ep_steps > 0:
            all_tokens.append(ep_tokens / ep_steps)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_tokens_per_turn": float(np.mean(all_tokens)) if all_tokens else 0,
        "mode_distribution": {
            m: all_modes.count(m) / max(len(all_modes), 1) for m in MODE_NAMES
        },
    }


def run_all():
    print("=" * 60)
    print("CS234 Experiment Runner")
    print("=" * 60)

    episodes_path = DATA_DIR / "episodes.json"
    if not episodes_path.exists():
        print(f"[error] {episodes_path} not found. Run extract_data.py first.")
        sys.exit(1)

    episodes = load_episodes(str(episodes_path))
    print(f"Loaded {len(episodes)} episodes")

    split = int(len(episodes) * 0.8)
    train_episodes = episodes[:split]
    test_episodes = episodes[split:]
    print(f"Train: {len(train_episodes)}, Test: {len(test_episodes)}")

    # --- Evaluate baselines ---
    baselines = [
        HeuristicPolicy(),
        AlwaysFullPolicy(),
        AlwaysMinimalPolicy(),
        RandomPolicy(seed=42),
    ]

    results = {}
    print("\n--- Evaluating Baselines ---")
    for policy in baselines:
        metrics = evaluate_policy(policy, test_episodes, n_eval_episodes=len(test_episodes))
        results[policy.name] = metrics
        print(
            f"  {policy.name:18s} | "
            f"Reward: {metrics['mean_reward']:+.3f} +/- {metrics['std_reward']:.3f} | "
            f"Tokens/turn: {metrics['mean_tokens_per_turn']:.0f}"
        )

    # --- Train Q-Learning ---
    print("\n--- Training Q-Learning ---")
    train_env = make_env(train_episodes, seed=42)
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        n_bins=5,
        seed=42,
    )
    training_rewards = agent.train_on_env(train_env, n_episodes=2000, verbose=True)

    # Evaluate trained Q-Learning
    metrics = evaluate_policy(agent, test_episodes, n_eval_episodes=len(test_episodes))
    results[agent.name] = metrics
    print(
        f"\n  {agent.name:18s} | "
        f"Reward: {metrics['mean_reward']:+.3f} +/- {metrics['std_reward']:.3f} | "
        f"Tokens/turn: {metrics['mean_tokens_per_turn']:.0f}"
    )

    policy_stats = agent.get_policy_stats()
    print(f"  Q-Learning policy distribution: {policy_stats}")

    # --- Save results ---
    results_path = DATA_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # --- Generate figures ---
    _plot_learning_curve(training_rewards)
    _plot_reward_comparison(results)
    _plot_mode_distributions(results)

    # --- Print LaTeX table ---
    print("\n--- LaTeX Results Table ---")
    _print_latex_table(results)

    print("\nDone! Check CS234/figures/ for plots.")


def _plot_learning_curve(rewards, window=50):
    """Plot Q-Learning training curve with moving average."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(rewards, alpha=0.15, color="steelblue", linewidth=0.5)

    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), ma, color="steelblue", linewidth=2,
                label=f"{window}-episode moving avg")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Q-Learning Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "learning_curve.png", dpi=150)
    plt.close()
    print(f"  Saved learning_curve.png")


def _plot_reward_comparison(results):
    """Bar chart comparing mean rewards across policies."""
    names = list(results.keys())
    means = [results[n]["mean_reward"] for n in names]
    stds = [results[n]["std_reward"] for n in names]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    bars = ax.bar(names, means, yerr=stds, capsize=4,
                  color=colors[: len(names)], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Policy Comparison: Mean Reward")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reward_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved reward_comparison.png")


def _plot_mode_distributions(results):
    """Stacked bar chart showing mode distribution per policy."""
    names = list(results.keys())
    minimal = [results[n]["mode_distribution"].get("MINIMAL", 0) for n in names]
    bounded = [results[n]["mode_distribution"].get("BOUNDED", 0) for n in names]
    full = [results[n]["mode_distribution"].get("FULL", 0) for n in names]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(names))
    w = 0.6
    ax.bar(x, minimal, w, label="MINIMAL", color="#55a868")
    ax.bar(x, bounded, w, bottom=minimal, label="BOUNDED", color="#4c72b0")
    ax.bar(x, full, w, bottom=[m + b for m, b in zip(minimal, bounded)],
           label="FULL", color="#dd8452")
    ax.set_ylabel("Proportion")
    ax.set_title("Context Mode Distribution by Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mode_distributions.png", dpi=150)
    plt.close()
    print(f"  Saved mode_distributions.png")


def _print_latex_table(results):
    """Print results as a LaTeX booktabs table."""
    print(r"\begin{table}[t]")
    print(r"\caption{Comparison of context selection policies on held-out test episodes.}")
    print(r"\label{tab:results}")
    print(r"\vskip 0.15in")
    print(r"\begin{center}")
    print(r"\begin{small}")
    print(r"\begin{sc}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Policy & Reward & Tokens/Turn & \% FULL \\")
    print(r"\midrule")

    for name, m in results.items():
        reward = f"{m['mean_reward']:+.2f}"
        tokens = f"{m['mean_tokens_per_turn']:.0f}"
        pct_full = f"{100 * m['mode_distribution'].get('FULL', 0):.0f}\\%"
        print(f"{name} & {reward} & {tokens} & {pct_full} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{sc}")
    print(r"\end{small}")
    print(r"\end{center}")
    print(r"\vskip -0.1in")
    print(r"\end{table}")


if __name__ == "__main__":
    run_all()
