#!/usr/bin/env python3
"""Run all final experiments: 8 policies, ablations, and comprehensive figures.

Policies:
  1. Random
  2. Always-MINIMAL
  3. Always-FULL
  4. Heuristic (production rule-based)
  5. LinUCB (contextual bandit)
  6. Q-Learning (tabular)
  7. DQN (deep Q-network)
  8. PPO (policy gradient via SB3)

Ablation studies:
  A1. Feature importance (leave-one-out)
  A2. Reward coefficient sensitivity (alpha_token, alpha_ttft)
  A3. Real-only vs. augmented data

Figures produced in CS234/figures/:
  - learning_curves_all.png
  - reward_comparison_final.png
  - mode_distributions_final.png
  - ablation_features.png
  - ablation_reward_coefficients.png
  - ablation_data_source.png
"""

import json
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import make_env, load_episodes, ContextSelectionEnv
from src.baselines import (
    RandomPolicy,
    AlwaysMinimalPolicy,
    AlwaysFullPolicy,
    HeuristicPolicy,
)
from src.q_learning import QLearningAgent
from src.contextual_bandits import LinUCBAgent
from src.dqn import DQNAgent
from src.ppo_agent import PPOAgent
from src.utils import MODE_NAMES, NUM_ACTIONS, compute_reward

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR.mkdir(exist_ok=True)

TRAIN_EPISODES_RL = 2000
SEED = 42


def evaluate_policy(policy, episodes, n_eval_episodes=None, seed=123):
    """Evaluate a policy on held-out episodes."""
    if n_eval_episodes is None:
        n_eval_episodes = len(episodes)
    env = make_env(episodes, seed=seed)
    all_rewards = []
    all_tokens = []
    all_modes = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_tokens = 0
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


def stratified_split(episodes, train_ratio=0.8, seed=42):
    """Stratified 80/20 split: separate real and synthetic, shuffle each, then split."""
    rng = np.random.RandomState(seed)

    real = [e for e in episodes if not e["conversation_id"].startswith("synthetic_")]
    synthetic = [e for e in episodes if e["conversation_id"].startswith("synthetic_")]

    rng.shuffle(real)
    rng.shuffle(synthetic)

    def split_list(lst, ratio):
        n = int(len(lst) * ratio)
        return lst[:n], lst[n:]

    real_train, real_test = split_list(real, train_ratio)
    synth_train, synth_test = split_list(synthetic, train_ratio)

    train = real_train + synth_train
    test = real_test + synth_test
    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


def run_all():
    print("=" * 60)
    print("CS234 Final Experiments")
    print("=" * 60)

    scored_path = DATA_DIR / "episodes_scored.json"
    fallback_path = DATA_DIR / "episodes.json"

    if scored_path.exists():
        episodes = load_episodes(str(scored_path))
        print(f"Loaded {len(episodes)} scored episodes")
    elif fallback_path.exists():
        episodes = load_episodes(str(fallback_path))
        print(f"Loaded {len(episodes)} episodes (unscored)")
    else:
        print(f"[error] No episodes found. Run extract_data.py first.")
        sys.exit(1)

    train_episodes, test_episodes = stratified_split(episodes, 0.8, seed=SEED)
    print(f"Train: {len(train_episodes)}, Test: {len(test_episodes)}")

    real_train = [e for e in train_episodes if not e["conversation_id"].startswith("synthetic_")]
    print(f"Real in train: {len(real_train)}, Real in test: {len([e for e in test_episodes if not e['conversation_id'].startswith('synthetic_')])}")

    # ===================================================================
    # Phase 1: Evaluate baselines
    # ===================================================================
    baselines = [
        RandomPolicy(seed=SEED),
        AlwaysMinimalPolicy(),
        AlwaysFullPolicy(),
        HeuristicPolicy(),
    ]

    results = {}
    print("\n--- Evaluating Baselines ---")
    for policy in baselines:
        metrics = evaluate_policy(policy, test_episodes)
        results[policy.name] = metrics
        _print_metrics(policy.name, metrics)

    # ===================================================================
    # Phase 2: Train and evaluate RL agents
    # ===================================================================
    training_curves = {}

    # LinUCB
    print("\n--- Training LinUCB ---")
    linucb = LinUCBAgent(alpha=1.5, seed=SEED)
    train_env = make_env(train_episodes, seed=SEED)
    linucb_rewards = linucb.train_on_env(train_env, n_episodes=TRAIN_EPISODES_RL)
    training_curves["LinUCB"] = linucb_rewards
    metrics = evaluate_policy(linucb, test_episodes)
    results[linucb.name] = metrics
    _print_metrics(linucb.name, metrics)

    # Q-Learning
    print("\n--- Training Q-Learning ---")
    qagent = QLearningAgent(
        alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
        n_bins=5, seed=SEED,
    )
    train_env = make_env(train_episodes, seed=SEED)
    ql_rewards = qagent.train_on_env(train_env, n_episodes=TRAIN_EPISODES_RL)
    training_curves["Q-Learning"] = ql_rewards
    metrics = evaluate_policy(qagent, test_episodes)
    results[qagent.name] = metrics
    _print_metrics(qagent.name, metrics)

    # DQN
    print("\n--- Training DQN ---")
    dqn = DQNAgent(
        lr=1e-3, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.997,
        batch_size=64, target_update_freq=50,
        hidden=64, seed=SEED,
    )
    train_env = make_env(train_episodes, seed=SEED)
    dqn_rewards = dqn.train_on_env(train_env, n_episodes=TRAIN_EPISODES_RL)
    training_curves["DQN"] = dqn_rewards
    metrics = evaluate_policy(dqn, test_episodes)
    results[dqn.name] = metrics
    _print_metrics(dqn.name, metrics)

    # PPO
    print("\n--- Training PPO ---")
    ppo = PPOAgent(lr=3e-4, gamma=0.99, seed=SEED)
    train_env = make_env(train_episodes, seed=SEED)
    ppo_rewards = ppo.train_on_env(train_env, n_episodes=TRAIN_EPISODES_RL)
    training_curves["PPO"] = ppo_rewards
    metrics = evaluate_policy(ppo, test_episodes)
    results[ppo.name] = metrics
    _print_metrics(ppo.name, metrics)

    # ===================================================================
    # Phase 3: Ablation studies
    # ===================================================================
    ablation_results = {}

    print("\n--- Ablation: Feature Importance ---")
    ablation_results["features"] = _ablation_feature_importance(train_episodes, test_episodes)

    print("\n--- Ablation: Reward Coefficients ---")
    ablation_results["reward_coeffs"] = _ablation_reward_coefficients(train_episodes, test_episodes)

    print("\n--- Ablation: Data Source ---")
    ablation_results["data_source"] = _ablation_data_source(episodes, test_episodes)

    # ===================================================================
    # Phase 4: Save results and generate figures
    # ===================================================================
    all_results = {
        "policies": results,
        "ablations": ablation_results,
    }
    results_path = DATA_DIR / "final_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    _plot_learning_curves(training_curves)
    _plot_reward_comparison(results)
    _plot_mode_distributions(results)
    _plot_ablation_features(ablation_results["features"])
    _plot_ablation_reward_coefficients(ablation_results["reward_coeffs"])
    _plot_ablation_data_source(ablation_results["data_source"])

    print("\n--- LaTeX Results Table ---")
    _print_latex_table(results)

    print("\nDone! Check CS234/figures/ and CS234/data/final_results.json")


def _print_metrics(name, metrics):
    print(
        f"  {name:18s} | "
        f"Reward: {metrics['mean_reward']:+.3f} +/- {metrics['std_reward']:.3f} | "
        f"Tokens/turn: {metrics['mean_tokens_per_turn']:.0f}"
    )


# ====================================================================
# Ablation Studies
# ====================================================================

def _ablation_feature_importance(train_episodes, test_episodes):
    """Leave-one-out feature ablation for Q-Learning."""
    import src.utils as utils_mod
    import src.environment as env_mod

    feature_names = [
        "diagram_token_ratio", "conversation_turn", "rag_hits",
        "is_minimal", "is_bounded", "is_full", "has_feedback",
    ]

    original_fn = utils_mod.extract_state_features

    baseline_agent = QLearningAgent(alpha=0.1, gamma=0.99, seed=SEED)
    train_env = make_env(train_episodes, seed=SEED)
    baseline_agent.train_on_env(train_env, n_episodes=1500, verbose=False)
    baseline_metrics = evaluate_policy(baseline_agent, test_episodes)
    baseline_reward = baseline_metrics["mean_reward"]

    results = {"baseline": baseline_reward, "features": {}}

    for i, fname in enumerate(feature_names):
        def make_masked_fn(mask_idx):
            def masked_extract(step):
                feats = original_fn(step)
                feats[mask_idx] = 0.0
                return feats
            return masked_extract

        masked_fn = make_masked_fn(i)
        utils_mod.extract_state_features = masked_fn
        env_mod.extract_state_features = masked_fn

        agent = QLearningAgent(alpha=0.1, gamma=0.99, seed=SEED)
        env = make_env(train_episodes, seed=SEED)
        agent.train_on_env(env, n_episodes=1500, verbose=False)
        m = evaluate_policy(agent, test_episodes)

        importance = baseline_reward - m["mean_reward"]
        results["features"][fname] = {
            "reward": m["mean_reward"],
            "importance": importance,
        }
        print(f"  {fname:25s}: reward={m['mean_reward']:+.3f}, importance={importance:+.3f}")

    utils_mod.extract_state_features = original_fn
    env_mod.extract_state_features = original_fn
    return results


def _ablation_reward_coefficients(train_episodes, test_episodes):
    """Sweep alpha_token and alpha_ttft coefficients."""
    import src.utils as utils_mod
    import src.environment as env_mod

    original_compute = utils_mod.compute_reward
    configs = [
        ("alpha_token=0", 0.0, 0.05),
        ("alpha_token=0.0001", 0.0001, 0.05),
        ("alpha_token=0.0002 (default)", 0.0002, 0.05),
        ("alpha_token=0.0005", 0.0005, 0.05),
        ("alpha_token=0.001", 0.001, 0.05),
        ("alpha_ttft=0", 0.0002, 0.0),
        ("alpha_ttft=0.025", 0.0002, 0.025),
        ("alpha_ttft=0.05 (default)", 0.0002, 0.05),
        ("alpha_ttft=0.1", 0.0002, 0.1),
    ]

    results = {}
    for label, a_tok, a_ttft in configs:
        def make_reward_fn(at, att):
            def custom_reward(feedback, total_tokens, ttft, alpha_token=at, alpha_ttft=att, llm_quality_score=None):
                return original_compute(feedback, total_tokens, ttft, at, att, llm_quality_score)
            return custom_reward

        patched = make_reward_fn(a_tok, a_ttft)
        utils_mod.compute_reward = patched
        env_mod.compute_reward = patched

        agent = QLearningAgent(alpha=0.1, gamma=0.99, seed=SEED)
        env = make_env(train_episodes, seed=SEED)
        agent.train_on_env(env, n_episodes=1500, verbose=False)
        m = evaluate_policy(agent, test_episodes)
        results[label] = m["mean_reward"]
        print(f"  {label:35s}: reward={m['mean_reward']:+.3f}")

    utils_mod.compute_reward = original_compute
    env_mod.compute_reward = original_compute
    return results


def _ablation_data_source(all_episodes, test_episodes):
    """Compare real-only vs. augmented training data."""
    real_only = [e for e in all_episodes if not e["conversation_id"].startswith("synthetic_")]
    real_train = real_only[:int(len(real_only) * 0.8)]

    augmented_train = all_episodes[:int(len(all_episodes) * 0.8)]

    results = {}

    for label, train_data in [("Real-Only", real_train), ("Augmented", augmented_train)]:
        if len(train_data) < 5:
            print(f"  {label}: insufficient data ({len(train_data)} episodes)")
            results[label] = {"mean_reward": 0.0, "std_reward": 0.0}
            continue

        agent = QLearningAgent(alpha=0.1, gamma=0.99, seed=SEED)
        env = make_env(train_data, seed=SEED)
        agent.train_on_env(env, n_episodes=1000, verbose=False)
        m = evaluate_policy(agent, test_episodes)
        results[label] = m
        print(f"  {label:15s}: reward={m['mean_reward']:+.3f} +/- {m['std_reward']:.3f}")

    return results


# ====================================================================
# Plotting Functions
# ====================================================================

COLORS = {
    "Random": "#999999",
    "Always-MINIMAL": "#55a868",
    "Always-FULL": "#dd8452",
    "Heuristic": "#c44e52",
    "LinUCB": "#937860",
    "Q-Learning": "#4c72b0",
    "DQN": "#8172b3",
    "PPO": "#da8bc3",
}


def _plot_learning_curves(training_curves, window=50):
    """Plot all RL agent learning curves on one figure."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name, rewards in training_curves.items():
        color = COLORS.get(name, "#333333")
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(rewards)), ma, color=color,
                    linewidth=2, label=f"{name} ({window}-ep MA)")

    ax.set_ylim(-5, -1)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves: RL Agents")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "learning_curves_all.png", dpi=150)
    plt.close()
    print("  Saved learning_curves_all.png")


def _plot_reward_comparison(results):
    """Bar chart comparing mean rewards across all 8 policies."""
    policy_order = [
        "Random", "Always-MINIMAL", "Always-FULL", "Heuristic",
        "LinUCB", "Q-Learning", "DQN", "PPO",
    ]
    names = [n for n in policy_order if n in results]
    means = [results[n]["mean_reward"] for n in names]
    stds = [results[n]["std_reward"] for n in names]
    colors = [COLORS.get(n, "#333333") for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Policy Comparison: Mean Reward (8 Policies)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "reward_comparison_final.png", dpi=150)
    plt.close()
    print("  Saved reward_comparison_final.png")


def _plot_mode_distributions(results):
    """Stacked bar chart of mode distributions for all 8 policies."""
    policy_order = [
        "Random", "Always-MINIMAL", "Always-FULL", "Heuristic",
        "LinUCB", "Q-Learning", "DQN", "PPO",
    ]
    names = [n for n in policy_order if n in results]

    minimal = [results[n]["mode_distribution"].get("MINIMAL", 0) for n in names]
    bounded = [results[n]["mode_distribution"].get("BOUNDED", 0) for n in names]
    full = [results[n]["mode_distribution"].get("FULL", 0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(names))
    w = 0.6
    ax.bar(x, minimal, w, label="MINIMAL", color="#55a868")
    ax.bar(x, bounded, w, bottom=minimal, label="BOUNDED", color="#4c72b0")
    ax.bar(x, full, w, bottom=[m + b for m, b in zip(minimal, bounded)], label="FULL", color="#dd8452")
    ax.set_ylabel("Proportion")
    ax.set_title("Context Mode Distribution by Policy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mode_distributions_final.png", dpi=150)
    plt.close()
    print("  Saved mode_distributions_final.png")


def _plot_ablation_features(feature_results):
    """Horizontal bar chart of feature importance."""
    features = feature_results["features"]
    names = list(features.keys())
    importances = [features[n]["importance"] for n in names]

    sorted_idx = np.argsort(importances)
    names = [names[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in importances]
    ax.barh(names, importances, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Reward Drop When Feature Removed")
    ax.set_title("Feature Importance (Leave-One-Out Ablation)")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_features.png", dpi=150)
    plt.close()
    print("  Saved ablation_features.png")


def _plot_ablation_reward_coefficients(coeff_results):
    """Grouped bar chart for reward coefficient sensitivity."""
    token_labels = [k for k in coeff_results if "alpha_token" in k]
    ttft_labels = [k for k in coeff_results if "alpha_ttft" in k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(range(len(token_labels)),
            [coeff_results[k] for k in token_labels],
            color="#4c72b0", edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(token_labels)))
    ax1.set_xticklabels([k.split("=")[1].split(" ")[0] for k in token_labels], rotation=30)
    ax1.set_xlabel("alpha_token")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Token Cost Sensitivity")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(range(len(ttft_labels)),
            [coeff_results[k] for k in ttft_labels],
            color="#dd8452", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(ttft_labels)))
    ax2.set_xticklabels([k.split("=")[1].split(" ")[0] for k in ttft_labels], rotation=30)
    ax2.set_xlabel("alpha_ttft")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("TTFT Cost Sensitivity")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Reward Coefficient Sensitivity Analysis", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_reward_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved ablation_reward_coefficients.png")


def _plot_ablation_data_source(data_results):
    """Bar chart comparing real-only vs augmented training."""
    names = list(data_results.keys())
    means = [data_results[n].get("mean_reward", data_results[n]) if isinstance(data_results[n], dict) else data_results[n] for n in names]
    stds = [data_results[n].get("std_reward", 0) if isinstance(data_results[n], dict) else 0 for n in names]

    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#55a868", "#4c72b0"]
    ax.bar(names, means, yerr=stds, capsize=5, color=colors[:len(names)],
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Training Data: Real-Only vs. Augmented")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_data_source.png", dpi=150)
    plt.close()
    print("  Saved ablation_data_source.png")


def _print_latex_table(results):
    """Print results as a LaTeX booktabs table."""
    policy_order = [
        "Random", "Always-MINIMAL", "Always-FULL", "Heuristic",
        "LinUCB", "Q-Learning", "DQN", "PPO",
    ]

    print(r"\begin{table}[t]")
    print(r"\caption{Comparison of all context selection policies on held-out test episodes.}")
    print(r"\label{tab:final-results}")
    print(r"\vskip 0.15in")
    print(r"\begin{center}")
    print(r"\begin{small}")
    print(r"\begin{sc}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Policy & Reward & $\pm$ Std & Tok/Turn & \% Full \\")
    print(r"\midrule")

    for name in policy_order:
        if name not in results:
            continue
        m = results[name]
        reward = f"{m['mean_reward']:+.2f}"
        std = f"{m['std_reward']:.2f}"
        tokens = f"{m['mean_tokens_per_turn']:.0f}"
        pct_full = f"{100 * m['mode_distribution'].get('FULL', 0):.0f}\\%"
        print(f"{name} & {reward} & {std} & {tokens} & {pct_full} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{sc}")
    print(r"\end{small}")
    print(r"\end{center}")
    print(r"\vskip -0.1in")
    print(r"\end{table}")


if __name__ == "__main__":
    run_all()
