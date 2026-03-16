"""
display.py - Output formatting for Reinforcement Learning Playground
===================================================================

Provides:
    - CLI banners and headers
    - Configuration display
    - RL performance report formatting
    - Session summary generation
"""


def format_header() -> str:
    """Return the CLI banner for this project.

    Returns:
        str: User-facing banner string.
    """
    return (
        "=" * 70
        + "\n"
        + "  REINFORCEMENT LEARNING PLAYGROUND - Q-LEARNING GRIDWORLD\n"
        + "=" * 70
    )


def format_startup_guide(config: dict, env_settings: dict) -> str:
    """Return startup guidance shown before training begins.

    Parameters:
        config (dict): Runtime configuration.
        env_settings (dict): Environment constants.

    Returns:
        str: Multi-line startup guide.
    """
    lines = [
        "",
        "Configuration:",
        f"  Episodes: {config['episodes']}",
        f"  Max steps per episode: {config['max_steps_per_episode']}",
        f"  Learning rate (alpha): {config['learning_rate']}",
        f"  Discount factor (gamma): {config['discount_factor']}",
        f"  Epsilon schedule: {config['epsilon_start']} -> {config['epsilon_min']}"
        f" (decay={config['epsilon_decay']})",
        f"  Evaluation episodes: {config['evaluation_episodes']}",
        f"  Random seed: {config['random_seed']}",
        "",
        "Environment:",
        f"  Grid size: {env_settings['grid_size']}x{env_settings['grid_size']}",
        f"  Start: {env_settings['start']}",
        f"  Goal: {env_settings['goal']}",
        f"  Trap: {env_settings['trap']}",
        f"  Obstacles: {env_settings['obstacles']}",
        "",
        "Rewards:",
        f"  Goal reward: {env_settings['rewards']['goal']}",
        f"  Trap penalty: {env_settings['rewards']['trap']}",
        f"  Step penalty: {env_settings['rewards']['step']}",
        f"  Collision penalty: {env_settings['rewards']['collision']}",
        "",
        "Session behavior:",
        "  1) Train a tabular Q-learning agent with epsilon-greedy exploration.",
        "  2) Track reward and success rate across episodes.",
        "  3) Evaluate greedy learned policy on held-out episodes.",
        "  4) Compare results against a random-action baseline policy.",
        "  5) Save run artifacts and learning-curve plots for review.",
    ]
    return "\n".join(lines)


def _format_training_metrics(metrics: dict) -> list[str]:
    """Return readable training metrics."""
    if not metrics:
        return ["  Training metrics: no data"]
    return [
        f"  Mean reward (last 50 episodes): {metrics['mean_reward_last_50']:.2f}",
        f"  Mean steps (last 50 episodes): {metrics['mean_steps_last_50']:.2f}",
        f"  Success rate (last 50 episodes): {metrics['success_rate_last_50']:.1%}",
        f"  Q-table stats: mean={metrics['q_table_mean']:.3f}, max={metrics['q_table_max']:.3f}",
    ]


def _format_eval_metrics(label: str, metrics: dict) -> list[str]:
    """Return evaluation metric lines for one policy."""
    if not metrics:
        return [f"  {label}: no data"]
    return [
        f"  {label}:",
        f"    Episodes: {metrics['episodes']}",
        f"    Avg reward: {metrics['avg_reward']:.2f}",
        f"    Avg steps: {metrics['avg_steps']:.2f}",
        f"    Success rate: {metrics['success_rate']:.1%}",
    ]


def format_run_report(summary: dict) -> str:
    """Return formatted session report for display.

    Parameters:
        summary (dict): Complete session summary.

    Returns:
        str: Formatted multi-line report.
    """
    lines = ["", "Session summary:"]
    lines.append(f"  Status: {summary.get('status', 'unknown')}")

    config = summary.get("config", {})
    lines.append(
        f"  Training episodes: {config.get('episodes', 0)}"
    )

    base_result = summary.get("base_result", {})
    training_metrics = base_result.get("training_metrics", {})
    evaluation_metrics = base_result.get("evaluation_metrics", {})
    baseline_metrics = base_result.get("baseline_metrics", {})
    comparison = base_result.get("comparison", {})

    lines.extend(_format_training_metrics(training_metrics))
    lines.extend(_format_eval_metrics("Learned policy", evaluation_metrics))
    lines.extend(_format_eval_metrics("Random baseline", baseline_metrics))

    if comparison:
        lines.extend(
            [
                "  Improvement over baseline:",
                f"    Success-rate gain: {comparison['improvement_success_rate']:.1%}",
                f"    Avg-reward gain: {comparison['improvement_avg_reward']:.2f}",
            ]
        )

    lines.extend(
        [
            "  Saved plots: data/runs/learning_curve.png, data/runs/success_rate_curve.png",
            "Saved run artifact: data/runs/latest_rl_playground.json",
            "",
        ]
    )
    return "\n".join(lines)


def format_message(message: str) -> str:
    """Format a user-facing message string.

    Parameters:
        message (str): Raw message text.

    Returns:
        str: Prefixed message string.
    """
    return f"[RL Playground] {message}"
