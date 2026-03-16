"""
models.py - Data models for Reinforcement Learning Playground
=============================================================

Defines:
    - Project configuration and environment settings
    - Episode records and training summaries
    - Evaluation and comparison metrics
"""

from datetime import datetime


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp.

    Returns:
        str: Timestamp string with trailing Z.
    """
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def create_project_config(
    episodes: int = 400,
    max_steps_per_episode: int = 80,
    learning_rate: float = 0.15,
    discount_factor: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_decay: float = 0.992,
    epsilon_min: float = 0.05,
    evaluation_episodes: int = 150,
    random_seed: int = 42,
) -> dict:
    """Create the default runtime configuration for Q-learning.

    Parameters:
        episodes (int): Number of training episodes.
        max_steps_per_episode (int): Step cap per episode.
        learning_rate (float): Q-learning step size (alpha).
        discount_factor (float): Future reward discount (gamma).
        epsilon_start (float): Initial exploration probability.
        epsilon_decay (float): Multiplicative epsilon decay per episode.
        epsilon_min (float): Minimum exploration probability.
        evaluation_episodes (int): Number of greedy evaluation episodes.
        random_seed (int): Random seed for reproducibility.

    Returns:
        dict: Configuration dictionary.
    """
    return {
        "episodes": episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon_start": epsilon_start,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min,
        "evaluation_episodes": evaluation_episodes,
        "random_seed": random_seed,
        "created_at": _utc_timestamp(),
    }


def create_episode_record(
    episode: int,
    total_reward: float,
    steps: int,
    success: bool,
    epsilon: float,
) -> dict:
    """Create one episode record.

    Parameters:
        episode (int): 1-based episode index.
        total_reward (float): Total reward for the episode.
        steps (int): Number of environment steps used.
        success (bool): Whether the goal was reached.
        epsilon (float): Exploration rate used in the episode.

    Returns:
        dict: Episode summary.
    """
    return {
        "episode": episode,
        "total_reward": total_reward,
        "steps": steps,
        "success": success,
        "epsilon": epsilon,
    }


def create_training_metrics(
    mean_reward_last_50: float,
    mean_steps_last_50: float,
    success_rate_last_50: float,
    q_table_mean: float,
    q_table_max: float,
) -> dict:
    """Create training summary metrics.

    Parameters:
        mean_reward_last_50 (float): Mean reward in final 50 episodes.
        mean_steps_last_50 (float): Mean steps in final 50 episodes.
        success_rate_last_50 (float): Goal-reaching rate in final 50 episodes.
        q_table_mean (float): Mean Q-value across all entries.
        q_table_max (float): Maximum Q-value learned.

    Returns:
        dict: Training metric summary.
    """
    return {
        "mean_reward_last_50": mean_reward_last_50,
        "mean_steps_last_50": mean_steps_last_50,
        "success_rate_last_50": success_rate_last_50,
        "q_table_mean": q_table_mean,
        "q_table_max": q_table_max,
    }


def create_comparison_metrics(
    q_learning_success_rate: float,
    baseline_success_rate: float,
    q_learning_avg_reward: float,
    baseline_avg_reward: float,
    improvement_success_rate: float,
    improvement_avg_reward: float,
) -> dict:
    """Create comparison metrics between learned policy and baseline.

    Parameters:
        q_learning_success_rate (float): Success rate of greedy Q policy.
        baseline_success_rate (float): Success rate of random baseline policy.
        q_learning_avg_reward (float): Average reward for Q policy.
        baseline_avg_reward (float): Average reward for baseline policy.
        improvement_success_rate (float): Success-rate gain over baseline.
        improvement_avg_reward (float): Reward gain over baseline.

    Returns:
        dict: Head-to-head metrics.
    """
    return {
        "q_learning_success_rate": q_learning_success_rate,
        "baseline_success_rate": baseline_success_rate,
        "q_learning_avg_reward": q_learning_avg_reward,
        "baseline_avg_reward": baseline_avg_reward,
        "improvement_success_rate": improvement_success_rate,
        "improvement_avg_reward": improvement_avg_reward,
    }


def create_training_result(
    config: dict,
    environment: dict,
    history: list[dict],
    q_table: list[list[float]],
    training_metrics: dict,
    evaluation_metrics: dict,
    baseline_metrics: dict,
    comparison: dict,
) -> dict:
    """Create complete RL training session result.

    Parameters:
        config (dict): Training configuration.
        environment (dict): Environment settings and rewards.
        history (list[dict]): Episode-by-episode training history.
        q_table (list[list[float]]): Learned Q-table as nested list.
        training_metrics (dict): Aggregate training metrics.
        evaluation_metrics (dict): Learned-policy evaluation metrics.
        baseline_metrics (dict): Baseline policy evaluation metrics.
        comparison (dict): Improvement metrics versus baseline.

    Returns:
        dict: Complete project output artifact.
    """
    return {
        "config": config,
        "environment": environment,
        "history": history,
        "q_table": q_table,
        "training_metrics": training_metrics,
        "evaluation_metrics": evaluation_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
        "completed_at": _utc_timestamp(),
    }
