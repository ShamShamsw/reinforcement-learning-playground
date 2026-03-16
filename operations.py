"""
operations.py - RL training engine and analysis workflows
=========================================================

Implements:
    - GridWorld environment transitions
    - Tabular Q-learning with epsilon-greedy exploration
    - Random-policy baseline comparison
    - Training artifact generation and persistence
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models import (
    create_comparison_metrics,
    create_episode_record,
    create_project_config,
    create_training_metrics,
    create_training_result,
)
from storage import RUNS_DIR, save_latest_session


@dataclass
class GridWorld:
    """Simple deterministic 5x5 GridWorld.

    State is represented by (row, col).
    """

    size: int = 5
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (4, 4)
    trap: tuple[int, int] = (3, 3)
    obstacles: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 3))
    step_penalty: float = -0.2
    wall_penalty: float = -1.0
    trap_penalty: float = -10.0
    goal_reward: float = 12.0

    def reset(self) -> tuple[int, int]:
        """Reset environment to start state.

        Returns:
            tuple[int, int]: Starting state.
        """
        return self.start

    def step(self, state: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool, bool]:
        """Take one environment step.

        Actions: 0=up, 1=right, 2=down, 3=left.

        Returns:
            tuple: next_state, reward, done, success
        """
        dr_dc = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
        }
        dr, dc = dr_dc[action]
        nr, nc = state[0] + dr, state[1] + dc

        # Boundary or obstacle collision: stay in place and penalize.
        if nr < 0 or nr >= self.size or nc < 0 or nc >= self.size:
            return state, self.wall_penalty, False, False
        if (nr, nc) in self.obstacles:
            return state, self.wall_penalty, False, False

        next_state = (nr, nc)
        if next_state == self.goal:
            return next_state, self.goal_reward, True, True
        if next_state == self.trap:
            return next_state, self.trap_penalty, True, False
        return next_state, self.step_penalty, False, False


def load_environment_settings() -> dict:
    """Return environment constants for startup display and reports.

    Returns:
        dict: Grid dimensions, rewards, and action mapping.
    """
    return {
        "grid_size": 5,
        "start": (0, 0),
        "goal": (4, 4),
        "trap": (3, 3),
        "obstacles": [(1, 1), (2, 1), (1, 3)],
        "actions": {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
        },
        "rewards": {
            "goal": 12.0,
            "trap": -10.0,
            "step": -0.2,
            "collision": -1.0,
        },
    }


def _state_to_index(state: tuple[int, int], grid_size: int) -> int:
    """Encode 2D grid state as flat integer index."""
    return state[0] * grid_size + state[1]


def _run_policy(
    env: GridWorld,
    q_table: np.ndarray | None,
    episodes: int,
    max_steps: int,
    rng: np.random.Generator,
) -> dict:
    """Evaluate either greedy (q_table) or random (None) policy.

    Parameters:
        env (GridWorld): Environment instance.
        q_table (np.ndarray | None): If provided, use greedy action selection.
        episodes (int): Number of episodes to evaluate.
        max_steps (int): Maximum steps per episode.
        rng (np.random.Generator): Random generator.

    Returns:
        dict: Aggregate evaluation metrics.
    """
    total_rewards = []
    total_steps = []
    successes = []

    for _ in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        success = False

        for step in range(1, max_steps + 1):
            if q_table is None:
                action = int(rng.integers(0, 4))
            else:
                s_idx = _state_to_index(state, env.size)
                action = int(np.argmax(q_table[s_idx]))

            next_state, reward, done, is_success = env.step(state, action)
            episode_reward += reward
            state = next_state

            if done:
                success = is_success
                total_steps.append(step)
                break
        else:
            total_steps.append(max_steps)

        total_rewards.append(episode_reward)
        successes.append(1.0 if success else 0.0)

    return {
        "episodes": episodes,
        "avg_reward": float(np.mean(total_rewards)),
        "avg_steps": float(np.mean(total_steps)),
        "success_rate": float(np.mean(successes)),
    }


def _save_training_plots(history: list[dict]) -> dict:
    """Generate and save training plots under data/runs.

    Parameters:
        history (list[dict]): Episode history.

    Returns:
        dict: Paths to generated plot files.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    episodes = np.array([h["episode"] for h in history])
    rewards = np.array([h["total_reward"] for h in history], dtype=float)
    success = np.array([1.0 if h["success"] else 0.0 for h in history], dtype=float)

    window = 20
    if len(rewards) >= window:
        kernel = np.ones(window) / window
        rewards_smooth = np.convolve(rewards, kernel, mode="valid")
        success_smooth = np.convolve(success, kernel, mode="valid")
        smooth_x = episodes[window - 1 :]
    else:
        rewards_smooth = rewards
        success_smooth = success
        smooth_x = episodes

    sns.set_theme(style="whitegrid")

    learning_curve_path = RUNS_DIR / "learning_curve.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(episodes, rewards, alpha=0.35, label="Episode reward")
    ax.plot(smooth_x, rewards_smooth, linewidth=2.2, label=f"{window}-episode moving average")
    ax.set_title("Q-Learning Training Curve")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(learning_curve_path, dpi=160)
    plt.close(fig)

    success_curve_path = RUNS_DIR / "success_rate_curve.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(smooth_x, success_smooth, color="#2b8cbe", linewidth=2.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Rolling Success Rate During Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate")
    fig.tight_layout()
    fig.savefig(success_curve_path, dpi=160)
    plt.close(fig)

    return {
        "learning_curve": str(Path("data") / "runs" / "learning_curve.png"),
        "success_curve": str(Path("data") / "runs" / "success_rate_curve.png"),
    }


def run_core_flow(config: dict | None = None) -> dict:
    """Execute the main RL training workflow.

    Parameters:
        config (dict | None): Optional override of default config.

    Returns:
        dict: Session summary with training/evaluation outputs.
    """
    if config is None:
        config = create_project_config()

    env = GridWorld()
    rng = np.random.default_rng(config.get("random_seed", 42))

    n_states = env.size * env.size
    n_actions = 4
    q_table = np.zeros((n_states, n_actions), dtype=float)

    alpha = config["learning_rate"]
    gamma = config["discount_factor"]
    epsilon = config["epsilon_start"]
    epsilon_decay = config["epsilon_decay"]
    epsilon_min = config["epsilon_min"]

    history: list[dict] = []

    for episode in range(1, config["episodes"] + 1):
        state = env.reset()
        total_reward = 0.0
        success = False

        for step in range(1, config["max_steps_per_episode"] + 1):
            s_idx = _state_to_index(state, env.size)

            if rng.random() < epsilon:
                action = int(rng.integers(0, n_actions))
            else:
                action = int(np.argmax(q_table[s_idx]))

            next_state, reward, done, is_success = env.step(state, action)
            ns_idx = _state_to_index(next_state, env.size)

            td_target = reward + gamma * np.max(q_table[ns_idx])
            td_error = td_target - q_table[s_idx, action]
            q_table[s_idx, action] += alpha * td_error

            total_reward += reward
            state = next_state

            if done:
                success = is_success
                history.append(
                    create_episode_record(
                        episode=episode,
                        total_reward=float(total_reward),
                        steps=step,
                        success=success,
                        epsilon=float(epsilon),
                    )
                )
                break
        else:
            history.append(
                create_episode_record(
                    episode=episode,
                    total_reward=float(total_reward),
                    steps=config["max_steps_per_episode"],
                    success=False,
                    epsilon=float(epsilon),
                )
            )

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    recent = history[-50:] if len(history) >= 50 else history
    mean_reward_last_50 = float(np.mean([r["total_reward"] for r in recent]))
    mean_steps_last_50 = float(np.mean([r["steps"] for r in recent]))
    success_rate_last_50 = float(np.mean([1.0 if r["success"] else 0.0 for r in recent]))

    training_metrics = create_training_metrics(
        mean_reward_last_50=mean_reward_last_50,
        mean_steps_last_50=mean_steps_last_50,
        success_rate_last_50=success_rate_last_50,
        q_table_mean=float(np.mean(q_table)),
        q_table_max=float(np.max(q_table)),
    )

    evaluation_metrics = _run_policy(
        env=env,
        q_table=q_table,
        episodes=config["evaluation_episodes"],
        max_steps=config["max_steps_per_episode"],
        rng=rng,
    )
    baseline_metrics = _run_policy(
        env=env,
        q_table=None,
        episodes=config["evaluation_episodes"],
        max_steps=config["max_steps_per_episode"],
        rng=rng,
    )

    comparison = create_comparison_metrics(
        q_learning_success_rate=evaluation_metrics["success_rate"],
        baseline_success_rate=baseline_metrics["success_rate"],
        q_learning_avg_reward=evaluation_metrics["avg_reward"],
        baseline_avg_reward=baseline_metrics["avg_reward"],
        improvement_success_rate=evaluation_metrics["success_rate"] - baseline_metrics["success_rate"],
        improvement_avg_reward=evaluation_metrics["avg_reward"] - baseline_metrics["avg_reward"],
    )

    plots = _save_training_plots(history)
    environment = load_environment_settings()

    base_result = create_training_result(
        config=config,
        environment=environment,
        history=history,
        q_table=q_table.tolist(),
        training_metrics=training_metrics,
        evaluation_metrics=evaluation_metrics,
        baseline_metrics=baseline_metrics,
        comparison=comparison,
    )
    base_result["plots"] = plots

    session_summary = {
        "status": "completed",
        "config": config,
        "base_result": base_result,
    }
    save_latest_session(session_summary)
    return session_summary
