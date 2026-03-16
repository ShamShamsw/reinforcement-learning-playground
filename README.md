# Beginner Project 18: Reinforcement Learning Playground

**Time:** 3-5 hours  
**Difficulty:** Intermediate Beginner  
**Focus:** tabular Q-learning, epsilon-greedy exploration, policy evaluation, baseline comparison, and learning curve analysis

---

## Why This Project?

Reinforcement learning teaches agents to make sequential decisions through trial and error.
This project gives you a practical beginner workflow where you can:

- build and train a tabular Q-learning agent in a custom GridWorld,
- explore how learning rate, discounting, and exploration schedules affect behavior,
- compare a learned policy against a simple random-action baseline,
- interpret learning dynamics through reward and success-rate curves,
- and save training runs for reproducibility and later analysis.

---

## More Projects

You can access this project and more in this separate repository:

[student-interview-prep](https://github.com/ShamShamsw/student-interview-prep.git)

---

## What You Will Build

You will build a command-line reinforcement learning playground that:

1. Defines a deterministic GridWorld environment with start, goal, trap, obstacles, and reward shaping.
2. Trains a tabular Q-learning agent using an epsilon-greedy policy over hundreds of episodes.
3. Tracks per-episode rewards, steps, and success outcomes through the full training run.
4. Evaluates the learned greedy policy on held-out episodes to estimate performance.
5. Runs a random-action baseline policy for direct, interpretable comparison.
6. Computes summary metrics including success-rate gain and reward improvement over baseline.
7. Saves JSON artifacts and learning-curve plots for reproducibility and review.

---

## Requirements

- Python 3.11+
- `numpy`
- `matplotlib`
- `seaborn`

Install with:

```bash
pip install -r requirements.txt
```

---

## Example Session

```text
======================================================================
   REINFORCEMENT LEARNING PLAYGROUND - Q-LEARNING GRIDWORLD
======================================================================

Configuration:
   Episodes: 400
   Max steps per episode: 80
   Learning rate (alpha): 0.15
   Discount factor (gamma): 0.95
   Epsilon schedule: 1.0 -> 0.05 (decay=0.992)
   Evaluation episodes: 150
   Random seed: 42

Environment:
   Grid size: 5x5
   Start: (0, 0)
   Goal: (4, 4)
   Trap: (3, 3)
   Obstacles: [(1, 1), (2, 1), (1, 3)]

Rewards:
   Goal reward: 12.0
   Trap penalty: -10.0
   Step penalty: -0.2
   Collision penalty: -1.0

Session behavior:
   1) Train a tabular Q-learning agent with epsilon-greedy exploration.
   2) Track reward and success rate across episodes.
   3) Evaluate greedy learned policy on held-out episodes.
   4) Compare results against a random-action baseline policy.
   5) Save run artifacts and learning-curve plots for review.

Session summary:
   Status: completed
   Training episodes: 400
   Mean reward (last 50 episodes): 9.08
   Mean steps (last 50 episodes): 8.36
   Success rate (last 50 episodes): 94.0%
   Q-table stats: mean=3.596, max=12.000
   Learned policy:
      Episodes: 150
      Avg reward: 10.60
      Avg steps: 8.00
      Success rate: 100.0%
   Random baseline:
      Episodes: 150
      Avg reward: -30.38
      Avg steps: 49.72
      Success rate: 17.3%
   Improvement over baseline:
      Success-rate gain: 82.7%
      Avg-reward gain: 40.98
   Saved plots: data/runs/learning_curve.png, data/runs/success_rate_curve.png
Saved run artifact: data/runs/latest_rl_playground.json
```

---

## Build Order

Follow this order for clean architecture:

1. `storage.py`
2. `models.py`
3. `operations.py`
4. `display.py`
5. `main.py`

---
