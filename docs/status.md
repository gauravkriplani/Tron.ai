---
layout: default
title:  Status
---

### Project Summary

Our project explores and compares reinforcement learning (RL) approaches in the arcade-style game Tron, a fast-paced, grid-based environment demanding real-time decision making, spatial reasoning, and long-term planning. In Tron, each agent pilots a light cycle that continuously moves on a grid, leaving an impassable trail; agents must avoid collisions with walls, their own trail, and opponents while attempting to outmaneuver adversaries. The game’s adversarial dynamics, delayed rewards, and rapidly changing states create a challenging RL problem.

We have (1) designed and trained agents that learn movement, trapping, and survival strategies directly from game observations and (2) performed a controlled comparison between value-based and policy-based methods, specifically Deep Q-Networks (DQN) for value-based learning and Proximal Policy Optimization (PPO) for policy-based learning. Our implementation adheres to the Gymnasium API and can provide either grid-based or pixel-based observations; actions map to discrete movement directions. We train and evaluate agents using metrics such as win rate, survival time, and cumulative reward, and compare against simple heuristic baselines. Through quantitative analysis and qualitative inspection of gameplay, we seek to highlight strengths and weaknesses of different RL approaches applied to adversarial, time-sensitive control tasks, informing best practices for similar arcade-style problems.
