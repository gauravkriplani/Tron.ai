---
layout: default
title:  Proposal
---

### Summary of the Project
Our project aims to explore and compare the effectiveness of reinforcement learning (RL) algorithms in the classic arcade-style game Tron, a fast-paced, grid-based environment that requires real-time decision making, spatial awareness, and long-term planning. In Tron, an agent controls a light cycle that moves continuously across a grid, leaving an intouchable trail behind. The agent must avoid colliding with walls, its own trail, and the opponent’s trail while attempting to outmaneuver the opponent and survive as long as possible. This environment poses a challenging learning problem due to its adversarial dynamics, delayed rewards, and rapidly changing game states.

The primary goals of this project are to (1) design and train an agent capable of learning effective movement, trapping, and survival strategies directly from Tron game states, and (2) perform a comparative analysis between value-based and policy-based reinforcement learning methods within this single, controlled environment. We will use a Python-based implementation of Tron compatible with the Gymnasium API, allowing the agent to receive a grid-based or pixel-based observation as input. The output will be one of a fixed set of actions, typically corresponding to movement directions such as up, down, left, or right. Through this project, we aim to gain deeper insight into the strengths and limitations of different RL approaches when applied to fast-paced, adversarial arcade-style control problems.

### Project Goals
#### Minimum Goal
The minimum goal that our group wants to achieve is to implement a functional Tron game environment and train a basic reinforcement learning (RL) agent that is able to interact with the Tron light-cycle dynamics. The goal of this RL agent is to achieve performance that is better than random movement, as measured by survival time and avoidance of collisions with walls and trails.

#### Realistic Goal
The realistic goal that our group wants to achieve is to train multiple RL agents for the Tron environment and compare their performance against heuristic, rule-based baselines. These baselines may include strategies such as wall-following, open-space seeking, or simple opponent-avoidance rules. The RL agents should demonstrate incremental improvements in average survival time, win rate, and overall stability during training and evaluation.

#### Moonshot Goal
The “shot in the dark” goal for our group is to develop a Tron agent that is able to learn long-term, strategic behaviors such as trapping the opponent, controlling space, and anticipating future board states. Near-optimal performance would be demonstrated by consistently outperforming strong human-designed heuristics and achieving high win rates in competitive Tron matches.

### AI/ML Algorithms

### Evaluation Plan

### Meet the Instructor
We met with Professor Fox to go over our proposal on 01/22/2026. Through this meeting, we learned about using Monte Carlo Tree Search (MCTS) and received some advice about utilizing it in the training phase instead of making simulations at runtime of the game. 

### AI Tool Usage
We did not use any AI tools to assist in the writing of this proposal. Any future utilization of AI tools will be noted in the status and final reports that are submitted to the course staff.