---
layout: default
title:  Proposal
---

### Summary of the Project
Our project aims to explore and compare the effectiveness of reinforcement learning (RL) algorithms in the classic arcade-style game Tron, a fast-paced, grid-based environment that requires real-time decision making, spatial awareness, and long-term planning. In Tron, an agent controls a light cycle that moves continuously across a grid, leaving an intouchable trail behind. The agent must avoid colliding with walls, its own trail, and the opponent’s trail while attempting to outmaneuver the opponent and survive as long as possible. This environment poses a challenging learning problem due to its adversarial dynamics, delayed rewards, and rapidly changing game states.

The primary goals of this project are to:
(1) Design and train an agent capable of learning effective movement, trapping, and survival strategies directly from Tron game states.
(2) Perform a comparative analysis between value-based and policy-based reinforcement learning methods within this single, controlled environment. 

We will be using a custom implementation of Tron compatible with the Gymnasium API, allowing the agent to receive a grid-based or pixel-based observation as input. The output will be one of a fixed set of actions, typically corresponding to movement directions such as up, down, left, or right. Through this project, we aim to gain deeper insight into the strengths and limitations of different RL approaches when applied to fast-paced, adversarial arcade-style control problems.

### Project Goals
#### Minimum Goal
The minimum goal that our group wants to achieve is to implement a functional Tron game environment and train a basic reinforcement learning (RL) agent that is able to interact with the Tron light-cycle dynamics. The goal of this agent would be to achieve performance that is better than random movement, and will be measured by survival time and avoidance of collisions with walls and trails.

#### Realistic Goal
The realistic goal that we aim to achieve is to train multiple RL agents for the Tron environment and compare their performance against heuristic, rule-based baselines. These baselines may include strategies such as wall-following, open-space seeking, or simple opponent-avoidance rules. The RL agents should demonstrate incremental improvements in average survival time, win rate, and overall stability during training and evaluation.

#### Moonshot Goal
The “shot in the dark” goal for us is to develop a Tron agent that is able to learn long-term, strategic behaviors and skills such as trapping the opponent, controlling space, and anticipating future board states. We anticipate that near optimal performance would be determined by consistently outperforming strong heuristics and achieving high win rates in competitive Tron matches.

### AI/ML Algorithms
We anticipate performing a comparative analysis of value-based and policy-based reinforcement learning methods, including Deep Q-Networks (DQN), Double DQN (DDQN), and Policy Gradient algorithms such as Proximal Policy Optimization (PPO).

Our reward function will be centered on the core objective of the game, which is staying alive, and making the opponent crash. Because of this, we will have a large positive reward for winning, and a large negative reward for losing. However, to reduce reward sparsity, we also plan to incorporate smaller rewards, such as space controlled and distance from the opponent, which will be general heuristic based rewards that push the agents towards strong and logical gameplay. We will observe how these smaller rewards affect gameplay, and adjust as we see fit. 


### Evaluation Plan
To evaluate the success of our project, we will mainly look at our agent’s win rate versus baseline opponents. Some examples of baseline opponents are a random-action agent, and a simple heuristic agent that follows predefined rules, such as prioritizing open space and avoiding immediate collisions. We expect reinforcement learning agents to significantly outperform these baselines. We will also utilize secondary metrics, such as survival time and cumulative reward, in order to evaluate differences in dynamics between algorithms.

In addition to quantitative evaluation, we will conduct qualitative analysis to verify correct learning behavior. This will include testing the algorithms in simplified environments with reduced grid sizes to ensure proper reward propagation and action selection before scaling to the full game. We will visualize gameplay trajectories, plot learning curves, and observe trained agents playing the game step by step. Expected qualitative outcomes include smoother/more logical and strategic movement patterns, improved obstacle detection and avoidance, and reduced oscillatory or random behavior as training progresses. A successful moonshot outcome would be an agent demonstrating consistent long-term survival or achieving exceptionally high win rates against heuristic-based strategies.

### Meet the Instructor
We met with Professor Fox to go over our proposal on 01/22/2026. Through this meeting, we learned about using Monte Carlo Tree Search (MCTS) and received some advice about utilizing it in the training phase instead of making simulations at runtime of the game. 

### AI Tool Usage
We did not use any AI tools to assist in the writing of this proposal. Any future utilization of AI tools will be noted in the status and final reports that are submitted to the course staff.