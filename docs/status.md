---
layout: default
title:  Status
---

### Project Summary

Our project explores and compares reinforcement learning (RL) approaches in the arcade-style game Tron, a fast-paced, grid-based environment demanding real-time decision making, spatial reasoning, and long-term planning. In Tron, each agent pilots a light cycle that continuously moves on a grid, leaving an impassable trail; agents must avoid collisions with walls, their own trail, and opponents while attempting to outmaneuver adversaries. The game's adversarial dynamics, delayed rewards, and rapidly changing states create a challenging RL problem.

We have (1) designed and trained agents that learn movement, trapping, and survival strategies directly from game observations and (2) performed a controlled comparison between value-based and policy-based methods, specifically Deep Q-Networks (DQN) for value-based learning and Proximal Policy Optimization (PPO) for policy-based learning. Our implementation adheres to the Gymnasium API and can provide either grid-based or pixel-based observations. We train and evaluate agents using metrics such as win rate, survival time, and cumulative reward, and compare against simple heuristic baselines. Through quantitative analysis and qualitative inspection of gameplay, we seek to highlight strengths and weaknesses of different RL approaches applied to adversarial, time-sensitive control tasks, informing best practices for similar arcade-style problems.

### Approach

#### Core Ideas

Since tron is a two-player competitive game, we will train our model incrementally from random to more well trained opponents. This will allow us to train results on more difficult and realistic bots rather than on repetition. So far, we stop at greedy opponents. We would like to achieve a higher win rate against space greedy opponents first before we continue to expand into more difficult opponents.

We are using a custom PyTorch implementation and a Gymnasium environment to run the DQN model. Training happens in `train_dqn.py` using our own QNetwork, ReplayBuffer, and dqn_loss from `dqn.py`. Our model evaluation is being done in `eval_dqn.py`.

In the current environment, we follow a modified setup of a 20x20 grid. Each time a player decides to go over a certain grid, that point is marked as taken and acts as an object that the players must avoid. There are 4 actions for each player: move up, down, left, and right. Each action is 1 move on the grid. The current reward is +1 for each step the agent survives, +10 for winning the game, and -10 for losing. We are still experimenting with this reward system and will revisit it more in the future.

#### Algorithms

**Deep Q-Network (DQN):** DQN is a reinforcement learning algorithm that acts as a combination between deep learning and Q-learning. It is used to solve complex reinforcement learning problems. We currently support 3 levels of architecture dependent on the size of the grid. When we have smaller grids, we use a Multi-Layer Perceptron (MLP) model. For larger grids, we use a Convolutional Neural Network (CNN) to capture spatial structure and predict the next best move. And we also are using a Dueling CNN where we separate value and advantage streams for better performance. The output is Q-values for all actions given an observation.

#### Evaluation

We evaluated our DQN agent against 2 heuristic baseline opponents: a random agent, which selects at random from a set of non-lethal actions, and a space-greedy agent, that greedily maximizes its Voronoi territory, Voronoi meaning the amount of free cells that the agent is closer to than its opponent. All evaluations are conducted on a 20x20 grid, over 500 games, with fixed random seeds for reproducibility. We measured two primary metrics: win rate (percentage of games where the opponent crashes first) and average episode length (how long the game lasted), which aims to capture the agent's survival skill.

![Win rate progression across 5 training stages](assets/Screenshot%202026-02-25%20at%208.39.59%20PM.png)

This figure shows the win rate progression across the 5 training stages. The baseline DQN agent, with 3 channel observation, only achieved a 60.4% win rate against the random opponent. The 6 channel agent added in three new channels that helped the agent understand its surroundings better, which raised its win rate to 85% against the random opponent. Finally, we introduced a Voronoi territory channel to the agent, which raised its win rate to 93.5% against the random opponent. All of these agents were trained against the random opponent.

At this point, we decided to move our sights onto the space-greedy agent, which our current agent was performing quite poorly against (15.5% win rate). We knew that to beat the space_greedy agent, we would need more than just observational improvements. And so we reevaluated our model architecture, rewards, and training stages. We started by expanding the CNN from 3 layers to 4 layers, with the hope that this would give the agent the ability to learn more complex patterns. We also added a small territory based reward, with the idea of giving the agent more nuanced feedback, although this reward could be seen as pushing the agent towards a certain strategy (space_greedy), and so we may remove it later. Finally, we decided to train this agent in two stages. We first trained the agent against the random opponent, in order to build basic survival skills. From there, we took that agent and trained it against the space greedy opponent. This two step approach made sure that the agent could establish fundamentals, to compete and meaningfully learn from facing a strong opponent.

![Game distribution of different agents](assets/Screenshot%202026-02-25%20at%208.40.43%20PM.png)

This figure shows the game distribution of the different agents we trained. All the games vs random seemed to average out to 35 steps, reflecting the random agent's poor survival skills. In contrast, the games against space_greedy's agent were significantly longer, and increased as our agent's skill increased, indicating that our model's survival skills were the bottleneck, and not space_greedy's.

### Remaining Goals and Challenges

As of writing, our current best agent achieves a 70% win rate against the space_greedy agent, and 96% win rate against the random agent. While this already exceeded our realistic goal of outperforming heuristic baselines, we believe that there is still room for improvement. The agent's plateau at 70% against space_greedy seems to be a limitation of DQN as an algorithm, as DQN seems to not have any sort of explicit lookahead. And so we plan to test Proximal Policy Optimization (PPO), as it's on policy learning and stochastic policy could capture patterns more advanced than DQN's deterministic greedy policy.

Another idea we plan to explore is self play training, where the agent trains against copies of itself, rather than fixed opponents that it can over optimize against. Self play forces the agent to continuously adapt, and could possibly encourage more sophisticated tactics and strategies. However, self play isn't a silver bullet, and can easily introduce its own set of challenges, such as training instability, or reward exploitation. For example, since the agent receives a reward for each step it survives, two copies of the agent could learn to avoid/delay confrontation entirely, maximizing mutual survival rather than competing to win. Addressing this will require careful analysis of the current reward system, in order to maximize performance.

### Resources Used

#### Tools & Libraries

- **NumPy:** Numerical computations and array-based operations
- **PyTorch:** Primary framework for implementing the DQN
- **Gymnasium:** Standard environment API
- **Pygame:** Graphical user interface

#### AI Chatbot: GitHub Copilot Chat in VS Code (Model: GPT-5.2)

**How it was used:**
- Assisted code completion
- Optimizing reward function structure
- Error message analysis
- Debugging assistance
- Markdown file editing guidance
- Outlining algorithms for understanding
- Discussing implementation approaches
