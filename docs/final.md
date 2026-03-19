---
layout: default
title:  Final Report
---

### Project Summary

This project investigates and compares reinforcement learning methods in Tron, an arcade style, grid based game that demands real-time decision making, spatial reasoning, and long-term planning. In Tron, each agent controls a light cycle that moves continuously and leaves an impassable trail. Agents must avoid collisions with walls, their own trails, and opponent trails while trying to outmaneuver the adversary. The game’s adversarial nature, sparse and delayed rewards, and rapidly changing state space make it a challenging benchmark for RL. In our work, we wanted to cover 2 core objectives. First, we design and train agents that learn movement, trapping, and survival behaviors directly from observations. Second, we perform a controlled comparison between value-based and policy-based approaches, using Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO), respectively. 

Our implementation follows the Gymnasium API and supports both grid-based and pixel observations. We evaluate performance with win rate, average survival time, and cumulative reward, and compare learned policies against heuristic baselines. To ensure fair comparison, we keep environment settings and evaluation protocols consistent across methods. We also analyze training stability, sample efficiency, and qualitative gameplay behavior to understand not only which approach performs better, but why. These findings highlight the strengths and limitations of each method for adversarial, time-sensitive control tasks and provide practical guidance for applying RL to similar arcade-style domains.


### Approaches

#### Deep Q-Networks (DQN)

DQN combines Q-learning with neural networks to approximate the action-value function. The key innovation is experience replay: transitions are buffered and sampled uniformly for training.

**Data & Loss:**
- Replay buffer: 200k transitions of (state, action, reward, next_state, done)
- Minibatch sampling: 128 transitions per training step
- Loss: Huber loss with Double DQN and Dueling architecture

$$\mathcal{L}(\theta) = \mathbb{E} \left[ \mathcal{H} \left( r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a')) - Q_\theta(s, a) \right) \right]$$

Dueling split: $Q(s,a) = V(s) + (A(s,a) - \text{mean}(A))$

**Tron Application:**
- Observation: 7-channel grid (20×20); channels include blocked cells, agent/opponent heads, trails, BFS distance to obstacles, reachable area (flood-fill), and Voronoi territory
- Actions: 4 discrete (UP, RIGHT, DOWN, LEFT); masked during exploration
- Rewards: +1 survival, +10 win, -10 loss, +0.1 × Voronoi territory bonus
- Training: 2M total steps; 20k warm-up; train every 4 steps; target update every 10k steps; ε decay 1.0→0.05 over 1M steps

**Key Hyperparameters:**
- Buffer: 200k | Batch: 128 | $\gamma$: 0.99 | Learning rate: 1e-4
- ε schedule: 1.0→0.05 over 1M steps | PER α: 0.6, β start: 0.4

**Architecture:** 4-layer CNN (32, 64, 128, 128 filters) → FC 512 → dueling heads (value + advantage)

---

#### Proximal Policy Optimization (PPO)

PPO learns a stochastic policy and value function via clipped policy gradients. It is on-policy: trajectories are collected fresh each update.

**Data & Loss:**
- Rollout buffer: 2,048 steps, then 10 epochs over minibatches of 64
- Advantage estimation: Generalized Advantage Estimation (GAE) with λ=0.95

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V, \quad \delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$$

PPO loss:
$$\mathcal{L} = \mathbb{E}_t \left[ \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t) - c_e \mathcal{H}[\pi] + c_v \mathcal{L}^V \right]$$

where $r_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$ and $\epsilon=0.2$.

**Tron Application:**
- Observation, actions, rewards: identical to DQN
- Action masking applied at policy sampling (unsafe actions → logits = -∞)
- Training: 2M total steps; 2,048 step rollouts; 10 update epochs per rollout

**Key Hyperparameters:**
- Rollout: 2,048 | Minibatch: 64 | Epochs: 10 | $\gamma$: 0.99 | λ: 0.95
- Learning rate: 3e-4 | Entropy coef: 0.01 | Value coef: 0.5 | Clip: 0.2
- Gradient norm clip: 0.5

**Architecture:** Shared 4-layer CNN (same as DQN) → FC 512 → split into actor (4 logits) and critic (1 value) heads

---

#### Curriculum Training

Both algorithms trained in two phases:
1. **Phase 1:** Train vs. RandomOpponent for 1M steps (basic survival)
2. **Phase 2:** Load Phase 1 checkpoint; train vs. SpaceGreedyOpponent for 1M steps

Direct training against SpaceGreedyOpponent from scratch failed to converge, confirming curriculum necessity.

---

#### Action Masking & Observation Design

**Action Masking:** Both algorithms prevent selecting moves causing immediate collision. DQN samples only from safe actions during exploration; PPO sets unsafe action logits to -∞.

**7-Channel Observation:**
- Ch 0: Blocked cells (walls + trails)
- Ch 1-2: Agent & opponent head positions
- Ch 3: Agent trail
- Ch 4: BFS distance to nearest obstacle (normalized)
- Ch 5: Flood-fill reachable area
- Ch 6: Voronoi territory (who is closer to each cell)

This rich representation provides spatial, reachability, and territorial information, significantly outperforming simpler observations.

### Final Results

### Evaluation

### Resources Used

**Tools & Libraries:**
- NumPy: numerical computations and array operations (Harris et al., 2020)
- PyTorch: neural network implementation and deep learning (Paszke et al., 2019)
- Gymnasium: standard RL environment API (Towers et al., 2023)
- Pygame: visualization and human-vs-AI gameplay (Pygame Community, n.d.)

**Algorithmic References:**
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv preprint arXiv:1312.5602.
- Mnih, V., Badia, A. P., Mirza, M., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. In International Conference on Machine Learning (ICML).
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. In AAAI Conference on Artificial Intelligence.
- Wang, Z., de Freitas, N., & Lanctot, M. (2016). Dueling Network Architectures for Deep Reinforcement Learning. In International Conference on Machine Learning (ICML).
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Openai, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. In International Conference on Learning Representations (ICLR).

**CleanRL Implementation Defaults:**
- Implementation references and default hyperparameters from CleanRL: Clean, Single-File Implementations of Deep Reinforcement Learning Algorithms. https://github.com/vwxyzjn/cleanrl

**AI Tool Usage:**
- ChatGPT and Claude: code assistance, algorithm explanation, error analysis, and report writing


