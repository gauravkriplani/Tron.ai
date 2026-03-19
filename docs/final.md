---
layout: default
title:  Final Report
---

### Project Summary

This project investigates and compares reinforcement learning methods in Tron, an arcade style, grid based game that demands real-time decision making, spatial reasoning, and long-term planning. In Tron, each agent controls a light cycle that moves continuously and leaves an impassable trail. Agents must avoid collisions with walls, their own trails, and opponent trails while trying to outmaneuver the adversary. The game’s adversarial nature, sparse and delayed rewards, and rapidly changing state space make it a challenging benchmark for RL. In our work, we wanted to cover 2 core objectives. First, we design and train agents that learn movement, trapping, and survival behaviors directly from observations. Second, we perform a controlled comparison between value-based and policy-based approaches, using Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO), respectively. 

Our implementation follows the Gymnasium API and supports both grid-based and pixel observations. We evaluate performance with win rate, average survival time, and cumulative reward, and compare learned policies against heuristic baselines. To ensure fair comparison, we keep environment settings and evaluation protocols consistent across methods. We also analyze training stability, sample efficiency, and qualitative gameplay behavior to understand not only which approach performs better, but why. These findings highlight the strengths and limitations of each method for adversarial, time-sensitive control tasks and provide practical guidance for applying RL to similar arcade-style domains.


### Approaches

#### Deep Q-Networks (DQN)

**Algorithmic Overview**

Deep Q-Networks combine Q-learning with deep neural networks to approximate the optimal action-value function $Q(s,a)$. The key innovation is experience replay: transitions are stored in a buffer and sampled uniformly for training, breaking temporal correlations in the data stream.

**Data Structure**

The replay buffer stores transitions $(s_t, a_t, r_t, s_{t+1}, \text{done}_t)$ in a circular numpy array of fixed capacity. We use a capacity of 200,000 transitions. For each transition:
- State $s_t$: 7-channel tensor of shape (7, 20, 20)
- Action $a_t$: integer in [0, 3]
- Reward $r_t$: scalar float
- Next state $s_{t+1}$: 7-channel tensor
- Done flag: whether episode terminated

**Sampling Strategy**

Each training step, we sample a minibatch of 128 transitions uniformly at random from the buffer. For Prioritized Experience Replay (optional), we assign priorities $p_i = (|\delta_i| + \epsilon)^\alpha$ to each transition based on TD error $\delta_i$, then sample proportionally to these priorities. Importance-sampling weights correct for distribution shift: $w_i = (N \cdot P(i))^{-\beta}$ where $N$ is buffer size and $P(i)$ is the sampling probability.

**Loss Function**

The Huber loss for standard DQN is:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{B}} \left[ \mathcal{H} \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') \cdot (1-d) - Q_\theta(s, a) \right) \right]$$

where $\mathcal{H}(x) = \begin{cases} \frac{1}{2}x^2 & \text{if } |x| \le 1 \\ |x| - \frac{1}{2} & \text{otherwise} \end{cases}$, $\theta$ are online network weights, $\theta^-$ are target network weights (updated every 10k steps), and $d$ is the done flag.

We employ Double DQN (van Hasselt et al., 2016) to reduce overestimation bias by decoupling action selection from value estimation:
$$y = r + \gamma \, Q_{\theta^-}(s',\, \arg\max_{a'} Q_\theta(s', a')) \cdot (1-d)$$

Additionally, we use Dueling DQN (Wang et al., 2016), which splits the Q-network output into value and advantage streams:
$$Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|A|} \sum_{a'} A(s,a') \right)$$

This decomposition allows the network to separately learn how good each state is and which actions are relatively better within that state. With PER, losses are weighted by the importance-sampling weights: $\mathcal{L} \leftarrow w_i \cdot \mathcal{L}$.

**Application to Tron**

*Observation Space:*
- Channel 0: Blocked cells (walls, trails from both agents)
- Channel 1: Agent head position (one-hot)
- Channel 2: Opponent head position (one-hot)
- Channel 3: Agent's own trail (one-hot)
- Channel 4: BFS distance to nearest obstacle, normalized to [0, 1]
- Channel 5: Flood-fill reachable area from agent head (one-hot)
- Channel 6: Voronoi territory (1 if agent closer, 0 if opponent closer, 0.5 if tied)

*Action Space:* Discrete(4) representing UP, RIGHT, DOWN, LEFT movements. Reverse moves and instantly lethal moves are masked during $\epsilon$-greedy exploration.

*Reward Structure:*
- +1 for each step survived
- +10 for winning (opponent crashes)
- -10 for losing (agent crashes)
- +0.1 × (agent_territory - opponent_territory) / total_cells for Voronoi shaping (small per-step bonus)

*Training Details:*
- Total environment steps: 2,000,000
- Warm-up steps before learning: 20,000
- Training frequency: every 4 steps
- Target network hard-update: every 10,000 steps
- Exploration schedule: $\epsilon_t = \max(0.05, 1.0 - t / 1,000,000)$, decaying linearly from 1.0 to 0.05 over 1M steps
- Discount factor: $\gamma = 0.99$
- Learning rate: $1 \times 10^{-4}$ (Adam optimizer)

*Hyperparameters (with citations):*

| Parameter | Value | Source / Notes |
|---|---|---|
| Replay buffer capacity | 200,000 | Standard for mid-scale tasks (Mnih et al., 2015 used 1M for Atari) |
| Batch size | 128 | Increased from standard 32 (Mnih et al., 2015) for stability |
| Discount $\gamma$ | 0.99 | Standard default (Schulman et al., 2017) |
| Learning rate | $1 \times 10^{-4}$ | Standard for Adam-based DQN |
| Warm-up steps | 20,000 | Tuned to ensure buffer filled before learning |
| Train frequency | every 4 steps | Standard (Mnih et al., 2015) |
| Target update | 10,000 steps | Standard hard-update interval |
| $\varepsilon$ start / end | 1.0 / 0.05 | Standard; decay schedule tuned experimentally |
| PER $\alpha$ | 0.6 | Default (Schaul et al., 2016) |
| PER $\beta$ start | 0.4 | Default (Schaul et al., 2016); anneals to 1.0 |

**Network Architecture**

CNN backbone: 4 convolutional layers (32, 64, 128, 128 filters, 3×3 kernels) with stride 2 for spatial downsampling, followed by flattening and a 512-unit fully connected layer. Dueling heads split into 1D value head and 4D advantage head (one per action). ReLU activations between layers.

---

#### Proximal Policy Optimization (PPO)

**Algorithmic Overview**

PPO (Schulman et al., 2017) is an on-policy algorithm that learns a stochastic policy $\pi(a|s)$ and value function $V(s)$ via clipped policy gradient optimization. Unlike DQN, PPO directly optimizes the policy and collects fresh trajectories for each update.

**Data Structure**

The rollout buffer stores one complete trajectory of 2,048 steps:
- Observations $s_t$: (7, 20, 20) tensors
- Actions $a_t$: integer in [0, 3]
- Log-probabilities $\log\pi(a_t|s_t)$: scalar float
- Rewards $r_t$: scalar float
- Done flags $d_t$: boolean
- Values $V(s_t)$: scalar float
- Action masks: (4,) array indicating safe actions

After collecting a rollout, we compute discounted returns and advantage estimates using Generalized Advantage Estimation (GAE), then shuffle and split into minibatches of 64 for 10 epochs of gradient updates.

**Sampling Strategy**

All data comes from the current policy $\pi_\theta$ during rollout collection. After rollout, we reuse this data for multiple epochs (10 epochs over 2,048 steps = ~320k gradient updates per 2,048 environment steps). Within each epoch, we shuffle and create minibatches for more stable updates.

**Loss Function and Advantage Estimation**

GAE computes advantages with exponential moving average of TD residuals:
$$\hat{A}_t^{(\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$
where $\delta_t^V = r_t + \gamma V(s_{t+1}) \cdot (1-d_t) - V(s_t)$ is the TD residual, $\gamma = 0.99$, and $\lambda = 0.95$. Returns are computed as $\hat{R}_t = \hat{A}_t + V(s_t)$.

The PPO loss combines three components:
$$\mathcal{L}^{\text{PPO}} = \mathbb{E}_t \left[ \mathcal{L}^{\text{CLIP}} - c_e \mathcal{H}[\pi_\theta] + c_v \mathcal{L}^V \right]$$

**Clipped Policy Loss:**
$$\mathcal{L}^{\text{CLIP}} = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio and $\epsilon = 0.2$ is the clip range. This constrains policy updates to prevent large, destabilizing changes.

**Value Loss (clipped MSE):**
$$\mathcal{L}^V = \frac{1}{2} \mathbb{E}_t \left[ \max \left( (V_\theta(s_t) - \hat{R}_t)^2,\, (\text{clip}(V_\theta(s_t), V_{\theta_{\text{old}}}(s_t) - \epsilon, V_{\theta_{\text{old}}}(s_t) + \epsilon) - \hat{R}_t)^2 \right) \right]$$

**Entropy Bonus:**
$$\mathcal{H}[\pi_\theta] = \mathbb{E}_t \left[ -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right]$$
encourages exploration with coefficient $c_e = 0.01$. Value loss has coefficient $c_v = 0.5$.

**Application to Tron**

*Observation, Action, and Reward Spaces:* Identical to DQN (same 7-channel observations, 4 discrete actions, same reward structure).

*Action Masking:* Safe actions are identified via the same heuristic as DQN (no instantly lethal moves). During policy sampling, unsafe action logits are set to $-\infty$, so $\text{softmax}$ assigns them zero probability. The mask is stored in the rollout buffer and re-applied during updates.

*Training Details:*
- Total environment steps: 2,000,000
- Rollout length: 2,048 steps per collection cycle
- Number of update epochs per rollout: 10
- Minibatch size: 64
- Discount $\gamma = 0.99$, GAE $\lambda = 0.95$
- Learning rate: $3 \times 10^{-4}$ (Adam optimizer, $\epsilon = 10^{-5}$)
- Gradient norm clipping: 0.5

*Hyperparameters (with citations):*

| Parameter | Value | Source / Notes |
|---|---|---|
| Rollout length | 2,048 | Default from CleanRL / Schulman et al., 2017 |
| Minibatch size | 64 | Default from CleanRL |
| Update epochs | 10 | Default from CleanRL |
| Discount $\gamma$ | 0.99 | Standard (Schulman et al., 2017) |
| GAE $\lambda$ | 0.95 | Default (Schulman et al., 2017) |
| Clip coefficient $\epsilon$ | 0.2 | Standard (Schulman et al., 2017) |
| Entropy coef $c_e$ | 0.01 | Default from CleanRL; smaller value reduces over-exploration |
| Value loss coef $c_v$ | 0.5 | Default from CleanRL |
| Learning rate | $3 \times 10^{-4}$ | Default Adam lr from CleanRL |
| Adam $\epsilon$ | $1 \times 10^{-5}$ | Default from CleanRL (tighter than PyTorch default) |
| Max gradient norm | 0.5 | Default from CleanRL |

**Network Architecture**

Shared CNN backbone (identical to DQN) outputs a 512-dimensional feature vector. Two heads diverge: actor head outputs 4 logits (one per action), and critic head outputs a single scalar value estimate. Orthogonal weight initialization (std=√2 for actor, std=1.0 for critic) as per OpenAI baselines.

---

#### Curriculum Training

Both DQN and PPO were trained using a two-phase curriculum:

1. **Phase 1:** Train against `RandomOpponent` for 1,000,000 steps. This builds basic survival and evasion skills.
2. **Phase 2:** Load the Phase 1 checkpoint and train against `SpaceGreedyOpponent` for an additional 1,000,000 steps. This adapts the agent to a skilled, territorial opponent.

Direct training against `SpaceGreedyOpponent` from scratch failed to converge, confirming that curriculum learning is critical for this problem (as also noted in the status report).

---

#### Key Implementation Details

**Action Masking Strategy**

Both algorithms use action masking to prevent selecting moves that result in immediate collision:
```
safe_actions = []
for action in [UP, RIGHT, DOWN, LEFT]:
    next_head = agent_head + action_delta
    if next_head is not blocked:
        safe_actions.append(action)
return safe_actions if safe_actions else [all actions]
```

This ensures the agent never learns to repeatedly crash into walls. The mask is applied:
- *DQN:* During $\epsilon$-greedy exploration, only sample from safe actions
- *PPO:* Set logits of unsafe actions to $-\infty$ before sampling from Categorical distribution

**Observation Feature Engineering**

The 7-channel representation was designed to provide complementary spatial information:
- **Reachability (Ch. 4, 5):** BFS distance and flood-fill allow the agent to estimate free space without explicit lookahead
- **Territory (Ch. 6):** Voronoi mapping provides fast feedback on relative space control, encouraging territorial strategies
- **Positional (Ch. 0-3):** Direct encoding of game state (walls, head positions, trails)

This rich observation space significantly outperformed hand-crafted heuristics and enabling both algorithms to learn competitive strategies.

---

### Evaluation

#### DQN Performance

#### PPO Performance

### Key Findings

### Technical Highlights

### Challenges and Limitations

### Conclusions

### Recommendations for Future Work

### Resources
