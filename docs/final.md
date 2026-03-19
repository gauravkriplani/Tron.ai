---
layout: default
title:  Final Report
---

### Project Summary

This project investigates and compares reinforcement learning methods in Tron, an arcade style, grid based game that demands real-time decision making, spatial reasoning, and long-term planning. In Tron, each agent controls a light cycle that moves continuously and leaves an impassable trail. Agents must avoid collisions with walls, their own trails, and opponent trails while trying to outmaneuver the adversary. The game’s adversarial nature, sparse and delayed rewards, and rapidly changing state space make it a challenging benchmark for RL. In our work, we wanted to cover 2 core objectives. First, we design and train agents that learn movement, trapping, and survival behaviors directly from observations. Second, we perform a controlled comparison between value-based and policy-based approaches, using Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO), respectively. 

Our implementation follows the Gymnasium API and supports both grid-based and pixel observations. We evaluate performance with win rate, average survival time, and cumulative reward, and compare learned policies against heuristic baselines. To ensure fair comparison, we keep environment settings and evaluation protocols consistent across methods. We also analyze training stability, sample efficiency, and qualitative gameplay behavior to understand not only which approach performs better, but why. These findings highlight the strengths and limitations of each method for adversarial, time-sensitive control tasks and provide practical guidance for applying RL to similar arcade-style domains.


### Approaches

#### Overview

Our approach can be broken down into the implementation and evaluation of two reinforcement learning algorithms under the same success criteria: `DQN` and `PPO`. They were selected because of their known strengths in solving similar adversarial and sequential decision-making tasks. We compare our learned agents against heuristic baselines to measure real progress in strategy learning.

**Baseline:** Our baseline model uses a `RandomOpponent` that selects uniformly from non-lethal moves each turn. This strategy tends to succeed early when the board is sparse, but fails to create strategic order as the game progresses. This serves as a reliable baseline to verify that our learned models are using long-term planning and achieving success in later game stages.

---

#### Deep Q-Networks (DQN)

Deep Q-Networks (Mnih et al., 2013) combine value-based Q-learning with deep neural networks to handle high-dimensional state spaces. The core idea is to learn an action-value function $Q(s,a)$ that estimates the expected cumulative reward for taking action $a$ in state $s$. The key algorithm is **experience replay**: past transitions are stored in a buffer and sampled uniformly for training, which breaks temporal correlations in the data stream and improves sample efficiency.

We use Double DQN (van Hasselt et al., 2016) to mitigate overestimation bias and Dueling DQN (Wang et al., 2016) to separately learn state value and action advantages. These enhancements allow the network to learn more robust value estimates.

**Data Structure**

The replay buffer stores transitions in a circular array:

| Component | Shape / Type | Description |
|---|---|---|
| State $s_t$ | (7, 20, 20) | 7-channel grid observation |
| Action $a_t$ | int ∈ [0, 3] | Discrete action (UP, RIGHT, DOWN, LEFT) |
| Reward $r_t$ | float | Immediate reward |
| Next state $s_{t+1}$ | (7, 20, 20) | Resulting state |
| Done flag $d_t$ | bool | Episode termination indicator |

Buffer capacity: 200,000 transitions (standard for mid-scale tasks; Mnih et al., 2015 used 1M for Atari).

**Data Sampling**

Each training iteration, we sample a minibatch of 128 transitions uniformly at random:

```
function SampleMinibatch(replay_buffer, batch_size):
    indices = random_integers(0, len(replay_buffer), batch_size)
    batch = replay_buffer[indices]
    return batch.states, batch.actions, batch.rewards, batch.next_states, batch.dones
```

For Prioritized Experience Replay (optional enhancement), transitions are sampled proportionally to their TD error: $P(i) \propto (|\delta_i| + \epsilon)^\alpha$, with importance-sampling correction weights $w_i = (N \cdot P(i))^{-\beta}$ (Schaul et al., 2016; $\alpha=0.6$, $\beta=0.4$).

**Loss Equation**

DQN minimizes the Huber loss (Huber, 1964) to handle outliers:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',d)} \left[ \mathcal{H} \Big( r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a')) \cdot (1-d) - Q_\theta(s, a) \Big) \right]$$

where $\mathcal{H}(x) = \begin{cases} \frac{1}{2}x^2 & |x| \le 1 \\ |x| - \frac{1}{2} & \text{else} \end{cases}$

- $Q_\theta$: online network weights
- $Q_{\theta^-}$: target network weights (updated every 10,000 steps)
- $\gamma = 0.99$: discount factor (standard; Sutton & Barto, 2018)
- $d$: done flag (masks future rewards at episode end)

**Dueling architecture** (Wang et al., 2016) decomposes the Q-value:

$$Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|A|}\sum_{a'} A(s,a') \right)$$

This allows separate learning of state value $V(s)$ and action advantages $A(s,a)$, improving stability and generalization.

**Tron Application**

*Observation Space:* 7-channel grid (20×20 pixels):
- Ch 0: Blocked cells (walls + both agents' trails)
- Ch 1: Agent head position (one-hot encoded)
- Ch 2: Opponent head position (one-hot encoded)
- Ch 3: Agent's own trail (one-hot encoded)
- Ch 4: Normalized BFS distance to nearest obstacle [0, 1]
- Ch 5: Reachable area via flood-fill from agent head
- Ch 6: Voronoi territory (1 if agent closer, 0 if opponent closer, 0.5 if tied)

*Action Space:* Discrete(4) representing {UP, RIGHT, DOWN, LEFT}. Reverse moves (immediate 180° turn) and instantly lethal moves are masked during exploration via the action masking heuristic.

*Reward Structure:*
- $+1$ per step survived
- $+10$ for winning (opponent crashes)
- $-10$ for losing (agent crashes)
- $+0.1 \times \frac{\text{agent territory} - \text{opp territory}}{\text{total cells}}$ (Voronoi shaping bonus, per step)

*Training Schedule:*
- Total environment steps: 2,000,000
- Warm-up steps before training: 20,000 (allow buffer to accumulate)
- Training frequency: every 4 steps (standard; Mnih et al., 2015)
- Target network hard-update: every 10,000 steps
- Total training duration: ~500k gradient updates

*Hyperparameters (with Sources):*

| Parameter | Value | Source / Justification |
|---|---|---|
| Buffer capacity | 200,000 | Standard for mid-scale (Mnih et al., 2015 used 1M for Atari) |
| Batch size | 128 | Increased from 32 (Mnih et al., 2015) for stability on smaller grid |
| Discount $\gamma$ | 0.99 | Standard default (Sutton & Barto, 2018) |
| Learning rate | $1 \times 10^{-4}$ | Standard for Adam optimizer in DQN literature |
| Warm-up steps | 20,000 | Tuned to ensure 10% buffer filled before learning begins |
| Train frequency | Every 4 steps | Standard (Mnih et al., 2015) |
| Target update interval | 10,000 steps | Standard hard-update interval (Mnih et al., 2015) |
| $\varepsilon$ start | 1.0 | Standard; full exploration initially |
| $\varepsilon$ end | 0.05 | Tuned; lower than typical 0.1 to encourage exploitation of learned policy |
| $\varepsilon$ decay duration | 1,000,000 steps | Tuned empirically; decays over ~50% of total training |
| PER $\alpha$ | 0.6 | Default (Schaul et al., 2016) |
| PER $\beta$ start | 0.4 | Default; anneals to 1.0 over training (Schaul et al., 2016) |

**Network Architecture**

CNN backbone: 4 convolutional layers (ReLU activations):
- Conv(in=7, out=32, kernel=3×3, stride=2, padding=1)
- Conv(in=32, out=64, kernel=3×3, stride=2, padding=1)
- Conv(in=64, out=128, kernel=3×3, stride=2, padding=1)
- Conv(in=128, out=128, kernel=3×3, stride=2, padding=1)
- Flatten → FC(512, ReLU)

Dueling heads:
- Value head: FC(512 → 1)
- Advantage head: FC(512 → 4)
- Combined: $Q = V + (A - \text{mean}(A))$

---

#### Proximal Policy Optimization (PPO)

PPO (Schulman et al., 2017) is a policy gradient method that learns a stochastic policy $\pi_\theta(a|s)$ and value function $V_\theta(s)$ via clipped policy objectives. Unlike value-based methods like DQN, PPO is **on-policy**: it collects fresh trajectories from the current policy for each update. The clipping mechanism constrains policy updates to prevent large, destabilizing changes, which stabilizes training compared to earlier policy gradient methods.

PPO is particularly well-suited for Tron because the stochastic policy naturally explores different strategies during rollout collection, potentially discovering emergent trapping or territorial behaviors that greedy value functions might miss.

**Data Structure**

The rollout buffer stores one complete trajectory of 2,048 environment steps:

| Component | Shape / Type | Description |
|---|---|---|
| Observation $s_t$ | (7, 20, 20) | 7-channel grid observation |
| Action $a_t$ | int ∈ [0, 3] | Sampled action |
| Log-probability $\log \pi(a_t\|s_t)$ | float | Log policy probability (for importance correction) |
| Reward $r_t$ | float | Immediate reward |
| Done flag $d_t$ | bool | Episode termination |
| Value estimate $V(s_t)$ | float | Critic's state value |
| Action mask $m_t$ | [4] bool | Safe actions during rollout |

After each rollout, we compute returns and advantages, then shuffle and split into minibatches of 64 for 10 epochs of gradient updates.

**Data Sampling**

Data is collected on-policy from rollout trajectories:

```
function CollectRollout(env, policy, num_steps):
    observations, actions, rewards, log_probs, values, dones, masks = []
    obs, _ = env.reset()
    
    for t in 1 to num_steps:
        action, log_prob, value, mask = policy.select_action(obs)
        next_obs, reward, done, _ = env.step(action)
        
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)
        dones.append(done)
        masks.append(mask)
        
        obs = next_obs
        if done: obs, _ = env.reset()
    
    return observations, actions, rewards, log_probs, values, dones, masks
```

After rollout, we compute advantages using GAE (Schulman et al., 2016), shuffle, and create minibatches.

**Loss Equation and Advantage Estimation**

Generalized Advantage Estimation (GAE; Schulman et al., 2016) computes advantages:

$$\hat{A}_t^{(\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$$

with $\gamma = 0.99$ and $\lambda = 0.95$ (default from Schulman et al., 2017). Returns: $\hat{R}_t = \hat{A}_t + V(s_t)$.

PPO loss (Schulman et al., 2017):

$$\mathcal{L}_t^{\text{PPO}} = \mathbb{E}_t \left[ \mathcal{L}_t^{\text{CLIP}} - c_e \mathcal{H}[\pi_\theta(s_t)] + c_v \mathcal{L}_t^V \right]$$

where:

**Clipped Policy Loss:**
$$\mathcal{L}_t^{\text{CLIP}} = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ is the probability ratio; $\epsilon = 0.2$ is the clip range.

**Value Loss (clipped MSE):**
$$\mathcal{L}_t^V = \frac{1}{2} \mathbb{E}_t \left[ \max \Big( (V_\theta(s_t) - \hat{R}_t)^2,\, (\text{clip}(V_\theta(s_t), V_{\text{old}}(s_t) - \epsilon, V_{\text{old}}(s_t) + \epsilon) - \hat{R}_t)^2 \Big) \right]$$

**Entropy Bonus:**
$$\mathcal{H}[\pi_\theta] = \mathbb{E}_t \left[ -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t) \right]$$

Encourages exploration with coefficient $c_e = 0.01$.

**Tron Application**

*Observation, Actions, Rewards:* Identical to DQN (7-channel grid, 4 discrete actions, same reward structure).

*Action Masking:* Safe actions identified via the same collision-checking heuristic as DQN. During policy sampling, unsafe action logits are set to $-\infty$, so softmax assigns them zero probability. Masks are stored in the rollout buffer and re-applied during updates.

*Training Schedule:*
- Total environment steps: 2,000,000
- Rollout length per update: 2,048 steps
- Update epochs per rollout: 10
- Number of updates: 2,000,000 / 2,048 ≈ 976 total update cycles
- Total gradient updates: 976 × 10 epochs × (2,048 / 64 minibatch size) ≈ 312,000 updates

*Hyperparameters (with Sources):*

| Parameter | Value | Source / Justification |
|---|---|---|
| Rollout length | 2,048 | Default from CleanRL / Schulman et al., 2017 |
| Minibatch size | 64 | Default from CleanRL |
| Update epochs per rollout | 10 | Default from CleanRL; balances stability vs. sample reuse |
| Discount $\gamma$ | 0.99 | Standard (Sutton & Barto, 2018) |
| GAE $\lambda$ | 0.95 | Default (Schulman et al., 2016) |
| Clip coefficient $\epsilon$ | 0.2 | Standard (Schulman et al., 2017) |
| Entropy coef $c_e$ | 0.01 | Default from CleanRL; smaller value = less over-exploration |
| Value loss coef $c_v$ | 0.5 | Default from CleanRL |
| Learning rate | $3 \times 10^{-4}$ | Default Adam lr from CleanRL |
| Adam $\epsilon$ | $1 \times 10^{-5}$ | Default from CleanRL (tighter than PyTorch default 1e-8) |
| Max gradient norm | 0.5 | Default from CleanRL; prevents gradient explosions |

**Network Architecture**

Shared CNN backbone (identical to DQN, outputs 512 features) → two independent heads:
- **Actor head:** FC(512 → 4 logits) with orthogonal initialization (std=$\sqrt{2}$)
- **Critic head:** FC(512 → 1 value) with orthogonal initialization (std=1.0)

Orthogonal weight initialization follows OpenAI baselines recommendations (Dhariwal et al., 2017).

---

#### Curriculum Training

Both DQN and PPO were trained using a two-phase curriculum (inspired by the status report findings):

1. **Phase 1 (1M steps):** Train against `RandomOpponent` to build basic survival and evasion reflexes.
2. **Phase 2 (1M steps):** Load Phase 1 checkpoint and train against `SpaceGreedyOpponent`, which greedily maximizes reachable area (Voronoi territory).

This curriculum was critical: direct training against `SpaceGreedyOpponent` from scratch failed to converge, confirming that agents benefit from learning fundamentals on weaker opponents before adapting to skilled adversaries.

---

#### Action Masking and Observation Design

**Action Masking Heuristic:** Both algorithms prevent selecting moves causing immediate collision (hitting walls or trails):

```
function GetSafeActions(obs):
    blocked_cells = obs[0]
    agent_head = extract_position(obs[1])
    safe_actions = []
    
    for action in [UP, RIGHT, DOWN, LEFT]:
        next_head = agent_head + action_delta[action]
        if in_bounds(next_head) and not blocked_cells[next_head]:
            safe_actions.append(action)
    
    return safe_actions if non-empty else [all actions]
```

This prevents trivial learning failures where agents spend countless steps crashing into walls.

**7-Channel Observation Space:** Designed to provide complementary spatial signals:

| Channel | Content | Purpose |
|---|---|---|
| 0 | Blocked cells | Direct state representation |
| 1-2 | Agent & opponent heads | Immediate position awareness |
| 3 | Agent trail | Own movement history |
| 4 | BFS distance to obstacle | Reachability without lookahead |
| 5 | Flood-fill reachable area | Free space from agent's perspective |
| 6 | Voronoi territory | Relative space control metric |

This rich representation significantly outperformed simpler 3-channel baselines (blocked, self head, opponent head) and provided agents with strategic information for long-term planning.

### Final Results

### Resources Used

**Tools & Libraries:**
- NumPy: numerical computations and array operations (Harris et al., 2020)
- PyTorch: neural network implementation and deep learning (Paszke et al., 2019)
- Gymnasium: standard RL environment API (Towers et al., 2023)
- Pygame: visualization and human-vs-AI gameplay (Pygame Community, n.d.)

**Algorithmic References:**
- Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics, 35(1), 73-101.
- Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv preprint arXiv:1312.5602.
- Mnih, V., Badia, A. P., Mirza, M., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. In International Conference on Machine Learning (ICML).
- van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. In AAAI Conference on Artificial Intelligence.
- Wang, Z., de Freitas, N., & Lanctot, M. (2016). Dueling Network Architectures for Deep Reinforcement Learning. In International Conference on Machine Learning (ICML).
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. In International Conference on Learning Representations (ICLR).
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. In International Conference on Learning Representations (ICLR).
- Dhariwal, P., Hesse, C., Klimov, O., et al. (2017). OpenAI Baselines. https://github.com/openai/baselines
- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

**CleanRL Implementation Defaults:**
- Implementation references and default hyperparameters from CleanRL: Clean, Single-File Implementations of Deep Reinforcement Learning Algorithms. https://github.com/vwxyzjn/cleanrl

**AI Tool Usage:**
- ChatGPT and Claude: code assistance, algorithm explanation, error analysis, and report grammar check


