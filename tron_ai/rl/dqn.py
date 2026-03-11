from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        num_actions: int,
        arch: str = "auto",
    ):
        super().__init__()
        c, h, w = obs_shape
        self.obs_shape = obs_shape
        self.num_actions = int(num_actions)

        arch = (arch or "auto").lower()
        if arch == "auto":
            arch = "cnn" if (h * w) >= 40 * 40 else "mlp"
        self.arch = arch

        if arch in ("mlp", "dueling_mlp"):
            in_features = c * h * w
            self._core = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            if arch == "mlp":
                self._head = nn.Linear(256, self.num_actions)
                self._dueling = False
            else:
                self._fc = nn.Identity()
                self._value = nn.Linear(256, 1)
                self._adv = nn.Linear(256, self.num_actions)
                self._dueling = True
        elif arch in ("cnn", "dueling_cnn"):
            self._core = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                core_out = self._core(dummy)
                core_dim = int(core_out.shape[1])

            if arch == "cnn":
                self._fc = nn.Sequential(nn.Linear(core_dim, 512), nn.ReLU())
                self._head = nn.Linear(512, self.num_actions)
                self._dueling = False
            else:
                self._fc = nn.Sequential(nn.Linear(core_dim, 512), nn.ReLU())
                self._value = nn.Linear(512, 1)
                self._adv = nn.Linear(512, self.num_actions)
                self._dueling = True
        else:
            raise ValueError(f"Unknown arch: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "mlp":
            z = self._core(x)
            return self._head(z)
        elif self.arch == "dueling_mlp":
            z = self._core(x)
            v = self._value(z)
            a = self._adv(z)
            return v + (a - a.mean(dim=1, keepdim=True))

        # CNN expects NCHW already (obs is stored as (C,H,W))
        z = self._core(x)
        z = self._fc(z)
        if not self._dueling:
            return self._head(z)
        v = self._value(z)
        a = self._adv(z)
        return v + (a - a.mean(dim=1, keepdim=True))


@dataclass
class ReplayBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device, n_step: int = 1, gamma: float = 0.99):
        self.capacity = int(capacity)
        self.device = device
        self.n_step = n_step
        self.gamma = gamma

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        self._size = 0
        self._idx = 0
        
        from collections import deque
        self.n_step_buffer = deque(maxlen=n_step)

    def __len__(self) -> int:
        return self._size
        
    def _compute_n_step_return(self) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Calculates the n-step return for the oldest experience in the n_step_buffer."""
        reward, next_obs, done = self.n_step_buffer[-1][-3:]
        
        # Calculate trailing rewards
        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, _, d = transition[-3:]
            reward = r + self.gamma * reward * (1.0 - d)
            if d:
                break
                
        # The true state is the one from n steps ago
        obs, action = self.n_step_buffer[0][:2]
        return obs, action, reward, next_obs, done

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        
        # Keep buffering until we hit n_steps, unless the episode ends early
        if len(self.n_step_buffer) < self.n_step and not done:
            return
            
        # At episode end, we need to flush out the rest of the buffer as truncated n-steps
        if done:
            while len(self.n_step_buffer) > 0:
                n_obs, n_action, n_reward, n_next_obs, n_done = self._compute_n_step_return()
                self._insert(n_obs, n_action, n_reward, n_next_obs, n_done)
                self.n_step_buffer.popleft()
        else:
            n_obs, n_action, n_reward, n_next_obs, n_done = self._compute_n_step_return()
            self._insert(n_obs, n_action, n_reward, n_next_obs, n_done)

    def _insert(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        i = self._idx
        self._obs[i] = obs
        self._actions[i] = int(action)
        self._rewards[i] = float(reward)
        self._next_obs[i] = next_obs
        self._dones[i] = 1.0 if done else 0.0

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        idxs = rng.integers(0, self._size, size=int(batch_size), endpoint=False)

        obs = torch.from_numpy(self._obs[idxs]).to(self.device)
        actions = torch.from_numpy(self._actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self._rewards[idxs]).to(self.device)
        next_obs = torch.from_numpy(self._next_obs[idxs]).to(self.device)
        dones = torch.from_numpy(self._dones[idxs]).to(self.device)

        return ReplayBatch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones)


class SumTree:
    """A binary tree data structure where the parent's value is the sum of its children."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        # The tree has 2 * capacity - 1 nodes. 
        # The first capacity - 1 nodes are parent nodes containing sums.
        # The last capacity nodes are the leaves containing individual priorities.
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        
    def add(self, idx: int, priority: float) -> None:
        tree_idx = idx + self.capacity - 1
        self.update(tree_idx, priority)
        
    def update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
            
    def get_leaf(self, v: float) -> tuple[int, int, float]:
        """Traverse tree to find the leaf index based on a value v in [0, total_priority]."""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # If we reach bottom, end the search
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx
                
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, data_idx, self.tree[leaf_idx]

    @property
    def total_priority(self) -> float:
        return float(self.tree[0])


@dataclass
class PERBatch(ReplayBatch):
    weights: torch.Tensor
    indices: np.ndarray


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device, alpha: float = 0.6, n_step: int = 1, gamma: float = 0.99):
        super().__init__(capacity, obs_shape, device, n_step, gamma)
        self.alpha = float(alpha)
        # Ensure capacity is a power of 2 for the SumTree if needed, though SumTree can handle non-powers
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
    def _insert(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        idx = self._idx
        super()._insert(obs, action, reward, next_obs, done)
        self.tree.add(idx, self.max_priority ** self.alpha)
        
    def sample_per(self, batch_size: int, rng: np.random.Generator, beta: float = 0.4) -> PERBatch:
        batch_size = int(batch_size)
        idxs = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        total_p = max(float(self.tree.total_priority), 1e-10)
        segment = total_p / batch_size
        
        valid_leaves = self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + self._size]
        valid_leaves = valid_leaves[valid_leaves > 0]
        min_p = float(np.min(valid_leaves)) if len(valid_leaves) > 0 else 1.0
        min_prob = min_p / total_p
        
        max_weight = (min_prob * self._size) ** (-beta) if min_prob > 0 else 1.0
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            
            if np.isnan(a) or np.isnan(b) or a >= b:
                v = 0.0
            else:
                v = rng.uniform(a, b)
                
            tree_idx, data_idx, priority = self.tree.get_leaf(v)
            
            # Failsafe if numerical drift caused it to pick an empty leaf
            if data_idx >= self._size or priority <= 0.0:
                data_idx = int(rng.integers(0, max(1, self._size)))
                priority = float(self.tree.tree[data_idx + self.tree.capacity - 1])
                if priority <= 0.0:
                    priority = float(self.max_priority ** self.alpha)
                
            idxs[i] = data_idx
            
            # Importance sampling weight calculation
            prob = float(priority) / total_p
            weight = (prob * self._size) ** (-beta)
            weights[i] = weight / max_weight
            
        obs = torch.from_numpy(self._obs[idxs]).to(self.device)
        actions = torch.from_numpy(self._actions[idxs]).to(self.device)
        rewards = torch.from_numpy(self._rewards[idxs]).to(self.device)
        next_obs = torch.from_numpy(self._next_obs[idxs]).to(self.device)
        dones = torch.from_numpy(self._dones[idxs]).to(self.device)
        weights_t = torch.from_numpy(weights).to(self.device)
        
        return PERBatch(obs=obs, actions=actions, rewards=rewards, next_obs=next_obs, dones=dones, weights=weights_t, indices=idxs)
        
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            if np.isnan(err) or np.isinf(err):
                err = 10.0 # Clip exploded gradients from poisoning the tree
            priority = (abs(float(err)) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.add(int(idx), priority)



# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
_ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


@torch.no_grad()
def select_action(
    q_net: nn.Module,
    obs: np.ndarray,
    epsilon: float,
    num_actions: int,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    if rng.random() < float(epsilon):
        # action masking: enforcenon immediate death
        blocked = obs[0]  
        head_pos = np.argwhere(obs[1] == 1.0)  
        if len(head_pos) > 0:
            r, c = int(head_pos[0, 0]), int(head_pos[0, 1])
            safe = []
            for a in range(num_actions):
                dr, dc = _ACTION_DELTAS[a]
                nr, nc = r + dr, c + dc
                if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1] and not blocked[nr, nc]:
                    safe.append(a)
            if safe:
                return int(rng.choice(safe))
        return int(rng.integers(0, num_actions))

    x = torch.from_numpy(obs).unsqueeze(0).to(device)
    q = q_net(x)
    return int(torch.argmax(q, dim=1).item())


def dqn_loss(
    *,
    q_net: nn.Module,
    target_net: nn.Module,
    batch: ReplayBatch,
    gamma: float,
    double_dqn: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: # returns loss_tensor, td_errors_tensor
    q = q_net(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if double_dqn:
            next_actions = q_net(batch.next_obs).argmax(dim=1)
            q_next = target_net(batch.next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            q_next = target_net(batch.next_obs).max(dim=1).values
        target = batch.rewards + (1.0 - batch.dones) * float(gamma) * q_next

    td_errors = target - q
    
    # Check if using PER
    if isinstance(batch, PERBatch):
        elementwise_loss = nn.functional.smooth_l1_loss(q, target, reduction='none')
        loss = (elementwise_loss * batch.weights).mean()
    else:
        loss = nn.functional.smooth_l1_loss(q, target)

    return loss, td_errors.detach()
