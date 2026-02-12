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

        if arch == "mlp":
            in_features = c * h * w
            self._core = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            self._head = nn.Linear(256, self.num_actions)
            self._dueling = False
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
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device):
        self.capacity = int(capacity)
        self.device = device

        self._obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions = np.zeros((capacity,), dtype=np.int64)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        self._size = 0
        self._idx = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
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
) -> torch.Tensor:
    q = q_net(batch.obs).gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if double_dqn:
            next_actions = q_net(batch.next_obs).argmax(dim=1)
            q_next = target_net(batch.next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        else:
            q_next = target_net(batch.next_obs).max(dim=1).values
        target = batch.rewards + (1.0 - batch.dones) * float(gamma) * q_next

    return nn.functional.smooth_l1_loss(q, target)
