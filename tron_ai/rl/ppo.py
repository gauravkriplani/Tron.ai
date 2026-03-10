from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
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
            self._actor_head = nn.Linear(256, self.num_actions)
            self._critic_head = nn.Linear(256, 1)
        elif arch == "cnn":
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

            self._shared_fc = nn.Sequential(self._layer_init(nn.Linear(core_dim, 512)), nn.ReLU())
            self._actor_head = self._layer_init(nn.Linear(512, self.num_actions), std=0.01)
            self._critic_head = self._layer_init(nn.Linear(512, 1), std=1.0)
        else:
            raise ValueError(f"Unknown arch for PPO: {arch}. PPO supports 'mlp' and 'cnn'.")

    def _layer_init(self, layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, value)"""
        # CNN expects NCHW. Scale inputs if it's a CNN to avoid blowing up activations.
        if self.arch == "cnn":
            # Observations can be 0 or 1, but scaling can't hurt if max is > 1.
            # DQN typically scales raw pixels by 255 if they are in 0-255.
            # In Tron it's mostly 0/1/2/3 so we can leave or cast to float properly.
            x = x.float()

        z = self._core(x)
        if hasattr(self, "_shared_fc"):
            z = self._shared_fc(z)
        logits = self._actor_head(z)
        value = self._critic_head(z).squeeze(-1)
        return logits, value


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advs: torch.Tensor
    values: torch.Tensor
    masks: torch.Tensor


class RolloutBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], num_actions: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.num_actions = num_actions

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.masks = np.zeros((capacity, num_actions), dtype=np.float32)

        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        mask: np.ndarray,
    ) -> None:
        if self._ptr >= self.capacity:
            raise RuntimeError("RolloutBuffer is full.")

        self.obs[self._ptr] = obs
        self.actions[self._ptr] = int(action)
        self.log_probs[self._ptr] = float(log_prob)
        self.rewards[self._ptr] = float(reward)
        self.dones[self._ptr] = 1.0 if done else 0.0
        self.values[self._ptr] = float(value)
        self.masks[self._ptr] = mask

        self._ptr += 1

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        advs = np.zeros_like(self.rewards)
        last_gae_lam = 0.0

        for t in reversed(range(self.capacity)):
            if t == self.capacity - 1:
                next_non_terminal = 1.0 - self.dones[t]  # not exactly correct if done on last step, handled loosely
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advs[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        returns = advs + self.values
        return advs, returns

    def get_batches(
        self, advs: np.ndarray, returns: np.ndarray, batch_size: int, rng: np.random.Generator
    ):
        indices = np.arange(self.capacity)
        rng.shuffle(indices)

        for start_idx in range(0, self.capacity, batch_size):
            batch_idxs = indices[start_idx : start_idx + batch_size]
            yield RolloutBatch(
                obs=torch.from_numpy(self.obs[batch_idxs]).to(self.device),
                actions=torch.from_numpy(self.actions[batch_idxs]).to(self.device),
                log_probs=torch.from_numpy(self.log_probs[batch_idxs]).to(self.device),
                returns=torch.from_numpy(returns[batch_idxs]).to(self.device),
                advs=torch.from_numpy(advs[batch_idxs]).to(self.device),
                values=torch.from_numpy(self.values[batch_idxs]).to(self.device),
                masks=torch.from_numpy(self.masks[batch_idxs]).to(self.device),
            )

    def clear(self):
        self._ptr = 0


# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
_ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def get_safe_actions(obs: np.ndarray, num_actions: int) -> list[int]:
    """Action masking heuristic: avoids immediate lethal collisions."""
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
            return safe
    return list(range(num_actions))


@torch.no_grad()
def select_action(
    actor_critic: nn.Module,
    obs: np.ndarray,
    num_actions: int,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[int, float, float, np.ndarray]:
    """Returns (action, log_prob, value, mask) for the given observation."""
    x = torch.from_numpy(obs).unsqueeze(0).to(device)
    logits, value = actor_critic(x)

    safe_actions = get_safe_actions(obs, num_actions)
    
    # Store the mask as a numpy array to pass to the buffer
    mask_np = np.full((num_actions,), float('-inf'), dtype=np.float32)
    mask_np[safe_actions] = 0.0
    
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(device)
    masked_logits = logits + mask_tensor
    dist = Categorical(logits=masked_logits)

    if deterministic:
        action = torch.argmax(masked_logits, dim=1)
    else:
        action = dist.sample()

    log_prob = dist.log_prob(action)

    return int(action.item()), float(log_prob.item()), float(value.item()), mask_np


def ppo_update(
    *,
    actor_critic: nn.Module,
    opt: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    advs: np.ndarray,
    returns: np.ndarray,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    batch_size: int,
    epochs: int,
    rng: np.random.Generator,
    max_grad_norm: float = 0.5,
) -> tuple[float, float, float]:
    
    # Normalize advantages at the rollout level
    advs_mean = advs.mean()
    advs_std = advs.std() + 1e-8
    norm_advs = (advs - advs_mean) / advs_std

    total_pg_loss = 0.0
    total_v_loss = 0.0
    total_ent_loss = 0.0
    num_updates = 0

    for _ in range(epochs):
        for batch in buffer.get_batches(norm_advs, returns, batch_size, rng):
            logits, new_values = actor_critic(batch.obs)
            
            # Re-apply the same masks that were used during rollout
            masked_logits = logits + batch.masks
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(batch.actions)
            entropy = dist.entropy().mean()

            logratio = new_log_probs - batch.log_probs
            ratio = logratio.exp()

            pg_loss1 = -batch.advs * ratio
            pg_loss2 = -batch.advs * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss clipping
            v_loss_unclipped = (new_values - batch.returns) ** 2
            v_clipped = batch.values + torch.clamp(
                new_values - batch.values,
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - batch.returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()

            loss = pg_loss - ent_coef * entropy + vf_coef * v_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
            opt.step()

            total_pg_loss += pg_loss.item()
            total_v_loss += v_loss.item()
            total_ent_loss += entropy.item()
            num_updates += 1

    return (
        total_pg_loss / num_updates,
        total_v_loss / num_updates,
        total_ent_loss / num_updates,
    )
