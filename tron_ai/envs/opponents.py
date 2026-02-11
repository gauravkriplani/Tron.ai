from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
ACTION_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


def opposite_action(action: int) -> int:
    return (action + 2) % 4


def add_pos(pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    return (pos[0] + delta[0], pos[1] + delta[1])


@dataclass
class OpponentPolicy:
    """Interface-like base for opponent policies."""

    def reset(self, rng: np.random.Generator) -> None:
        _ = rng

    def act(
        self,
        rng: np.random.Generator,
        grid_blocked: np.ndarray,
        self_pos: tuple[int, int],
        self_dir: int,
        other_pos: tuple[int, int],
    ) -> int:
        raise NotImplementedError


@dataclass
class RandomOpponent(OpponentPolicy):
    avoid_instant_death: bool = True

    def act(
        self,
        rng: np.random.Generator,
        grid_blocked: np.ndarray,
        self_pos: tuple[int, int],
        self_dir: int,
        other_pos: tuple[int, int],
    ) -> int:
        _ = other_pos

        actions = [0, 1, 2, 3]
        rng.shuffle(actions)

        if not self.avoid_instant_death:
            a = int(actions[0])
            if a == opposite_action(self_dir):
                return self_dir
            return a

        for a in actions:
            if a == opposite_action(self_dir):
                continue
            nxt = add_pos(self_pos, ACTION_DELTAS[int(a)])
            if not grid_blocked[nxt]:
                return int(a)

        # No safe move found: allow reverse or just continue
        return int(self_dir)


def flood_fill_free_space(grid_blocked: np.ndarray, start: tuple[int, int]) -> int:
    """Count reachable free cells from start (4-neighborhood)."""
    if grid_blocked[start]:
        return 0

    h, w = grid_blocked.shape
    q = [start]
    seen = np.zeros((h, w), dtype=bool)
    seen[start] = True
    count = 0

    while q:
        r, c = q.pop()
        count += 1
        for dr, dc in ACTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (not seen[nr, nc]) and (not grid_blocked[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))

    return count


@dataclass
class SpaceGreedyOpponent(OpponentPolicy):
    """Heuristic opponent: pick safe move maximizing reachable free space."""

    def act(
        self,
        rng: np.random.Generator,
        grid_blocked: np.ndarray,
        self_pos: tuple[int, int],
        self_dir: int,
        other_pos: tuple[int, int],
    ) -> int:
        _ = other_pos

        candidates: list[tuple[int, int]] = []  # (score, action)
        for a in (0, 1, 2, 3):
            if a == opposite_action(self_dir):
                continue
            nxt = add_pos(self_pos, ACTION_DELTAS[a])
            if grid_blocked[nxt]:
                continue
            score = flood_fill_free_space(grid_blocked, nxt)
            candidates.append((score, a))

        if not candidates:
            return int(self_dir)

        best_score = max(s for s, _ in candidates)
        best_actions = [a for s, a in candidates if s == best_score]
        return int(rng.choice(best_actions))


@dataclass
class DQNOpponent(OpponentPolicy):
    """Opponent driven by a DQN checkpoint trained on the same observation format.

    Note: The DQN was trained from the "agent" perspective. For controlling the opponent,
    we build an observation where channel 1 is the opponent head and channel 2 is the agent head.
    """

    checkpoint_path: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        import torch

        self._torch = torch
        self._device = torch.device(self.device)

        # Robust checkpoint loading across torch versions
        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self._device, weights_only=True)
        except Exception:
            ckpt = torch.load(self.checkpoint_path, map_location=self._device, weights_only=False)

        obs_shape = tuple(int(x) for x in ckpt["obs_shape"])
        num_actions = int(ckpt["num_actions"])

        from tron_ai.rl.dqn import QNetwork

        self._net = QNetwork(obs_shape=obs_shape, num_actions=num_actions)
        self._net.load_state_dict(ckpt["model"])
        self._net.to(self._device)
        self._net.eval()

    def act(
        self,
        rng: np.random.Generator,
        grid_blocked: np.ndarray,
        self_pos: tuple[int, int],
        self_dir: int,
        other_pos: tuple[int, int],
    ) -> int:
        _ = rng
        _ = self_dir

        h, w = grid_blocked.shape
        obs = np.zeros((3, h, w), dtype=np.float32)
        obs[0] = grid_blocked.astype(np.float32)
        obs[1, self_pos[0], self_pos[1]] = 1.0
        obs[2, other_pos[0], other_pos[1]] = 1.0

        x = self._torch.from_numpy(obs).unsqueeze(0).to(self._device)
        q = self._net(x)
        return int(self._torch.argmax(q, dim=1).item())
