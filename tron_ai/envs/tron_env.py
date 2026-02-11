from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .opponents import ACTION_DELTAS, OpponentPolicy, RandomOpponent, SpaceGreedyOpponent, add_pos, opposite_action


@dataclass
class TronConfig:
    grid_size: int = 100
    max_steps: int = 1000
    opponent: str = "random"  # random|space_greedy

    # Rewards
    step_alive_reward: float = 1.0
    crash_reward: float = -10.0
    win_reward: float = 10.0

    # Rule tweaks
    disallow_reverse: bool = True


class TronEnv(gym.Env):
    """A minimal Tron light-cycle environment.

    Observation: (C,H,W) float32 with channels:
      0: blocked cells (walls/trails/heads)
      1: agent head
      2: opponent head

    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

    Episode ends when agent or opponent collides with a blocked cell.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        grid_size: int = 100,
        max_steps: int = 1000,
        opponent: str = "random",
        opponent_policy: OpponentPolicy | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.cfg = TronConfig(grid_size=grid_size, max_steps=max_steps, opponent=opponent)
        self.render_mode = render_mode

        h = self.cfg.grid_size
        w = self.cfg.grid_size
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(3, h, w), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

        self._rng = np.random.default_rng(seed)
        self._opponent_policy: OpponentPolicy = opponent_policy or self._make_opponent_policy(self.cfg.opponent)

        self._blocked: np.ndarray | None = None
        self._agent_pos: tuple[int, int] | None = None
        self._opp_pos: tuple[int, int] | None = None
        self._agent_dir: int = 1
        self._opp_dir: int = 3
        self._steps: int = 0

        self._viewer = None

    def _make_opponent_policy(self, name: str) -> OpponentPolicy:
        name = (name or "random").lower()
        if name == "random":
            return RandomOpponent(avoid_instant_death=True)
        if name in ("space_greedy", "greedy", "space"):
            return SpaceGreedyOpponent()
        raise ValueError(f"Unknown opponent policy: {name}")

    def _spawn_positions(self) -> tuple[tuple[int, int], tuple[int, int]]:
        n = self.cfg.grid_size
        # Spawn in opposite-ish halves, away from borders
        agent = (n // 2, n // 4)
        opp = (n // 2, (3 * n) // 4)
        if agent == opp:
            opp = (n // 2, n // 4 + 2)
        return agent, opp

    def _obs(self) -> np.ndarray:
        assert self._blocked is not None
        assert self._agent_pos is not None
        assert self._opp_pos is not None

        h, w = self._blocked.shape
        obs = np.zeros((3, h, w), dtype=np.float32)
        obs[0] = self._blocked.astype(np.float32)
        obs[1, self._agent_pos[0], self._agent_pos[1]] = 1.0
        obs[2, self._opp_pos[0], self._opp_pos[1]] = 1.0
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        _ = options

        n = self.cfg.grid_size
        self._blocked = np.zeros((n, n), dtype=bool)

        # Add outer walls as blocked.
        self._blocked[0, :] = True
        self._blocked[-1, :] = True
        self._blocked[:, 0] = True
        self._blocked[:, -1] = True

        self._agent_pos, self._opp_pos = self._spawn_positions()
        self._agent_dir = 1
        self._opp_dir = 3

        # Heads occupy their cells.
        self._blocked[self._agent_pos] = True
        self._blocked[self._opp_pos] = True

        self._steps = 0
        self._opponent_policy.reset(self._rng)

        info = {"agent_pos": self._agent_pos, "opp_pos": self._opp_pos}
        return self._obs(), info

    def get_state(self) -> dict:
        """Return a minimal snapshot of the current board for external renderers/UI."""
        return {
            "blocked": None if self._blocked is None else self._blocked.copy(),
            "agent_pos": self._agent_pos,
            "opp_pos": self._opp_pos,
            "agent_dir": self._agent_dir,
            "opp_dir": self._opp_dir,
            "steps": self._steps,
            "grid_size": self.cfg.grid_size,
        }

    def _apply_no_reverse(self, action: int, current_dir: int) -> int:
        if not self.cfg.disallow_reverse:
            return int(action)
        if int(action) == opposite_action(int(current_dir)):
            return int(current_dir)
        return int(action)

    def step(self, action: int):
        assert self._blocked is not None
        assert self._agent_pos is not None
        assert self._opp_pos is not None

        self._steps += 1

        action = self._apply_no_reverse(int(action), self._agent_dir)
        opp_action = self._opponent_policy.act(
            self._rng,
            self._blocked,
            self._opp_pos,
            self._opp_dir,
            self._agent_pos,
        )
        opp_action = self._apply_no_reverse(int(opp_action), self._opp_dir)

        agent_next = add_pos(self._agent_pos, ACTION_DELTAS[action])
        opp_next = add_pos(self._opp_pos, ACTION_DELTAS[opp_action])

        agent_crash = bool(self._blocked[agent_next])
        opp_crash = bool(self._blocked[opp_next])

        # Head-on same cell collision
        if agent_next == opp_next:
            agent_crash = True
            opp_crash = True

        terminated = False
        truncated = False
        reward = 0.0

        if agent_crash and opp_crash:
            terminated = True
            reward = 0.0
        elif agent_crash:
            terminated = True
            reward = float(self.cfg.crash_reward)
        elif opp_crash:
            terminated = True
            reward = float(self.cfg.win_reward)
        else:
            # Both survive: advance state and add trails.
            self._agent_pos = agent_next
            self._opp_pos = opp_next
            self._agent_dir = action
            self._opp_dir = opp_action

            self._blocked[self._agent_pos] = True
            self._blocked[self._opp_pos] = True

            reward = float(self.cfg.step_alive_reward)

        if (not terminated) and self._steps >= self.cfg.max_steps:
            truncated = True

        obs = self._obs()
        info = {
            "steps": self._steps,
            "agent_crash": agent_crash,
            "opp_crash": opp_crash,
            "agent_pos": self._agent_pos,
            "opp_pos": self._opp_pos,
            "agent_dir": self._agent_dir,
            "opp_dir": self._opp_dir,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if self._viewer is None:
            from ..rendering.pygame_viewer import PygameTronViewer

            # Auto-scale for large boards so the window is manageable.
            cell_size = 24
            if self.cfg.grid_size >= 60:
                cell_size = 8
            elif self.cfg.grid_size >= 40:
                cell_size = 12

            self._viewer = PygameTronViewer(
                grid_size=self.cfg.grid_size,
                cell_size=cell_size,
                fps=self.metadata.get("render_fps", 15),
            )

        assert self._blocked is not None
        assert self._agent_pos is not None
        assert self._opp_pos is not None

        return self._viewer.draw(
            blocked=self._blocked,
            agent_pos=self._agent_pos,
            opp_pos=self._opp_pos,
        )

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
