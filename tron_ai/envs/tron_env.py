from __future__ import annotations

from collections import deque
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
    territory_reward_weight: float = 0.5  # bonus per step based on Voronoi territory advantage

    # Rule tweaks
    disallow_reverse: bool = True


class TronEnv(gym.Env):
    """A minimal Tron light-cycle environment.

    Observation: (C,H,W) float32 with channels:
      0: blocked cells (walls/trails/heads)
      1: agent head
      2: opponent head
      3: agent's own trail
      4: normalized distance to nearest obstacle
      5: flood-fill reachable area from agent
      6: Voronoi territory (1=agent closer, 0=opponent closer, 0.5=tie)

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
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(7, h, w), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

        self._rng = np.random.default_rng(seed)
        self._opponent_policy: OpponentPolicy = opponent_policy or self._make_opponent_policy(self.cfg.opponent)

        self._blocked: np.ndarray | None = None
        self._agent_trail: np.ndarray | None = None
        self._opp_trail: np.ndarray | None = None
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

    def _distance_to_obstacle(self) -> np.ndarray:
        """BFS from all blocked cells; return normalized distance for each cell."""
        h, w = self._blocked.shape
        dist = np.full((h, w), h + w, dtype=np.float32)
        q: deque[tuple[int, int]] = deque()

        for r in range(h):
            for c in range(w):
                if self._blocked[r, c]:
                    dist[r, c] = 0.0
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))

        max_dist = float(max(h, w))
        return np.clip(dist / max_dist, 0.0, 1.0)

    def _flood_fill_from(self, pos: tuple[int, int]) -> np.ndarray:
        """BFS from pos neighbours; return binary mask of reachable cells."""
        h, w = self._blocked.shape
        visited = np.zeros((h, w), dtype=np.float32)
        q: deque[tuple[int, int]] = deque()

        pr, pc = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < h and 0 <= nc < w and not self._blocked[nr, nc]:
                visited[nr, nc] = 1.0
                q.append((nr, nc))

        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not self._blocked[nr, nc] and visited[nr, nc] == 0.0:
                    visited[nr, nc] = 1.0
                    q.append((nr, nc))

        return visited

    def _voronoi_territory(self) -> np.ndarray:
        """BFS from both heads simultaneously; agent-closer cells=1, opp-closer=0, tie=0.5."""
        h, w = self._blocked.shape
        territory = np.full((h, w), 0.5, dtype=np.float32)
        dist_agent = np.full((h, w), h * w, dtype=np.int32)
        dist_opp = np.full((h, w), h * w, dtype=np.int32)

        qa: deque[tuple[int, int]] = deque()
        qo: deque[tuple[int, int]] = deque()

        # Seed BFS from agent head neighbours
        ar, ac = self._agent_pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < h and 0 <= nc < w and not self._blocked[nr, nc]:
                dist_agent[nr, nc] = 1
                qa.append((nr, nc))

        # Seed BFS from opponent head neighbours
        opr, opc = self._opp_pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = opr + dr, opc + dc
            if 0 <= nr < h and 0 <= nc < w and not self._blocked[nr, nc]:
                dist_opp[nr, nc] = 1
                qo.append((nr, nc))

        # BFS agent
        while qa:
            r, c = qa.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not self._blocked[nr, nc] and dist_agent[nr, nc] > dist_agent[r, c] + 1:
                    dist_agent[nr, nc] = dist_agent[r, c] + 1
                    qa.append((nr, nc))

        # BFS opponent
        while qo:
            r, c = qo.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not self._blocked[nr, nc] and dist_opp[nr, nc] > dist_opp[r, c] + 1:
                    dist_opp[nr, nc] = dist_opp[r, c] + 1
                    qo.append((nr, nc))

        # Assign territory
        agent_closer = dist_agent < dist_opp
        opp_closer = dist_opp < dist_agent
        territory[agent_closer] = 1.0
        territory[opp_closer] = 0.0
        # Blocked cells stay at 0.5
        territory[self._blocked] = 0.5

        return territory

    def _obs(self) -> np.ndarray:
        assert self._blocked is not None
        assert self._agent_pos is not None
        assert self._opp_pos is not None

        h, w = self._blocked.shape
        obs = np.zeros((7, h, w), dtype=np.float32)
        obs[0] = self._blocked.astype(np.float32)
        obs[1, self._agent_pos[0], self._agent_pos[1]] = 1.0
        obs[2, self._opp_pos[0], self._opp_pos[1]] = 1.0
        obs[3] = self._agent_trail.astype(np.float32)
        obs[4] = self._distance_to_obstacle()
        obs[5] = self._flood_fill_from(self._agent_pos)
        obs[6] = self._cached_voronoi if self._cached_voronoi is not None else self._voronoi_territory()
        self._cached_voronoi = None  # clear after use
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        _ = options

        n = self.cfg.grid_size
        self._blocked = np.zeros((n, n), dtype=bool)
        self._agent_trail = np.zeros((n, n), dtype=bool)
        self._opp_trail = np.zeros((n, n), dtype=bool)

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
        self._agent_trail[self._agent_pos] = True
        self._opp_trail[self._opp_pos] = True

        self._steps = 0
        self._cached_voronoi = None
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
            self._agent_trail[self._agent_pos] = True
            self._opp_trail[self._opp_pos] = True

            reward = float(self.cfg.step_alive_reward)

            # Territory-based reward shaping: bonus for controlling more space
            voronoi = self._voronoi_territory()
            self._cached_voronoi = voronoi  # cache for _obs() reuse
            if self.cfg.territory_reward_weight > 0:
                agent_territory = float(np.sum(voronoi == 1.0))
                opp_territory = float(np.sum(voronoi == 0.0))
                total_free = agent_territory + opp_territory
                if total_free > 0:
                    territory_ratio = (agent_territory - opp_territory) / total_free
                    reward += self.cfg.territory_reward_weight * territory_ratio

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
