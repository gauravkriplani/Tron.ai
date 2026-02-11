from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PygameTronViewer:
    grid_size: int
    cell_size: int = 24
    fps: int = 15

    def __post_init__(self) -> None:
        import pygame

        pygame.init()
        self._pygame = pygame

        w = self.grid_size * self.cell_size
        h = self.grid_size * self.cell_size
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("TronEnv")
        self._clock = pygame.time.Clock()

        self._colors = {
            "bg": (18, 18, 22),
            "grid": (28, 28, 34),
            "blocked": (60, 60, 70),
            "agent": (80, 220, 120),
            "opp": (240, 90, 90),
        }

    def draw(
        self,
        *,
        blocked: np.ndarray,
        agent_pos: tuple[int, int],
        opp_pos: tuple[int, int],
    ):
        pygame = self._pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit

        self._screen.fill(self._colors["bg"])

        n = self.grid_size
        cs = self.cell_size

        # Grid background
        for r in range(n):
            for c in range(n):
                rect = pygame.Rect(c * cs, r * cs, cs, cs)
                color = self._colors["blocked"] if blocked[r, c] else self._colors["grid"]
                pygame.draw.rect(self._screen, color, rect)

        # Heads on top
        ar, ac = agent_pos
        or_, oc = opp_pos
        pygame.draw.rect(self._screen, self._colors["agent"], pygame.Rect(ac * cs, ar * cs, cs, cs))
        pygame.draw.rect(self._screen, self._colors["opp"], pygame.Rect(oc * cs, or_ * cs, cs, cs))

        pygame.display.flip()
        self._clock.tick(self.fps)

        # rgb_array mode: return pixels
        arr = pygame.surfarray.array3d(self._screen)
        # array3d gives (W,H,3) -> convert to (H,W,3)
        return np.transpose(arr, (1, 0, 2)).copy()

    def close(self) -> None:
        pygame = self._pygame
        pygame.display.quit()
        pygame.quit()
