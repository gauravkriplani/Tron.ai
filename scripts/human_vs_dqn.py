from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as: `python scripts/human_vs_dqn.py` without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tron_ai.envs import TronEnv
from tron_ai.envs.opponents import DQNOpponent
from tron_ai.rendering.pygame_viewer import PygameTronViewer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="checkpoints/dqn.pt")
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--cell", type=int, default=8, help="pixel size of each grid cell")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    args = p.parse_args()

    # Human controls the green "agent".
    dqn_opp = DQNOpponent(checkpoint_path=args.model, device=args.device)
    env = TronEnv(grid_size=args.grid, max_steps=args.max_steps, opponent_policy=dqn_opp, render_mode=None)

    viewer = PygameTronViewer(grid_size=args.grid, cell_size=args.cell, fps=args.fps)

    rng = np.random.default_rng(args.seed)
    obs, info = env.reset(seed=args.seed)

    last_action = 1  # start moving right
    episodes = 0

    # Pygame lives inside the viewer.
    pygame = viewer._pygame  # intentionally using viewer's pygame init

    while True:
        # Process input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise SystemExit
                if event.key == pygame.K_r:
                    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                    last_action = 1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            last_action = 0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            last_action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            last_action = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            last_action = 3

        obs, reward, terminated, truncated, info = env.step(last_action)

        state = env.get_state()
        viewer.draw(
            blocked=state["blocked"],
            agent_pos=state["agent_pos"],
            opp_pos=state["opp_pos"],
        )

        if terminated or truncated:
            episodes += 1
            if info.get("agent_crash") and info.get("opp_crash"):
                outcome = "DRAW"
            elif info.get("opp_crash"):
                outcome = "YOU WIN"
            elif info.get("agent_crash"):
                outcome = "YOU LOSE"
            else:
                outcome = "TIMEOUT"
            print(f"episode={episodes} outcome={outcome} steps={info.get('steps')}")

            obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            last_action = 1


if __name__ == "__main__":
    main()
