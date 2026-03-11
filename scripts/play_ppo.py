from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as: `python scripts/play_ppo.py` without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tron_ai.envs import TronEnv
from tron_ai.rl.ppo import ActorCritic, select_action


def load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="checkpoints/ppo.pt")
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--opponent", type=str, default="random", choices=["random", "space_greedy"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt = load_checkpoint(args.model, device)
    
    # Auto-infer the grid size from the model's saved observation shape to prevent crashes
    obs_shape = tuple(ckpt["obs_shape"])
    checkpoint_grid_size = obs_shape[1]

    env = TronEnv(grid_size=checkpoint_grid_size, max_steps=args.max_steps, opponent=args.opponent, render_mode="human")
    env.metadata["render_fps"] = args.fps

    obs, _ = env.reset(seed=args.seed)

    arch = ckpt.get("arch", "auto")
    actor_critic = ActorCritic(obs_shape=tuple(ckpt["obs_shape"]), num_actions=int(ckpt["num_actions"]), arch=arch)
    actor_critic.load_state_dict(ckpt["model"])
    actor_critic.to(device)
    actor_critic.eval()

    ep = 0
    while True:
        action, _, _, _ = select_action(
            actor_critic=actor_critic,
            obs=obs,
            num_actions=env.action_space.n,
            device=device,
            deterministic=True,
        )
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            ep += 1
            print(
                f"episode={ep} steps={info.get('steps')} reward={reward} "
                f"agent_crash={info.get('agent_crash')} opp_crash={info.get('opp_crash')}"
            )
            obs, _ = env.reset(seed=int(np.random.randint(0, 2**31 - 1)))


if __name__ == "__main__":
    main()
