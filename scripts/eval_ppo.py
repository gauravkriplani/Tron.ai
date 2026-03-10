from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as: `python scripts/eval_ppo.py`
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
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--opponent", type=str, default="random", choices=["random", "space_greedy"])
    p.add_argument("--seed", type=int, default=0)
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

    env = TronEnv(grid_size=args.grid, max_steps=args.max_steps, opponent=args.opponent)
    
    arch = ckpt.get("arch", "auto")
    actor_critic = ActorCritic(
        obs_shape=tuple(ckpt["obs_shape"]), 
        num_actions=int(ckpt["num_actions"]), 
        arch=arch
    )
    actor_critic.load_state_dict(ckpt["model"])
    actor_critic.to(device)
    actor_critic.eval()

    wins = 0
    losses = 0
    draws = 0
    lengths: list[int] = []

    rng = np.random.default_rng(args.seed)

    for _ in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        last_info = {}
        while not done:
            # PPO evaluation acts deterministically
            a, _, _, _ = select_action(actor_critic, obs, env.action_space.n, device, deterministic=True)
            obs, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            last_info = info

        if last_info.get("agent_crash") and last_info.get("opp_crash"):
            draws += 1
        elif last_info.get("opp_crash"):
            wins += 1
        elif last_info.get("agent_crash"):
            losses += 1
        lengths.append(int(last_info.get("steps", 0)))

    n = args.episodes
    print(
        f"episodes={n} wins={wins} losses={losses} draws={draws} "
        f"win_rate={wins/n:.3f} avg_len={np.mean(lengths):.1f}"
    )


if __name__ == "__main__":
    main()
