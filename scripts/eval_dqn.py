from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as: `python scripts/eval_dqn.py` without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tron_ai.envs import TronEnv
from tron_ai.rl.dqn import QNetwork


def load_checkpoint(path: str, device: torch.device):
    """Load a checkpoint across PyTorch versions.

    PyTorch 2.6+ defaults `weights_only=True`, which can reject older/some dicts.
    We try weights-only first, and if that fails, fall back to full unpickling.
    Only use the fallback for checkpoints you trust.
    """

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def act_greedy(q_net: QNetwork, obs: np.ndarray, device: torch.device) -> int:
    x = torch.from_numpy(obs).unsqueeze(0).to(device)
    q = q_net(x)
    return int(torch.argmax(q, dim=1).item())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="checkpoints/dqn.pt")
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
    obs, _ = env.reset(seed=args.seed)

    arch = ckpt.get("arch", "auto")
    q_net = QNetwork(obs_shape=tuple(ckpt["obs_shape"]), num_actions=int(ckpt["num_actions"]), arch=arch)
    q_net.load_state_dict(ckpt["model"])
    q_net.to(device)
    q_net.eval()

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
            a = act_greedy(q_net, obs, device)
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
