from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as: `python scripts/play_dqn.py` without `pip install -e .`
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

    env = TronEnv(grid_size=args.grid, max_steps=args.max_steps, opponent=args.opponent, render_mode="human")
    env.metadata["render_fps"] = args.fps

    obs, _ = env.reset(seed=args.seed)

    arch = ckpt.get("arch", "auto")
    q_net = QNetwork(obs_shape=tuple(ckpt["obs_shape"]), num_actions=int(ckpt["num_actions"]), arch=arch)
    q_net.load_state_dict(ckpt["model"])
    q_net.to(device)
    q_net.eval()

    ep = 0
    while True:
        action = act_greedy(q_net, obs, device)
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
