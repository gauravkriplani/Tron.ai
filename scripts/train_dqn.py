from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as: `python scripts/train_dqn.py` without `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tron_ai.envs import TronEnv
from tron_ai.rl.dqn import QNetwork, ReplayBuffer, dqn_loss, select_action


def linear_schedule(start: float, end: float, duration: int, t: int) -> float:
    if duration <= 0:
        return float(end)
    frac = min(max(t / duration, 0.0), 1.0)
    return float(start + frac * (end - start))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--start-learning", type=int, default=20_000)
    p.add_argument("--train-every", type=int, default=4)
    p.add_argument("--target-update", type=int, default=10_000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=1_000_000)
    p.add_argument("--opponent", type=str, default="random", choices=["random", "space_greedy"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="checkpoints/dqn.pt")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "mlp", "cnn", "dueling_cnn"],
        help="Q-network architecture; auto picks cnn for larger grids",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    rng = np.random.default_rng(args.seed)
    env = TronEnv(grid_size=args.grid, max_steps=args.max_steps, opponent=args.opponent)
    obs, _ = env.reset(seed=args.seed)

    q_net = QNetwork(obs_shape=env.observation_space.shape, num_actions=env.action_space.n, arch=args.arch).to(device)
    target_net = QNetwork(obs_shape=env.observation_space.shape, num_actions=env.action_space.n, arch=args.arch).to(device)
    target_net.load_state_dict(q_net.state_dict())

    opt = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    rb = ReplayBuffer(args.buffer_size, obs_shape=env.observation_space.shape, device=device)

    ep_return = 0.0
    ep_len = 0
    episodes = 0

    for t in range(1, args.total_steps + 1):
        epsilon = linear_schedule(args.epsilon_start, args.epsilon_end, args.epsilon_decay, t)
        action = select_action(
            q_net=q_net,
            obs=obs,
            epsilon=epsilon,
            num_actions=env.action_space.n,
            rng=rng,
            device=device,
        )

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        rb.add(obs, action, reward, next_obs, done)

        ep_return += float(reward)
        ep_len += 1

        obs = next_obs

        if done:
            episodes += 1
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            if episodes % 10 == 0:
                print(
                    f"t={t} episodes={episodes} ep_return={ep_return:.1f} ep_len={ep_len} "
                    f"eps={epsilon:.3f} agent_crash={info.get('agent_crash')} opp_crash={info.get('opp_crash')}"
                )
            ep_return = 0.0
            ep_len = 0

        if t >= args.start_learning and t % args.train_every == 0 and len(rb) >= args.batch_size:
            batch = rb.sample(args.batch_size, rng)
            loss = dqn_loss(q_net=q_net, target_net=target_net, batch=batch, gamma=args.gamma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            opt.step()

        if t % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        if t % 25_000 == 0:
            torch.save(
                {
                    "model": q_net.state_dict(),
                    "obs_shape": tuple(int(x) for x in env.observation_space.shape),
                    "num_actions": int(env.action_space.n),
                    "arch": getattr(q_net, "arch", None),
                    "args": vars(args),
                },
                args.save,
            )
            print(f"Saved checkpoint to {args.save}")

    torch.save(
        {
            "model": q_net.state_dict(),
            "obs_shape": tuple(int(x) for x in env.observation_space.shape),
            "num_actions": int(env.action_space.n),
            "arch": getattr(q_net, "arch", None),
            "args": vars(args),
        },
        args.save,
    )
    print(f"Done. Saved final model to {args.save}")


if __name__ == "__main__":
    main()
