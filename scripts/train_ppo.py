from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as: `python scripts/train_ppo.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tron_ai.envs import TronEnv
from tron_ai.rl.ppo import ActorCritic, RolloutBuffer, select_action, ppo_update


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--total-steps", type=int, default=2_000_000)
    
    # PPO specific parameters
    p.add_argument("--num-steps", type=int, default=2048, help="Steps per rollout")
    p.add_argument("--batch-size", type=int, default=64, help="Minibatch size for updating")
    p.add_argument("--update-epochs", type=int, default=10, help="Epochs to update the policy")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    p.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coeffient")
    p.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient for exploration")
    p.add_argument("--vf-coef", type=float, default=0.5, help="Value Function Loss coefficient")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    p.add_argument("--opponent", type=str, default="random", choices=["random", "space_greedy"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="checkpoints/ppo.pt")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--arch",
        type=str,
        default="auto",
        choices=["auto", "mlp", "cnn"],
        help="ActorCritic network architecture",
    )
    p.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
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
    
    # Optional PyTorch reproducibility
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env = TronEnv(grid_size=args.grid, max_steps=args.max_steps, opponent=args.opponent)
    obs, _ = env.reset(seed=args.seed)

    actor_critic = ActorCritic(
        obs_shape=env.observation_space.shape, 
        num_actions=env.action_space.n, 
        arch=args.arch
    ).to(device)

    if args.load:
        print(f"Loading checkpoint from {args.load}")
        try:
            ckpt = torch.load(args.load, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(args.load, map_location=device, weights_only=False)
        actor_critic.load_state_dict(ckpt["model"])
        print("Checkpoint loaded successfully")

    opt = torch.optim.Adam(actor_critic.parameters(), lr=args.lr, eps=1e-5)
    
    # Adjust total steps to fit exact number of updates
    num_updates = args.total_steps // args.num_steps

    buffer = RolloutBuffer(capacity=args.num_steps, obs_shape=env.observation_space.shape, num_actions=env.action_space.n, device=device)

    global_step = 0
    ep_return = 0.0
    ep_len = 0
    episodes = 0

    print(f"Starting PPO training on {device}...")
    for update in range(1, num_updates + 1):
        
        # Linear lr annealing could be added here
        
        actor_critic.eval()
        for step in range(args.num_steps):
            global_step += 1

            action, log_prob, value, mask = select_action(
                actor_critic=actor_critic,
                obs=obs,
                num_actions=env.action_space.n,
                device=device,
                deterministic=False
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            buffer.add(obs, action, log_prob, float(reward), done, value, mask)

            ep_return += float(reward)
            ep_len += 1

            obs = next_obs

            if done:
                episodes += 1
                obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
                if episodes % 25 == 0:
                    print(
                        f"t={global_step} update={update}/{num_updates} episodes={episodes} "
                        f"ep_return={ep_return:.1f} ep_len={ep_len} "
                        f"agent_crash={info.get('agent_crash')} opp_crash={info.get('opp_crash')}"
                    )
                ep_return = 0.0
                ep_len = 0

        # Bootstrapping next value for GAE
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(device)
            _, next_value = actor_critic(x)
            next_value = float(next_value.item())

        advs, returns = buffer.compute_returns_and_advantages(
            last_value=next_value, gamma=args.gamma, gae_lambda=args.gae_lambda
        )

        actor_critic.train()
        pg_loss, v_loss, ent_loss = ppo_update(
            actor_critic=actor_critic,
            opt=opt,
            buffer=buffer,
            advs=advs,
            returns=returns,
            clip_coef=args.clip_coef,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            batch_size=args.batch_size,
            epochs=args.update_epochs,
            rng=rng,
        )
        buffer.clear()

        if update % 50 == 0:
            print(f"Update: {update} | pg_loss: {pg_loss:.4f} | v_loss: {v_loss:.4f} | ent: {ent_loss:.4f}")

        # Save checkpoint periodically
        if global_step % 100_000 < args.num_steps and update > 1:
            torch.save(
                {
                    "model": actor_critic.state_dict(),
                    "obs_shape": tuple(int(x) for x in env.observation_space.shape),
                    "num_actions": int(env.action_space.n),
                    "arch": getattr(actor_critic, "arch", None),
                    "args": vars(args),
                },
                args.save,
            )
            print(f"Saved checkpoint to {args.save} at step {global_step}")

    # Final save
    torch.save(
        {
            "model": actor_critic.state_dict(),
            "obs_shape": tuple(int(x) for x in env.observation_space.shape),
            "num_actions": int(env.action_space.n),
            "arch": getattr(actor_critic, "arch", None),
            "args": vars(args),
        },
        args.save,
    )
    print(f"Done. Saved final PPO model to {args.save}")


if __name__ == "__main__":
    main()
