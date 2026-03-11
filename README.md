# Tron AI (Phase 1)

Minimal Tron grid environment (Gymnasium) + a simple PyTorch DQN trainer.

## Setup

From this folder:

1. Create/activate a virtualenv (recommended)

2. Install deps:

`pip install -r requirements.txt`

Optional (nice for imports):

`pip install -e .`

Note: The scripts are runnable as `python scripts/...` without `pip install -e .`,
but you still need the dependencies installed in whatever Python you use.

## Train (DQN & PPO)

`python scripts/train_dqn.py --grid 20 --max-steps 500 --total-steps 200000 --opponent random`

*Advanced DQN options (Double Q-Learning is on by default):*
- *`--arch dueling_cnn` or `--arch dueling_mlp`: Use Dueling Network Architecture.*
- *`--per`: Use Prioritized Experience Replay buffer instead of uniform sampling.*
- *`--n-step 3`: Use 3-step returns instead of 1-step returns.*

`python scripts/train_ppo.py --grid 20 --total-steps 5000000 --opponent random --arch cnn`

These write checkpoints to `checkpoints/dqn.pt` and `checkpoints/ppo.pt`.

## Evaluate

`python scripts/eval_dqn.py --model checkpoints/dqn.pt --episodes 200 --opponent random`
`python scripts/eval_ppo.py --model checkpoints/ppo.pt --episodes 200 --opponent random`

## Watch the agent (Pygame)

`python scripts/play_dqn.py --model checkpoints/dqn.pt --opponent random --fps 20`
`python scripts/play_ppo.py --model checkpoints/ppo.pt --opponent random --fps 20`

If you want a tougher baseline, use:

`--opponent space_greedy`

## Play vs the RL bot (Pygame)

You control the green cycle (arrow keys or WASD). Red cycle is the RL model.

`python scripts/human_vs_dqn.py --model checkpoints/dqn.pt --fps 20`
`python scripts/human_vs_ppo.py --model checkpoints/ppo.pt --fps 20`

Controls:

- Move: arrows or WASD
- Reset round: `R`
- Quit: `Esc` or window close
