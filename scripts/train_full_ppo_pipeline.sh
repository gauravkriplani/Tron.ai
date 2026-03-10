#!/bin/bash

# Ensure we use the virtual environment
source venv/bin/activate

echo "Starting Phase 1: Training PPO against Random Opponent"
python scripts/train_ppo.py \
  --opponent random \
  --grid 20 \
  --total-steps 5000000 \
  --arch cnn \
  --save checkpoints/ppo_phase1.pt

# The && operator means phase 2 only starts if phase 1 finishes successfully
if [ $? -eq 0 ]; then
    echo "Phase 1 Complete. Starting Phase 2: Training PPO against Space Greedy"
    python scripts/train_ppo.py \
      --load checkpoints/ppo_phase1.pt \
      --opponent space_greedy \
      --grid 20 \
      --total-steps 5000000 \
      --arch cnn \
      --save checkpoints/ppo_phase2.pt
    
    echo "All training complete!"
else
    echo "Phase 1 encountered an error. Stopping."
fi
