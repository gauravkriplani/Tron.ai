#!/bin/bash
#SBATCH -A cs175_class_gpu     ## Account to charge
#SBATCH --time=12:00:00        ## Maximum running time of program
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --partition=gpu        ## Partition name (using allocated guaranteed queue)
#SBATCH --mem=12GB             ## Allocated Memory
#SBATCH --cpus-per-task=8      ## Number of CPU cores
#SBATCH --gres=gpu:1           ## Request 1 GPU

# Load Anaconda and activate the conda environment created in Exercise 1
module load anaconda/2024.06
source activate rl

# Pre-reqs from the class syllabus
module load ffmpeg
export MUJOCO_GL=egl

echo "Starting Rainbow DQN pipeline on cluster node $(hostname)"

echo "Training Custom Rainbow DQN against Space Greedy Baseine"
srun python -u scripts/train_dqn.py \
  --opponent space_greedy \
  --grid 20 \
  --total-steps 5000000 \
  --arch dueling_cnn \
  --per \
  --n-step 3 \
  --save checkpoints/dqn_rainbow.pt

if [ $? -eq 0 ]; then
    echo "All training complete!"
else
    echo "Training encountered an error. Stopping."
fi
