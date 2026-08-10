#!/bin/bash -x
#SBATCH --account=ebrains-0000010
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=ff_snn
#SBATCH --array=0-9
#SBATCH --output=logs/snn_%A_%a.out
#SBATCH --error=logs/snn_%A_%a.err

# Feed-forward SNN, one hidden layer.

# --- Environment setup ---
module purge
module load Stages/2026 GCC Python

cd /p/project1/ebrains-0000010/shd_snn
source .snn/bin/activate


# --- Run parameters ---
export SNN_TEST_RUN=0
export SNN_NUM_EPOCHS=150
export SNN_SEED=$SLURM_ARRAY_TASK_ID

# --- Hyperparameters ---
export SNN_ARCH=feedforward
export SNN_MODEL_NAME=feedforwardSNN
export SNN_IN_NEURONS=700
export SNN_HIDDEN_UNITS=128
export SNN_TAU_SYN_MS=10
export SNN_TAU_MEM_MS=20
export SNN_TRANSFORM=0

echo "Task $SLURM_ARRAY_TASK_ID: arch=$SNN_ARCH seed=$SNN_SEED"

srun python -m snn_shd.train
