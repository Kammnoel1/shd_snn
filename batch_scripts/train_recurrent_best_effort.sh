#!/bin/bash -x
#SBATCH --account=ebrains-0000010
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=rsnn_be
#SBATCH --array=0-9
#SBATCH --output=logs/rsnn_best-effort_%A_%a.out
#SBATCH --error=logs/rsnn_best-effort_%A_%a.err

# Best-effort recurrent SNN.

# --- Environment setup ---
module purge
module load Stages/2026 GCC Python

cd /p/project1/ebrains-0000010/shd_snn
source .snn/bin/activate

# --- Run parameters ---
export SNN_TEST_RUN=0
export SNN_NUM_EPOCHS=150
export SNN_SEED=$SLURM_ARRAY_TASK_ID

# --- Job specific hyperparameters (read by snn_shd/config.py) ---
export SNN_ARCH=recurrent
export SNN_MODEL_NAME=recurrentSNN_best-effort
export SNN_IN_NEURONS=70
export SNN_HIDDEN_UNITS=1024
export SNN_TAU_SYN_MS=40
export SNN_TAU_MEM_MS=80
export SNN_TRANSFORM=1
export SNN_MERGE_FACTOR=10
export SNN_JITTER_SIGMA=20

echo "Task $SLURM_ARRAY_TASK_ID: arch=$SNN_ARCH seed=$SNN_SEED"

srun python -m snn_shd.train
