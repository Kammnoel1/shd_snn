#!/bin/bash -x
#SBATCH --account=ebrains-0000010
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --job-name=shd_snn_sweep
#SBATCH --array=0-3
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# --- Environment setup ---
module purge
module load Stages/2026 GCC Python

cd /p/project1/ebrains-0000010/shd_snn
source .snn/bin/activate

BATCH_SIZES=(256 256 1024 1024) 
SEEDS=(0 42 0 42)

export SNN_TEST_RUN=0
export SNN_NUM_EPOCHS=150
export SNN_BATCH_SIZE=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}
export SNN_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: batch_size=$SNN_BATCH_SIZE, seed=$SNN_SEED"

srun python -m snn_shd.train