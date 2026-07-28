#!/bin/bash -x
#SBATCH --account=ebrains-0000010
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=shd_snn
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# --- Environment setup ---
module purge
module load Stages/2026 GCC Python

cd /p/project1/ebrains-0000010/shd_snn
source .snn/bin/activate

export SNN_TEST_RUN=0
export SNN_NUM_EPOCHS=150

srun python -m snn_shd.train