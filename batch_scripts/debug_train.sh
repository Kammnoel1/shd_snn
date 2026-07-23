#!/bin/bash -x
#SBATCH --account=ebrains-0000010
#SBATCH --partition=develgpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=shd_snn_debug
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

# --- Environment setup ---
module purge
module load Stages/2026 GCC Python

source /p/project1/ebrains-0000010/shd_snn/.snn/bin/activate

cd /p/project1/ebrains-0000010/shd_snn

echo "=== Running short training (TEST_RUN mode) ==="
srun python train.py