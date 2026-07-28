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

cd /p/project1/ebrains-0000010/shd_snn
source .snn/bin/activate

echo "=== Running short training (TEST_RUN mode) ==="
srun python -m snn_shd.train