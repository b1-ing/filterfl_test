#!/bin/bash --login
#SBATCH --job-name=ffl_s1a
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-12:00
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_s1a_s%a_%j.out

# Step 1, Arm A — blind-spot config (rate=0.25, epochs=6), FULL defence, 500 rounds.
# Seeds 0, 1, 2 via SLURM array (SLURM_ARRAY_TASK_ID = seed).
# Submit ONLY after Step 0 confirms reproduction within tolerance.

SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "Step1-ArmA seed=${SEED}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 500 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed ${SEED} \
  --run-name hpc_s1a_s${SEED} \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 25 \
  --resume

echo "Done seed=${SEED}: $(date)"
