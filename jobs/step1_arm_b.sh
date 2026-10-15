#!/bin/bash --login
#SBATCH --job-name=ffl_s1b
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-12:00
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_s1b_s%a_%j.out

# Step 1, Arm B — blind-spot config (rate=0.25, epochs=6), NO defence, 500 rounds.
# Tells us whether persistence failure (if observed in Arm A) is about FilterFL at
# all, or about this attack config being intrinsically weak.
# Seeds 0, 1, 2 via SLURM array.

SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "Step1-ArmB seed=${SEED}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense none \
  --epochs 500 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed ${SEED} \
  --run-name hpc_s1b_s${SEED} \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 25 \
  --resume

echo "Done seed=${SEED}: $(date)"
