#!/bin/bash --login
#SBATCH --job-name=ffl_ext1k_b
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-02:30
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_ext1k_b_s%a_%j.out

# Extend arm B (NO defence, rate=0.25, epochs=6) to 1000 rounds, all seeds.
# All seeds already at 500 — each job runs epochs 501-1000 from checkpoint.
#
# Purpose: undefended ceiling for comparison with arm A at 1000 rounds.

SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "ExtendB-1000 seed=${SEED}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense none \
  --epochs 1000 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed ${SEED} \
  --run-name hpc_s1b_s${SEED} \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 25 \
  --resume

echo "Done seed=${SEED}: $(date)"
