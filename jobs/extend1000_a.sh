#!/bin/bash --login
#SBATCH --job-name=ffl_ext1k_a
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-03:00
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_ext1k_a_s%a_%j.out

# Extend arm A (full defence, rate=0.25, epochs=6) to 1000 rounds, all seeds.
#
# Seed 0: resumes from results/hpc_s1a_s0_ckpt.pt (epoch 500) → epochs 501-1000.
# Seeds 1, 2: no prior checkpoint — run from scratch to 1000 in one job.
#
# Question: does acceptance saturate at 100%? Does ASR plateau after 500?
# At γ=0.998, LR at round 1000 is ~13% of initial — late flatness may be the
# schedule, not the defence. Report MA alongside to distinguish.

SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "ExtendA-1000 seed=${SEED}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 1000 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed ${SEED} \
  --run-name hpc_s1a_s${SEED} \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 25 \
  --resume

echo "Done seed=${SEED}: $(date)"
