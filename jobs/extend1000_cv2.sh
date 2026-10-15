#!/bin/bash --login
#SBATCH --job-name=ffl_ext1k_cv2
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-03:00
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_ext1k_cv2_s%a_%j.out

# Extend arm C v2 (full defence, rate=0.50, epochs=6, train probe, reference loss)
# to 1000 rounds, all seeds.
#
# Current progress: seed 0 ≈ 155 rounds, seed 1 ≈ 157 rounds, seed 2 ≈ 29 rounds.
# All have checkpoints — each job resumes from its checkpoint.
#
# Purpose: check whether the defence holds at 1000 rounds (rate=0.50 ordinary attack)
# or eventually degrades. Paired control for arm A (blind-spot, rate=0.25).

SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "ExtendCV2-1000 seed=${SEED}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 1000 \
  --poison-rate 0.50 \
  --poison-epochs 6 \
  --seed ${SEED} \
  --run-name hpc_s1c_v2_s${SEED} \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 25 \
  --resume

echo "Done seed=${SEED}: $(date)"
