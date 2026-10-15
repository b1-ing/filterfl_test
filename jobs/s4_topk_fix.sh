#!/bin/bash --login
#SBATCH --job-name=ffl_topk_fix
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-06:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_topk_fix_%j.out

# Constructive contribution: top-16 pooling as the one-line fix.
#
# Arm A config (rate=0.25, poison_epochs=6, probe_mode=train, stage2_loss=reference)
# run for 1000 rounds with --stage3-pool-topk 16 substituted for the default
# full-32 mean.
#
# Hypothesis: detection recovers — the UNDETECTED-POISONED blind spot closes —
# and the attack (ASR) collapses while clean accuracy is unaffected.
# Result file: results/hpc_s4_topk16_s0.csv
#
# The static diagnostic (results/separability_topk_curve.csv) showed that
# all 5 UNDETECTED-POISONED reps have A_16 > 0.5 while all 5 benign clients
# have A_16 < 0.06, providing the prior that motivates this run.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== top-16 fix run  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 1000 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_s4_topk16_s0 \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --stage3-pool-topk 16 \
  --checkpoint-every 25

echo "=== done: $(date) ==="
echo "Results: results/hpc_s4_topk16_s0.csv"
