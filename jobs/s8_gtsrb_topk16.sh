#!/bin/bash --login
#SBATCH --job-name=ffl_gtsrb_topk
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-08:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_gtsrb_topk_%j.out

# Top-16 pooling fix on GTSRB (C=43), blind-spot attacker.
#
# Mirrors s6 idx 0 (hpc_s6_gtsrb_full_a05_s0: full defence, alpha=0.5,
# poison_rate=0.25, poison_epochs=6, probe_mode=train, stage2_loss=reference,
# 500 epochs) with ONLY --stage3-pool-topk 16 added. images_per_class=32 so
# k=16 is the same half-of-batch pooling as the CIFAR-10 s4 run.
#
# Baseline (s6 full a05, ~407 rounds before the 4h limit): per-round
# acceptance 82.6%, ASR(last 100) ~90%, MA(last 100) ~76%.
# If top-16 restores detection here the fix generalises beyond CIFAR-10.
# Result file: results/hpc_s8_gtsrb_topk16_a05_s0.csv
#
# --dump-probe-responses writes results/hpc_s8_gtsrb_topk16_a05_s0_probe.npz (one consolidated file:
# every client x probe image x class softmax + ground-truth malicious flag + live-rule outcome) for the
# offline pooling taxonomy.
#
# Uses --resume; checkpoints every 25 rounds. If preempted/timed out, resubmit as-is.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== GTSRB top-16  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m filterfl.run \
  --config configs/gtsrb_primary.yaml \
  --defense filterfl \
  --epochs 500 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_s8_gtsrb_topk16_a05_s0 \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --stage3-pool-topk 16 \
  --dump-probe-responses \
  --checkpoint-every 25 \
  --resume

echo "=== done: $(date) ==="
echo "Results: results/hpc_s8_gtsrb_topk16_a05_s0.csv"
