#!/bin/bash --login
#SBATCH --job-name=ffl_topk_warmup
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-06:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_topk_warmup_%j.out

# Convergence-effect test for top-k fix early leakage.
#
# Hypothesis: the 83 accepted malicious updates in hpc_s4_topk16_s0 are
# concentrated in early rounds because the GAN triggers are poor before
# the global model has converged — not because top-16 pooling is
# fundamentally unable to catch this attacker.
#
# Design:
#   Phase 1 — 200 rounds of clean FedAvg (no attack, no defense).
#              Saves results/clean_ref_e200_ckpt.pt.
#   Phase 2 — arm A config (rate=0.25, poison_epochs=6, probe_mode=train,
#              stage2_loss=reference) + top-16 pooling, 700 rounds,
#              warm-started from the phase-1 clean checkpoint.
#              --warmup-from loads model weights + LR only; adversarial
#              namelist and RNG are freshly seeded so the attack starts
#              from round 1 of the attack phase on a pre-converged model.
#
# If early leakage disappears (rounds 1–20 acceptance drops sharply vs
# hpc_s4_topk16_s0), it is a convergence effect.
# If early leakage persists, something else is driving it.
#
# Result files:
#   results/hpc_s5_topk16_warmup_s0.csv   (attack-phase metrics)

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

CLEAN_CKPT=results/clean_ref_e200_ckpt.pt

echo "=== Phase 1: clean warmup (200 rounds)  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m scripts.gen_clean_ref \
  --config configs/cifar_primary.yaml \
  --rounds 200 \
  --out "${CLEAN_CKPT}"

echo "=== Phase 1 done: $(date) ==="

if [ ! -f "${CLEAN_CKPT}" ]; then
    echo "ERROR: ${CLEAN_CKPT} not found — aborting"
    exit 1
fi

echo "=== Phase 2: top-16 attack from converged model  Start: $(date) ==="

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 700 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_s5_topk16_warmup_s0 \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --stage3-pool-topk 16 \
  --warmup-from "${CLEAN_CKPT}" \
  --checkpoint-every 25

echo "=== Phase 2 done: $(date) ==="
echo "Results: results/hpc_s5_topk16_warmup_s0.csv"
echo ""
echo "Compare early leakage (rounds 1-20) against hpc_s4_topk16_s0.csv:"
echo "  if acceptance drops:  convergence effect confirmed"
echo "  if acceptance persists: something else is driving early leakage"
