#!/bin/bash --login
#SBATCH --job-name=ffl_dense_invar
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-03:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_dense_invar_%j.out

# Priority 1 — Dense-snapshot arm A rerun + invariance tests.
#
# Phase 1: Re-run arm A (rate=0.25, epochs=6, train probe, reference loss, α=0.5,
# seed=0) to 500 rounds with --snapshot-every 1, saving a distinct checkpoint at
# every round (~10 GB total: 500 × 20 MB). Run name hpc_s1a_dense_s0 — does NOT
# overwrite the existing hpc_s1a_s0 run.
#
# Phase 2: Early invariance test — load round-20 snapshot, simulate 5 fresh
# rounds (diag-seed 9999), report degenerate-pair comparison at rounds 20-25.
#
# Phase 3: Late invariance test — load round-498 snapshot, simulate 5 fresh
# rounds, report degenerate-pair comparison at rounds 498-503.
#
# Hypothesis: invariance (A==B across all seeds) holds early and breaks late.
# Flag immediately if late A vs B vectors DIFFER — that changes the writeup.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== Phase 1: dense-snapshot arm A  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 500 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_s1a_dense_s0 \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --snapshot-every 1 \
  --checkpoint-every 25

echo "=== Phase 1 done: $(date) ==="
echo ""

EARLY_SNAP=results/hpc_s1a_dense_s0_snap_e00020_ckpt.pt
LATE_SNAP=results/hpc_s1a_dense_s0_snap_e00498_ckpt.pt

if [ ! -f "${EARLY_SNAP}" ]; then
    echo "ERROR: early snapshot ${EARLY_SNAP} not found — aborting invariance tests"
    exit 1
fi
if [ ! -f "${LATE_SNAP}" ]; then
    echo "ERROR: late snapshot ${LATE_SNAP} not found — aborting invariance tests"
    exit 1
fi

echo "=== Phase 2: EARLY invariance test (from round 20)  Start: $(date) ==="
python -m scripts.late_round_degenerate \
    --ckpt "${EARLY_SNAP}" \
    --config configs/cifar_primary.yaml \
    --seed 0 \
    --poison-rate 0.25 \
    --stage3-probe-mode train \
    --stage2-loss reference \
    --rounds 5 \
    --diag-seed 9999 \
    --out results/hpc_s1a_dense_invar_early.out

echo "=== Phase 2 done: $(date) ==="
echo ""

echo "=== Phase 3: LATE invariance test (from round 498)  Start: $(date) ==="
python -m scripts.late_round_degenerate \
    --ckpt "${LATE_SNAP}" \
    --config configs/cifar_primary.yaml \
    --seed 0 \
    --poison-rate 0.25 \
    --stage3-probe-mode train \
    --stage2-loss reference \
    --rounds 5 \
    --diag-seed 9999 \
    --out results/hpc_s1a_dense_invar_late.out

echo "=== Phase 3 done: $(date) ==="
echo ""
echo "=== ALL DONE: $(date) ==="
echo "Check results/hpc_s1a_dense_invar_{early,late}.out for A vs B comparison."
echo "FLAG: if late A-vs-B pred-class vectors differ across seeds, invariance is bounded to early training."
