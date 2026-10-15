#!/bin/bash --login
#SBATCH --job-name=ffl_noise_cal
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-01:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_noise_cal_%j.out

# Noise calibration for the late-round degenerate-pair invariance test.
#
# Question: does A vary from itself (same trigger, different GAN seed group)
# as much as A varies from B (trigger swap)? If yes, the A≠B finding at late
# rounds is explained by GAN-init noise on boundary clients, not by genuine
# trigger-swap sensitivity.
#
# Uses the round-498 snapshot from hpc_s1a_dense_s0 (same checkpoint, same
# diag-seed as the original late invariance test) so results are directly
# comparable to hpc_s1a_dense_invar_late.out.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

LATE_SNAP=results/hpc_s1a_dense_s0_snap_e00498_ckpt.pt

if [ ! -f "${LATE_SNAP}" ]; then
    echo "ERROR: ${LATE_SNAP} not found — run hpc_s1a_dense_s0 first"
    exit 1
fi

echo "=== Noise calibration  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m scripts.noise_calibration \
    --ckpt "${LATE_SNAP}" \
    --config configs/cifar_primary.yaml \
    --seed 0 \
    --poison-rate 0.25 \
    --stage3-probe-mode train \
    --stage2-loss reference \
    --rounds 5 \
    --diag-seed 9999 \
    --out results/hpc_noise_calibration.out

echo "=== Done: $(date) ==="
echo "Results: results/hpc_noise_calibration.out"
