#!/bin/bash --login
#SBATCH --job-name=ffl_refswap
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-04:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_refswap_%j.out

# Reference-swap diagnostic: tests whether contamination of G_old (the Stage-2
# GAN reference) is what degrades detection at late rounds in arm A.
#
# Step 1: generate a clean 50-round reference model (no poisoning, arm A config).
# Step 2: simulate 30 late-round detection events from the arm A epoch-500
#         checkpoint, comparing detection with contaminated vs clean reference.
#
# Interpret the output:
#   TP recovers under clean reference  -> contamination of G_old is the cause
#   TP stays low under clean reference -> client models themselves have changed

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== ref-swap diagnostic  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

CLEAN_REF=results/clean_ref_e50_ckpt.pt
CONT_CKPT=results/hpc_s1a_s0_ckpt.pt
OUT=results/ref_swap_diag.csv

echo "--- Step 1: generate clean 50-round reference ---"
python -m scripts.gen_clean_ref \
  --config configs/cifar_primary.yaml \
  --rounds 50 \
  --out ${CLEAN_REF}

echo "--- Step 2: reference-swap detection comparison ---"
python -m scripts.ref_swap_diagnostic \
  --config configs/cifar_primary.yaml \
  --contaminated-ckpt ${CONT_CKPT} \
  --clean-ckpt ${CLEAN_REF} \
  --trials 30 \
  --rho 0.5 \
  --out ${OUT}

echo "Done: $(date)"
