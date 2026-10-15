#!/bin/bash --login
#SBATCH --job-name=ffl_lrdeg
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-02:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_lrdeg_%j.out

# Late-round degenerate-pair invariance test.
#
# Replicates the round-1 d1_d2_role_test protocol at arm A's epoch-500
# checkpoint to test whether d1-invariance holds at late training.
#
# Protocol: load epoch-500 as G_old(500), run 5 fresh FedAvg rounds to get
# G_old(501..505), then for each consecutive pair test:
#   A) d1=d2=G_old(round)    vs   B) d1=d2=G_old(round+1)
# classifying the SAME clients under both triggers (3 GAN seeds each).
# If the pred-class vector is identical -> invariance holds at round 500.
# If it differs -> invariance is bounded to early training.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== late-round degenerate test  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m scripts.late_round_degenerate \
    --ckpt results/hpc_s1a_s0_ckpt.pt \
    --config configs/cifar_primary.yaml \
    --rounds 5 \
    --diag-seed 9999 \
    --out results/late_round_degenerate.out

echo "Done: $(date)"
