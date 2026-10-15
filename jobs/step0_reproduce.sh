#!/bin/bash --login
#SBATCH --job-name=ffl_s0_repro
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-0:30
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_s0_repro_%j.out

# Step 0 — Reproduce the known 10-round blind-spot result (seed 0).
# Expected: 12 malicious appearances, 0 flagged, mean conf ~0.42, peak ASR ~45.7%.
# Run FIRST, before submitting Steps 1-3. Confirm numbers match before proceeding.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 10 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_step0_repro \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 10

echo "Done: $(date)"
