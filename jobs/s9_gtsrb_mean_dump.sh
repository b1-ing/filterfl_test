#!/bin/bash --login
#SBATCH --job-name=ffl_gtsrb_meandump
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-12:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_gtsrb_meandump_%j.out

# s6 full-defence alpha=0.5 config (original mean pooling, blind-spot attacker,
# 500 epochs, seed 0) re-run with --dump-probe-responses, for the pooling
# taxonomy on the trajectory where the attack actually works (s8 was logged
# under the top-16 gate, whose clean trajectory understates how often mean
# pooling accepts). Only differences from s6 idx 0: run name and the dump flag.
# Result: results/hpc_s9_gtsrb_mean_a05_s0.csv + _probe.npz
# Uses --resume; checkpoints every 25 rounds. If timed out, resubmit as-is.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== GTSRB mean+dump  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m filterfl.run \
  --config configs/gtsrb_primary.yaml \
  --defense filterfl \
  --epochs 500 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_s9_gtsrb_mean_a05_s0 \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --dump-probe-responses \
  --checkpoint-every 25 \
  --resume

echo "=== done: $(date) ==="
