#!/bin/bash --login
#SBATCH --job-name=ffl_cifar_cap15
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-06:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_cifar_cap15_%j.out

# Controlled class-0 skew test. IDENTICAL to jobs/s10_cifar_mean_dump.sh (the uncapped control: CIFAR-10 arm A,
# rate 0.25, 6 poison epochs, full defence, mean pooling, seed 0, 1000 rounds, --dump-probe-responses) except:
#   (1) run name hpc_s11_cifar_cap15_a_s0, (2) --cap-class-share 0.15.
# The cap swaps target-class (class-0) samples out of benign clients whose class-0 share exceeds 15% for
# non-target samples from low-skew benign clients (shard sizes unchanged, malicious clients untouched, RNG-free;
# see CODE_NOTES.md "Class-share cap intervention"). A CONSTRUCTED condition, not the paper's setup.
# Predictions: results/s11_cifar_capskew_PREDICTIONS.md (registered before the run).
# Result: results/hpc_s11_cifar_cap15_a_s0.csv + _probe.npz + _capreport.json
# Uses --resume; checkpoints every 25 rounds. If timed out, resubmit as-is.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== CIFAR-10 cap15  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 1000 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_s11_cifar_cap15_a_s0 \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --dump-probe-responses \
  --cap-class-share 0.15 \
  --checkpoint-every 25 \
  --resume

echo "=== done: $(date) ==="
