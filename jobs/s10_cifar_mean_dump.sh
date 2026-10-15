#!/bin/bash --login
#SBATCH --job-name=ffl_cifar_meandump
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-14:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_cifar_meandump_%j.out

# CIFAR-10 arm A (blind-spot attacker, rate=0.25, epochs=6, FULL defence, original
# mean pooling), seed 0, extended to 1000 rounds like hpc_s1a_s0 / extend1000_a,
# re-run from scratch with --dump-probe-responses so the pooling-taxonomy and
# held-out-threshold analyses (scripts/pooling_taxonomy.py, pooling_heldout.py)
# can be repeated on CIFAR-10. Only differences from step1_arm_a: run name, 1000
# epochs, the dump flag. rho follows the config's rho_schedule (logged per record).
# Result: results/hpc_s10_cifar_mean_a_s0.csv + _probe.npz
# Uses --resume; checkpoints every 25 rounds. If timed out, resubmit as-is.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== CIFAR-10 arm A mean+dump  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 1000 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_s10_cifar_mean_a_s0 \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --dump-probe-responses \
  --checkpoint-every 25 \
  --resume

echo "=== done: $(date) ==="
