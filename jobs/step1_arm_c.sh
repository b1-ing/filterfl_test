#!/bin/bash --login
#SBATCH --job-name=ffl_s1c
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-12:00
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_s1c_s%a_%j.out

# Step 1, Arm C — ordinary attack (rate=0.50, epochs=6), FULL defence, 500 rounds.
# Paired control: shows FilterFL's behaviour against a non-stealthy attacker
# at the same scale, for direct comparison against Arm A.
# Seeds 0, 1, 2 via SLURM array.

SEED=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "Step1-ArmC seed=${SEED}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 500 \
  --poison-rate 0.50 \
  --poison-epochs 6 \
  --seed ${SEED} \
  --run-name hpc_s1c_s${SEED} \
  --checkpoint-every 25 \
  --resume

echo "Done seed=${SEED}: $(date)"
