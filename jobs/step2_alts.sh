#!/bin/bash --login
#SBATCH --job-name=ffl_s2
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-12:00
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_s2_%a_%j.out

# Step 2 — Alternative operating points, full defence, 500 rounds, seed 0 each.
# Array index maps to config:
#   0: rate=0.20, epochs=10  (top grid cell by ASR/confidence margin)
#   1: rate=0.26, epochs=6   (top of verified evasion region)
#   2: rate=0.28, epochs=6   (just outside prior safe boundary — boundary probe)

CONFIG_IDX=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

if [ "${CONFIG_IDX}" -eq 0 ]; then
    RATE=0.20; PEPOCHS=10; NAME=hpc_s2_r20e10
elif [ "${CONFIG_IDX}" -eq 1 ]; then
    RATE=0.26; PEPOCHS=6;  NAME=hpc_s2_r26e6
elif [ "${CONFIG_IDX}" -eq 2 ]; then
    RATE=0.28; PEPOCHS=6;  NAME=hpc_s2_r28e6
fi

echo "Step2 idx=${CONFIG_IDX} rate=${RATE} pepochs=${PEPOCHS} name=${NAME}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 500 \
  --poison-rate ${RATE} \
  --poison-epochs ${PEPOCHS} \
  --seed 0 \
  --run-name ${NAME} \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 25 \
  --resume

echo "Done idx=${CONFIG_IDX}: $(date)"
