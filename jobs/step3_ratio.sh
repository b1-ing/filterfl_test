#!/bin/bash --login
#SBATCH --job-name=ffl_s3
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 1-12:00
#SBATCH --array=0-1
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_s3_%a_%j.out

# Step 3 — Attacker ratio sweep, blind-spot config, full defence, 500 rounds, seed 0.
# Array index maps to malicious client count:
#   0: 20 malicious clients (20% of 100) — expect ~2 sampled per round
#   1: 40 malicious clients (40% of 100) — expect ~4 sampled per round
# More accepted contributions per round should compound persistence without
# changing per-client detectability.

CONFIG_IDX=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

if [ "${CONFIG_IDX}" -eq 0 ]; then
    MAL_N=20; NAME=hpc_s3_mal20
elif [ "${CONFIG_IDX}" -eq 1 ]; then
    MAL_N=40; NAME=hpc_s3_mal40
fi

echo "Step3 idx=${CONFIG_IDX} mal_clients=${MAL_N} name=${NAME}  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 500 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --poison-client-num ${MAL_N} \
  --run-name ${NAME} \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 25 \
  --resume

echo "Done idx=${CONFIG_IDX}: $(date)"
