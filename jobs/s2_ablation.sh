#!/bin/bash --login
#SBATCH --job-name=ffl_s2abl
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-02:00
#SBATCH --array=0-2
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_s2abl_%a_%j.out

# Priority 2 — Stage-2 ablation at 500 rounds.
#
# All three arms: blind-spot config (rate=0.25, epochs=6), train probe,
# reference loss, seed=0. Report in 50-round windows to compare arms that
# matched at 30 rounds but may diverge at 500.
#
# Array index:
#   0: full defence, α=0.5          (hpc_s2abl_full_s0)  — fresh run, NOT arm A's ckpt
#   1: skip Stage 2, α=0.5          (hpc_s2abl_skip_s0)
#   2: skip Stage 2, α=0.1          (hpc_s2abl_skip_a01_s0) — paper's strongest-effect regime

IDX=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

if [ "${IDX}" -eq 0 ]; then
    NAME=hpc_s2abl_full_s0
    SKIP_FLAG=""
    ALPHA_FLAG=""
    echo "Step2-Abl idx=0: full defence α=0.5  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
elif [ "${IDX}" -eq 1 ]; then
    NAME=hpc_s2abl_skip_s0
    SKIP_FLAG="--skip-stage2"
    ALPHA_FLAG=""
    echo "Step2-Abl idx=1: skip Stage 2 α=0.5  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
elif [ "${IDX}" -eq 2 ]; then
    NAME=hpc_s2abl_skip_a01_s0
    SKIP_FLAG="--skip-stage2"
    ALPHA_FLAG="--dirichlet-alpha 0.1"
    echo "Step2-Abl idx=2: skip Stage 2 α=0.1  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
fi

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 500 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name ${NAME} \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --checkpoint-every 25 \
  --resume \
  ${SKIP_FLAG} \
  ${ALPHA_FLAG}

echo "Done idx=${IDX} name=${NAME}: $(date)"
