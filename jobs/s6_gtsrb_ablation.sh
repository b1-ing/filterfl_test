#!/bin/bash --login
#SBATCH --job-name=ffl_gtsrb_abl
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-04:00
#SBATCH --array=0-3
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_gtsrb_abl_%a_%j.out

# Stage-2 ablation on GTSRB (C=43) — added to test whether Stage 2 matters
# at higher class counts where the C-dependent loss-balance shift predicts a
# different effective loss landscape.
#
# Design mirrors the CIFAR-10 Stage-2 ablation (s2_ablation.sh) exactly,
# with ONLY the dataset changed (type=GTSRB, C=43). ResNet-8 accepts
# num_classes=43 through its existing parameter; GeneratorCIFAR outputs
# 32x32 which matches the resized GTSRB images. No architecture changes.
#
# Array index:
#   0: full defence, α=0.5           (hpc_s6_gtsrb_full_a05_s0)
#   1: skip Stage 2, α=0.5           (hpc_s6_gtsrb_skip_a05_s0)
#   2: full defence, α=0.1           (hpc_s6_gtsrb_full_a01_s0)
#   3: skip Stage 2, α=0.1           (hpc_s6_gtsrb_skip_a01_s0)
#
# Interpretation:
#   If full vs skip Stage 2 produces a LARGER ASR gap on GTSRB than on
#   CIFAR-10, the C-dependent loss-balance shift is mechanistically active:
#   more classes → naturalness term shrinks → Stage 2 adds proportionally
#   more value on higher-C datasets. If the gap is similar or smaller, the
#   loss-balance shift does not drive the ablation result.
#
# Note: GTSRB download (~300 MB) happens on first run to ./data/. If the
# cluster's compute nodes have no outbound internet, download it first on
# a login node with:
#   ~/.conda/envs/filterfl/bin/python3 -c \
#     "from torchvision import datasets; datasets.GTSRB('./data', download=True)"

IDX=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

if [ "${IDX}" -eq 0 ]; then
    NAME=hpc_s6_gtsrb_full_a05_s0
    SKIP_FLAG=""
    ALPHA_FLAG=""
    echo "GTSRB-Abl idx=0: full defence α=0.5  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
elif [ "${IDX}" -eq 1 ]; then
    NAME=hpc_s6_gtsrb_skip_a05_s0
    SKIP_FLAG="--skip-stage2"
    ALPHA_FLAG=""
    echo "GTSRB-Abl idx=1: skip Stage 2 α=0.5  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
elif [ "${IDX}" -eq 2 ]; then
    NAME=hpc_s6_gtsrb_full_a01_s0
    SKIP_FLAG=""
    ALPHA_FLAG="--dirichlet-alpha 0.1"
    echo "GTSRB-Abl idx=2: full defence α=0.1  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
elif [ "${IDX}" -eq 3 ]; then
    NAME=hpc_s6_gtsrb_skip_a01_s0
    SKIP_FLAG="--skip-stage2"
    ALPHA_FLAG="--dirichlet-alpha 0.1"
    echo "GTSRB-Abl idx=3: skip Stage 2 α=0.1  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
fi

python -m filterfl.run \
  --config configs/gtsrb_primary.yaml \
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
echo "Results: results/${NAME}.csv"
