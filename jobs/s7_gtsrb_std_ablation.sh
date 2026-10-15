#!/bin/bash --login
#SBATCH --job-name=ffl_gtsrb_std
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-18:00
#SBATCH --array=0-3
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_gtsrb_std_%a_%j.out

# Stage-2 ablation on GTSRB using the STANDARD attacker (poison_rate=0.5),
# matching the paper's Table IV setup (CIFAR primary config attacker).
#
# Motivation: s6 runs (hpc_s6_gtsrb_*) used the blind-spot config
# (poison_rate=0.25, poison_epochs=6), which is designed to evade Stage 3's
# pooling. That makes the Stage-2 ablation question unanswerable from s6 alone
# — if the blind spot evades on GTSRB, full defence can't suppress it regardless
# of Stage 2. These runs use the paper's ordinary BadNets-style attacker so the
# defence should reproduce near-zero ASR on the full arm (Table IV: 0.14% at
# α=0.5, 0.48% at α=0.1), making the full-vs-skip gap meaningful.
#
# Target: full arm ASR near paper's Table IV values. If it does, Stage-2 gap
# is interpretable. If it doesn't, reproducing the full defence on GTSRB is
# itself the finding.
#
# Array index:
#   0: full defence, α=0.5           (hpc_s7_gtsrb_std_full_a05_s0)
#   1: skip Stage 2, α=0.5           (hpc_s7_gtsrb_std_skip_a05_s0)
#   2: full defence, α=0.1           (hpc_s7_gtsrb_std_full_a01_s0)
#   3: skip Stage 2, α=0.1           (hpc_s7_gtsrb_std_skip_a01_s0)
#
# Timing estimate: ~25 s/round × 2000 rounds ≈ 14 h. 18-hour limit gives margin.
# Uses --resume; checkpoints every 25 rounds. If preempted, resubmit as-is.

IDX=${SLURM_ARRAY_TASK_ID}

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

if [ "${IDX}" -eq 0 ]; then
    NAME=hpc_s7_gtsrb_std_full_a05_s0
    SKIP_FLAG=""
    ALPHA_FLAG=""
    echo "GTSRB-Std idx=0: full defence α=0.5  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
elif [ "${IDX}" -eq 1 ]; then
    NAME=hpc_s7_gtsrb_std_skip_a05_s0
    SKIP_FLAG="--skip-stage2"
    ALPHA_FLAG=""
    echo "GTSRB-Std idx=1: skip Stage 2 α=0.5  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
elif [ "${IDX}" -eq 2 ]; then
    NAME=hpc_s7_gtsrb_std_full_a01_s0
    SKIP_FLAG=""
    ALPHA_FLAG="--dirichlet-alpha 0.1"
    echo "GTSRB-Std idx=2: full defence α=0.1  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
elif [ "${IDX}" -eq 3 ]; then
    NAME=hpc_s7_gtsrb_std_skip_a01_s0
    SKIP_FLAG="--skip-stage2"
    ALPHA_FLAG="--dirichlet-alpha 0.1"
    echo "GTSRB-Std idx=3: skip Stage 2 α=0.1  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"
fi

python -m filterfl.run \
  --config configs/gtsrb_primary.yaml \
  --defense filterfl \
  --epochs 2000 \
  --poison-rate 0.5 \
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
