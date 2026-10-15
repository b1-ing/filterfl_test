#!/bin/bash --login
#SBATCH --job-name=ffl_cifar_noatk
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 4
#SBATCH -t 0-06:00
#SBATCH -o /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/scratch/ffl_cifar_noatk_%j.out

# No-attacker control for the benign probe-response "spikes" (see CODE_NOTES.md). IDENTICAL to
# jobs/s11_cifar_cap15.sh (CIFAR-10, full defence, mean pooling, seed 0, 1000 rounds, class-0 share cap 0.15,
# --dump-probe-responses) except --no-attack: the adversarial namelist is drawn (so partition, cap and the
# client-sampling sequence match s11 exactly) and then discarded, so NO client ever trains a backdoor. The 10 slots that
# were malicious in s11 train benignly here. Question: do benign records still reach s11's spike levels
# (>0.208 = benign p99, >0.30) when no backdoor exists anywhere? Compare with scripts/spike_vs_asr.py.
# Result: results/hpc_s12_cifar_noattack_s0.csv + _probe.npz + _capreport.json
# Uses --resume; checkpoints every 25 rounds. If timed out, resubmit as-is.

module purge
module load miniforge/24.11.2
source activate filterfl

cd /mnt/iusers01/fse-ugpgt01/compsci01/x77599jl/filterfl_test/filterfl_test

echo "=== CIFAR-10 no-attack control  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date) ==="

python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 1000 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name hpc_s12_cifar_noattack_s0 \
  --stage3-probe-mode train \
  --stage2-loss reference \
  --dump-probe-responses \
  --cap-class-share 0.15 \
  --no-attack \
  --checkpoint-every 25 \
  --resume

echo "=== done: $(date) ==="
