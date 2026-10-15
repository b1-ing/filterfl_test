# HPC Job Submission — FilterFL Blind-Spot Study

All scripts use `gpuL` (L40S, 48 GB), 1 GPU, 4 CPUs.
Time limit: Step 0 → 30 min; Steps 1-3 → 36 hrs (500 rounds × ~30 s + margin).
All output goes to `~/scratch/ffl_<name>_%j.out`.
All CSV metrics go to `results/hpc_<name>.csv` in the project directory.

## Order of submission

### Step 0 — Reproduce known result first (required gate)

```bash
cd ~/filterfl_test/filterfl_test
sbatch jobs/step0_reproduce.sh
```

Wait for it to finish, then read the output:

```bash
tail ~/scratch/ffl_s0_repro_<JOBID>.out
```

**Gate criteria** (must all pass within noise before launching Steps 1–3):
- 12 malicious appearances across 10 rounds
- 0 flagged (malicious acceptance rate 100%)
- Mean Stage-3 malicious confidence ≈ 0.42 (all rounds except round 7)
- Peak ASR ≈ 45.7% (round 5)

If Step 0 fails, stop and investigate before submitting anything else.

---

### Steps 1–3 — Launch together after Step 0 passes (can all run in parallel)

Each array job spawns up to 3 tasks; each task = 1 GPU.
With the 4-GPU quota: submit one arm at a time if you want headroom, or all
three Step-1 arrays at once and let SLURM queue the overflow.

```bash
# Step 1 — persistence: 3 arms × 3 seeds = 9 array tasks
sbatch jobs/step1_arm_a.sh   # blind-spot + full defence
sbatch jobs/step1_arm_b.sh   # blind-spot + NO defence (control)
sbatch jobs/step1_arm_c.sh   # ordinary attack + full defence (control)

# Step 2 — alternative operating points: 3 tasks
sbatch jobs/step2_alts.sh

# Step 3 — attacker ratio: 2 tasks
sbatch jobs/step3_ratio.sh
```

Monitor with `squeue` (aliased to `squeue --me` on CSF3).

---

## Where to find results

| Run name | CSV | SLURM out |
|---|---|---|
| hpc_step0_repro | results/hpc_step0_repro.csv | ~/scratch/ffl_s0_repro_\<J\>.out |
| hpc_s1a_s{0,1,2} | results/hpc_s1a_s{0,1,2}.csv | ~/scratch/ffl_s1a_s{seed}_\<J\>.out |
| hpc_s1b_s{0,1,2} | results/hpc_s1b_s{0,1,2}.csv | ~/scratch/ffl_s1b_s{seed}_\<J\>.out |
| hpc_s1c_s{0,1,2} | results/hpc_s1c_s{0,1,2}.csv | ~/scratch/ffl_s1c_s{seed}_\<J\>.out |
| hpc_s2_r20e10 | results/hpc_s2_r20e10.csv | ~/scratch/ffl_s2_0_\<J\>.out |
| hpc_s2_r26e6 | results/hpc_s2_r26e6.csv | ~/scratch/ffl_s2_1_\<J\>.out |
| hpc_s2_r28e6 | results/hpc_s2_r28e6.csv | ~/scratch/ffl_s2_2_\<J\>.out |
| hpc_s3_mal20 | results/hpc_s3_mal20.csv | ~/scratch/ffl_s3_0_\<J\>.out |
| hpc_s3_mal40 | results/hpc_s3_mal40.csv | ~/scratch/ffl_s3_1_\<J\>.out |

## Mid-run progress

Every 50 rounds, run.py prints a window summary line:

```
--- W[1-50] mean_MA=xx.xx mean_ASR=xx.xx max_ASR=xx.xx final_ASR=xx.xx accept_rate=xx.x% mal_app=N ---
```

Windows appear at rounds 50, 100, 150, 200, 250, 300, 350, 400, 450, 500.
The round-100, round-250, and round-500 windows are the ones to watch for Arm A.

To tail a running job's output (replace JOBID):
```bash
tail -f ~/scratch/ffl_s1a_s0_<JOBID>.out
```

## Resuming a preempted job

Jobs use `--resume`; checkpoints are saved every 25 rounds to `results/<run-name>_ckpt.pt`.
To resume, just resubmit the same script — it picks up where it left off.
