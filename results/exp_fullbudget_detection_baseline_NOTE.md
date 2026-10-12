# Full-budget (paper Table 2) re-run of the detection baseline

**Why:** our Stage 1/2 CGAN hyperparameters (`stage1_rounds=10`, `stage2_rounds=5`,
`images_per_class=32`) were copied directly from the reference repo's hardcoded
`gan.py` constants (`n_epochs1=10`, `n_epochs2=5`, `image_num=32` -- see
`filterfl/config.py` comments and `reference/CODE_NOTES.md`'s "Hyperparameters"
table), NOT from the paper. Per the paper's Table 2 (as read by the user),
the intended budget is stage1=12, stage2=15, images_per_class=1 -- a 3x longer
Stage 2 trigger-filtering budget. If 5 rounds is too short to narrow a generic
perturbation into the real planted trigger, every finding downstream of
Stage 1/2 (attractor mechanism, partition correlation, AUC=0.46) could be an
artifact of an under-trained Stage 2 rather than a property of the defense as
specified.

Setup: IDENTICAL to `exp1_detection_baseline_NOTE.md` -- same client (36),
same checkpoint, same malicious model (MA=48.07, ASR=87.87,
poison_epochs=6, poison_rate=0.5, unscaled alpha=1.0) -- ONLY the
three GAN hyperparameters above changed: stage1_rounds=12,
stage2_rounds=15, images_per_class=1.
rho=0.4, target_class=0.

## Results

| run | pred_class | confidence | class_match (==target) | would trip rho |
|---|---|---|---|---|
| 0 | 4 | 0.6403 | no | no |
| 1 | 4 | 0.6559 | no | no |
| 2 | 4 | 0.5379 | no | no |
| 3 | 4 | 0.5531 | no | no |
| 4 | 4 | 0.4277 | no | no |
| 5 | 4 | 0.4204 | no | no |
| 6 | 4 | 0.5636 | no | no |
| 7 | 4 | 0.7143 | no | no |
| 8 | 4 | 0.5999 | no | no |
| 9 | 4 | 0.4167 | no | no |

**pred_class distribution across 10 runs:** {4: 10}

**class_match (pred_class==target_class) rate: 0/10**
**would actually trip Stage 3 (class_match AND confidence>rho): 0/10**

## Comparison to the original (reference-budget) baseline

| budget | stage1_rounds | stage2_rounds | images_per_class | pred_class distribution | class_match rate |
|---|---|---|---|---|---|
| reference (original) | 10 | 5 | 32 | see `exp1_detection_baseline_NOTE.md` | 0/10 |
| paper Table 2 (this run) | 12 | 15 | 1 | {4: 10} | 0/10 |
