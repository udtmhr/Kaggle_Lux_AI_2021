# Strategic RL endgame arms

Each arm writes to a new Hydra run directory. Do not merge artifacts from control,
behavior-KL, categorical critic, or curriculum runs.

## Stage order

1. Control: `survival_strategic_strength_v5_lr5e7_kl001_2gpu`.
2. Behavior KL: `survival_strategic_strength_v6_behavior_kl_2gpu`.
3. Categorical critic: `survival_strategic_strength_v7_categorical_2gpu`, initialized from the promoted KL arm.
4. State curriculum: `survival_strategic_strength_v8_curriculum_2gpu`, initialized from the promoted categorical arm.

Example behavior-KL launch from the evaluated v4 100k bundle:

```bash
ulimit -n 8192
UV_CACHE_DIR=/tmp/lux-fork-uv-cache uv run --locked python run_monobeast.py \
  --config-name survival_strategic_strength_v6_behavior_kl_2gpu \
  +load_dir=/home/ueda/workspace/Kaggle_Lux_AI_2021_Fork/outputs/strength_v4_stable_2gpu_004/step_0100000 \
  +checkpoint_file=100096_weights.pt
```

For categorical and curriculum arms, replace `load_dir` and `checkpoint_file`
with the previous stage's promoted weights bundle. They intentionally use a fresh
optimizer. A full checkpoint from the same arm may instead be resumed with
`weights_only=false`; adaptive KL state, value metadata, optimizer, scheduler, and
step are restored.

## Required gates

- Evaluate 25k/50k/100k only from turn 0 with matched seed, map, and orientation keys.
- Internal records include candidate digest and RNG ID. `backend_profile.json`
  includes resolver change and friendly-collision candidate rates by map and turn band.
- Categorical checkpoints set `evaluation_eligible=false` when the maximum observed
  target-outside-support fraction exceeds 0.1%; `evaluate_checkpoint` refuses them.
- Use `analysis.ipynb` to compare score, survival, behavior KL, normalized entropy,
  importance ratios, value accuracy, and gradient diagnostics using identical keys.
- Promotion still requires the paired bootstrap and extinction/survival gates; a
  finite loss or completed run is not promotion evidence.
