# burst_diffusion

A standalone training pipeline, inspired by DDPM/DDIM, that reconstructs clean
images from noisy low-SNR measurements (SEM / microscopy style). The twist vs.
DDPM: the noise "epsilon" is a **real noisy image** — one frame of a burst —
and the timestep corresponds to **how many burst frames are averaged**.

It shares this repository with the original DDIM code but imports none of it
(the U-Net and EMA helper are adapted copies; see module docstrings). The only
intra-repo import is `noising_pipeline`, the synthetic-data tool.

**Documentation:** [experiment report](../docs/burst_diffusion_report.md)
(what was run and what came out) ·
[user guide](../docs/burst_diffusion_guide.md) (step-by-step usage, config
reference, troubleshooting) ·
[method & derivations](../docs/burst_diffusion_method.md) (the full math) ·
[Q&A](../docs/burst_diffusion_qna.md) (design-review questions with measured
answers: why one-shot denoises, the DDPM symmetry, why iteration lagged,
self-rollout validity). This README is the condensed version of them all.

## Method

**Setup.** Each clean image `x_0` has `N` noisy realizations `y_1..y_N` with
`E[y_j | x_0] ≈ x_0` (unbiased noise; single-frame variance `σ²`). On real
equipment this is a burst: `N` fast low-frame acquisitions of the same area.
Synthetically, `noising_pipeline` generates the replicas.

**Schedule.** `t ∈ {1..T}` runs in the DDPM direction (`t = T` noisiest). The
number of frames averaged at level `t` is

    m(t) = T + 1 − t          (m(T) = 1, m(1) = T, m(0) = T + 1)

and the state is `x_t = mean(y_j, j ∈ S)` for a random subset `S`, `|S| = m(t)`,
so `Var(x_t) = σ² / m(t)`. The signal is never attenuated — this is a
cold-diffusion / variance-exploding style degradation, not DDPM's
`√ᾱ`-scaled mixture. Training needs `N ≥ T + 1`.

**Training** (DDPM Algorithm 1 analog). Sample a source, `t` (antithetic
pairing `t ↔ T+1−t`), a subset `S` of `m(t)` frames, and a target frame
`ε = y_k` with **k ∉ S** (a *fresh* frame). Minimize the plain MSE
`‖ε_θ(x_t, t) − ε‖²` in model space `[−1, 1]`.

*Why the target must be fresh:* if `k ∈ S`, exchangeability gives
`E[y_k | mean(S)] = mean(S)` — the MSE-optimal network is the **identity** and
learns nothing. With a fresh target, the optimum is
`E[y_fresh | x_t] = E[x_0 | x_t]`: the posterior-mean clean estimate at every
averaging level — a noise-level-conditioned generalization of Noise2Noise
(Lehtinen et al., 2018). `schedule.target_mode: included` exists only as a
documented-degenerate ablation and warns.

*Real-equipment property:* the clean image never enters training — only noisy
frames do. The identical code trains on real SEM bursts with no ground truth;
clean images are needed only for synthetic generation and evaluation.

**Sampling** (DDIM analog). Start from the measurement `x_T` (one frame). For
each schedule step `t → t_next` the network predicts a plausible fresh frame
`ε̂ = ε_θ(x_t, t)` and the cumulative-average update folds it in as if it were
`m(t_next) − m(t)` real acquisitions:

    x_{t_next} = ( m(t)·x_t + (m(t_next) − m(t))·ε̂ ) / m(t_next)

The unit step is `x + (ε̂ − x)/(m(t)+1)`. The update **composes exactly**
(chaining any decreasing path with a shared prediction equals the direct
jump), so accelerated schedules (`--steps K`) only approximate the *changing*
prediction, never the update. Two outputs:

- `average` — the final running average (spec-faithful; still carries
  `1/(T+1)` of the original frame's real noise),
- `prediction` — the last `ε̂` (theoretically the cleanest estimate).

Evaluation reports both.

## What to expect during training

- **The loss plateaus and that is correct.** The target is always a noisy
  frame, so the loss converges to ≈ the single-frame noise variance in model
  space (**4×** the `[0,1]`-space variance; ≈ 0.13 at a 15 dB single-frame
  PSNR) plus estimation error. Do not chase it to zero.
- The target-noise part of the floor is *t-independent*, so the per-level
  `train/loss_by_t/*` TensorBoard scalars isolate estimation quality.
- Real progress is `val/psnr_pred_t*`: the EMA model's prediction PSNR against
  the clean image on held-out sources. It should exceed the single-frame PSNR
  within a few thousand steps and keep climbing.

## Known caveats (measured, not hidden)

1. **Clip bias.** `noising_pipeline` clips to `[0,1]` after every noise
   application; clipped noise is not zero-mean, so the N-frame average
   converges to a slightly biased limit. Mitigation: `generate` pre-scales
   clean sources into `[margin, 1−margin]` (default 0.15) and **measures** the
   residual (`stats.json`: avg-of-N PSNR and bias; warnings out of band). The
   network's target converges to the *same* clipped-mean limit as averaging,
   so model-vs-averaging comparisons stay fair.
2. **avg-of-N is a reference, not a ceiling.** A learned prior can beat
   N-frame averaging (standard Noise2Noise result). `iter_prediction` beating
   `avg_of_n` from ONE frame is the success case, not an anomaly.
3. **Salt-and-pepper noise breaks the math**: `E[y | x_0] = (1−a)·x_0 +
   a·salt_ratio ≠ x_0`, so the MSE objective converges to a biased limit.
   Generation supports it but warns; use gaussian/poisson (SEM-physical:
   shot + read noise) for quantitative runs.
4. **Train/inference gap.** At inference the running average mixes one real
   frame with predictions (residual real-noise variance `σ²/m²` vs the
   `σ²/m` seen in training). Accepted for v1; the `prediction` output and
   accelerated schedules (less compounding) are the practical mitigations.
5. **Never resize or color-convert noisy frames.** Resampling averages pixels
   (a hidden partial denoise) and correlates the noise; RGB→gray averages
   channels (≈ +3.5 dB hidden denoise). All content-scale and color changes
   happen on the *clean* sources before noising; the loaders only crop.
6. **t-embedding range.** `t ∈ {1..T}` occupies a sliver of the sinusoidal
   embedding designed for `0..1000`. Kept raw in v1; rescaling `t·1000/T` is
   the first knob if per-level diagnostics show weak conditioning.

## Workflow

```powershell
# 1. Generate + verify a synthetic burst dataset (BBBC038 example)
python -m burst_diffusion generate --source-dir data/BBBC038v1 `
  --output-dir data/BBBC038-burst-p10 --num-sources 96 --replicas 16 `
  --noise-type poisson --peak 10 --margin 0.15 --max-side 512 --overwrite

# 2. Eyeball it: single frames grainy, averages converging to clean
python -m burst_diffusion preview --dataset data/BBBC038-burst-p10 `
  --source-index 0 --out runs/burst_diffusion/preview.png

# 3. Train (TensorBoard logs under <run_dir>/tb)
python -m burst_diffusion train --config configs/burst_diffusion/bbbc038_p10.yml
#    stop:   Ctrl+C, or create <run_dir>/stop     resume: add --resume

# 4. Denoise one measurement
python -m burst_diffusion sample --checkpoint runs/burst_diffusion/bbbc038_p10/ckpt_latest.pt `
  --dataset data/BBBC038-burst-p10 --source-index 90 --replica 0 `
  --out runs/burst_diffusion/bbbc038_p10/samples --trajectory

# 5. Compare against the baselines on the held-out split
python -m burst_diffusion evaluate --config configs/burst_diffusion/bbbc038_p10.yml `
  --checkpoint runs/burst_diffusion/bbbc038_p10/ckpt_latest.pt `
  --out runs/burst_diffusion/bbbc038_p10/eval
```

`evaluate` reports `single_frame`, `avg_of_n`, `one_shot` (single forward
pass), `iter_average`, and `iter_prediction` — mean PSNR/SSIM plus a captioned
comparison grid. Expected ordering: `single_frame < iter_average`, and
`one_shot ≲ iter_prediction` (iteration helping is the key readout).

## Dataset layout

`generate` writes (and the loaders read):

```
<dataset_dir>/
  _sources/00000.png       staged prepped clean images (margin applied)
  sources.json             provenance: staged file -> original path + settings
  burst/
    clean/00000.png        the pipeline's clean copies (= ground truth)
    noisy/00000_00000.png  {source:05d}_{replica:05d}
    manifest.jsonl         one row per noisy frame
  stats.json               measured noise statistics + warnings
```

`BurstCache` accepts either `<dataset_dir>` or `<dataset_dir>/burst`, groups
frames by `source_index`, drops under-replicated or undersized sources with a
warning (interrupted generation runs stay usable), splits train/val **by
source** with a seeded shuffle, and caches everything in RAM as uint8. There
is deliberately **no DataLoader** (see `docs/sem_dataset_migration.md` for the
worker-restart trap this sidesteps); a seeded `BatchFactory` assembles batches
with aligned random crops and full RNG-state checkpointing for exact resume.

Real-equipment data: write the same layout (a `manifest.jsonl` with
`source_index`, `replica_index`, `clean_path`, `noisy_path` per frame — a
nominal clean/reference image is required by the loader but never used in
training) and everything downstream works unchanged.

## Data sources

- **BBBC038v1** (Broad Bioimage Benchmark Collection, nuclei microscopy):
  downloaded/cached by `noising_pipeline`, license CC0.
- **MIIC** (Microscopic Images of Integrated Circuits, SEM), NTU research
  data repository, doi:10.21979/N9/WBLTFI, license **CC BY-NC 4.0**
  (non-commercial, attribution). Not auto-downloaded; point `generate
  --source-dir` at your extracted copy.

Generated datasets, checkpoints, and run directories are git-ignored; never
commit them.

## Dependencies

torch, numpy, Pillow, PyYAML, pydantic, typer, tensorboard — all already in
the repository's `pyproject.toml`. No scikit-image/scipy/matplotlib: PSNR and
Gaussian-window SSIM are implemented in `metrics.py`, grids with PIL.
