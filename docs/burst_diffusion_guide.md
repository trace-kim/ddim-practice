# Burst-Averaging Diffusion — User Guide

How to run the [`burst_diffusion/`](../burst_diffusion/) pipeline end to end:
generate a dataset, verify it, train, denoise, and evaluate. For the theory see
[`burst_diffusion_method.md`](burst_diffusion_method.md); for the results of
the reference runs see [`burst_diffusion_report.md`](burst_diffusion_report.md).

All commands run from the repository root. PowerShell syntax is shown; drop the
backticks-for-line-continuation (`` ` ``) on Linux.

## 0. Install and self-check

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests -k burst_diffusion -q     # ~30 s, CPU-only, should be all green
```

## 1. Generate a synthetic burst dataset

Each clean source image gets `--replicas` independent noisy frames (same noise
distribution, different draws) via the repo's `noising_pipeline`.

```powershell
python -m burst_diffusion generate --source-dir data/BBBC038v1 `
  --output-dir data/BBBC038-burst-p10 --num-sources 96 --replicas 16 `
  --noise-type poisson --peak 10 --margin 0.15 --max-side 512 --overwrite
```

Key options:

| Option | Meaning |
|---|---|
| `--source-dir` | any directory of images (searched recursively; png/jpg/bmp/tif) |
| `--num-sources` | how many clean images to use (seeded subset via `--select-seed`) |
| `--replicas` | frames per source, **must be ≥ T+1** for the training schedule you plan |
| `--noise-type` | `poisson`, `gaussian`, `salt_pepper` — repeat the flag to stack them in order |
| `--peak` | Poisson photon count at intensity 1.0 (lower = noisier; 10 ≈ 14–17 dB frames) |
| `--gaussian-std` / `--gaussian-mean` | Gaussian parameters per step |
| `--steps` | corruption multiplier (fused: std·√s, peak/s) |
| `--margin` | pre-scales clean pixels into `[margin, 1−margin]` so the noiser's [0,1] clipping stays rare (clipped noise is biased) |
| `--grayscale/--rgb` | color conversion happens **before** noising, deliberately |
| `--max-side` | downscales clean sources above this size — clean only, never noisy frames |
| `--noise-seed` | reproducible draws; same seed + sources ⇒ bit-identical dataset |

Output layout:

```
data/BBBC038-burst-p10/
  _sources/00000.png       staged prepped clean images (margin applied)
  sources.json             provenance: staged file -> original path + settings
  burst/
    clean/00000.png        ground truth ({source:05d})
    noisy/00000_00000.png  frames ({source:05d}_{replica:05d})
    manifest.jsonl         one row per frame
  stats.json               measured noise statistics + warnings
```

**Read `stats.json` before training.** Targets: median single-frame PSNR in
**10–18 dB** ("noisy enough" but learnable), avg-of-N PSNR ≥ 26 dB and
|bias| ≤ 0.01 (averaging really converges to clean — the method's premise).
Out-of-band values print warnings with the suggested fix (raise `--margin`,
change `--peak`/`--gaussian-std`, add replicas). A `salt_pepper` warning means
the MSE objective will converge to a biased limit — use it only knowingly.

### Choosing T and N

`schedule.num_steps` (T) needs `replicas ≥ T+1` (T frames averaged + 1 held-out
target). Averaging gain grows like 10·log₁₀T dB while generation time, RAM and
sampling cost grow linearly — T=15/N=16 is the sensible start; T=31/N=32 is a
config-only regeneration later.

## 2. Preview — the visual check

```powershell
python -m burst_diffusion preview --dataset data/BBBC038-burst-p10 `
  --source-index 0 --out runs/burst_diffusion/preview.png
```

Top row: clean + individual frames. Bottom row: running averages with PSNR
captions. Single frames should look clearly grainy; the average row should
visibly converge toward clean at +3 dB per doubling.

## 3. Train

```powershell
python -m burst_diffusion train --config configs/burst_diffusion/bbbc038_p10.yml
python -m tensorboard.main --logdir runs/burst_diffusion/bbbc038_p10/tb   # separate shell
```

Config reference (`configs/burst_diffusion/*.yml`; unknown keys are rejected):

| Key | Meaning / constraint |
|---|---|
| `data.dataset_dir` | dataset root (or its `burst/` subdir) |
| `data.image_size` | training crop size; must divide by 2^(levels−1); sources smaller than this are dropped |
| `data.channels` | 1 or 3; grayscale bursts should use 1 |
| `data.val_fraction`, `data.split_seed` | held-out split, **by source**, deterministic |
| `schedule.num_steps` | T; every kept source needs ≥ T+1 frames |
| `schedule.target_mode` | `fresh` (real training) — `included` is a documented-degenerate ablation and warns |
| `model.ch`, `ch_mult`, `num_res_blocks`, `attn_resolutions` | U-Net size; keep attention ≤ 16; `ch` must divide by `num_groups` (default min(32, ch)) |
| `training.batch_size` | 8 on this machine — see Troubleshooting |
| `training.max_steps`, `lr`, `grad_clip`, `ema`, `ema_rate`, `antithetic`, `seed` | optimization; defaults follow the legacy DDIM recipe |
| `training.log_every / val_every / checkpoint_every / val_images` | cadence |
| `training.device` | `auto` / `cpu` / `cuda` |
| `sampling.output_mode`, `num_sample_steps` | defaults for the sampler |

**Reading the logs — important:**

- `train/loss` **plateaus at ≈ 4·10^(−PSNR_single/10) and that is correct**
  (≈0.13 for 17 dB frames, ≈0.15 for 14 dB). The target is itself noisy; the
  loss floor is its variance. Do not chase it to zero.
- Real progress = `val/psnr_pred_t15` (and `_t08`): the EMA model's prediction
  PSNR vs clean on fixed validation crops. It should exceed the single-frame
  PSNR within a few thousand steps and keep climbing.
- `train/loss_by_t/*` isolates estimation quality per noise level (the floor
  part is level-independent).

**Stop / resume:**

- Stop: `Ctrl+C`, or create a file named `stop` in the run dir
  (`New-Item runs/burst_diffusion/bbbc038_p10/stop`). Either way a checkpoint
  is saved on the way out.
- Resume: `... train --config <same.yml> --resume` (or
  `--resume-from <path.pt>`). Resume is **exact** — model, EMA, optimizer, and
  all RNG streams are restored, so an interrupted run reproduces the
  uninterrupted one bit-for-bit.
- Checkpoints: `ckpt_latest.pt` (rolling) + `ckpt_<step>.pt` milestones;
  written atomically; loaded with `weights_only=True`.

## 4. Denoise a measurement

```powershell
# a frame from a dataset (prints PSNR vs clean when clean exists):
python -m burst_diffusion sample --checkpoint runs/burst_diffusion/bbbc038_p10/ckpt_latest.pt `
  --dataset data/BBBC038-burst-p10 --source-index 12 --replica 0 `
  --out runs/burst_diffusion/bbbc038_p10/samples --trajectory

# or any PNG/TIFF (center-cropped to the model resolution, never resized):
python -m burst_diffusion sample --checkpoint ... --input my_noisy_scan.png --out out/
```

Writes `*_input.png`, `*_average.png` (the running-average output),
`*_prediction.png` (the final prediction — usually the better one), and
optionally `*_trajectory.png`. `--steps K` runs the accelerated K-step
schedule; `--no-ema` uses the raw weights.

## 5. Evaluate against the baselines

```powershell
python -m burst_diffusion evaluate --config configs/burst_diffusion/bbbc038_p10.yml `
  --checkpoint runs/burst_diffusion/bbbc038_p10/ckpt_latest.pt `
  --out runs/burst_diffusion/bbbc038_p10/eval
```

Reports, per held-out source and on average (PSNR + SSIM, `results.json` + a
captioned `comparison_grid.png`):

| Method | What it is |
|---|---|
| `single_frame` | the raw measurement (input floor) |
| `avg_of_n` | classical N-frame averaging — the *reference*, not a ceiling; beating it from one frame is the point |
| `one_shot` | one forward pass at t=T |
| `iter_average` / `iter_prediction` | the two outputs of the full iterative sampler |

`--split train`, `--limit K`, `--steps K` (accelerated), and `--tile`
(full-image inference via overlapping 64px tiles, uniform blending) are
available.

## 6. Real equipment data (no clean images)

Training never uses clean images, so real bursts work directly:

1. Acquire **N ≥ T+1 fast frames of the same area** (e.g. low-frame SEM scans),
   repeated over enough distinct areas for generalization (tens of areas is a
   reasonable start). **Frames must be pixel-aligned** — correct drift before
   ingestion; averaging misaligned frames blurs and breaks the premise.
2. Lay the files out as `clean/<src>.png` + `noisy/<src>_<rep>.png` and write a
   `manifest.jsonl` with one row per frame:

   ```json
   {"source_index": 0, "replica_index": 0, "clean_path": "clean/00000.png", "noisy_path": "noisy/00000_00000.png"}
   ```

   The loader requires a `clean_path`, but training ignores it — use the
   N-frame average as a stand-in reference. Validation PSNR then reads "vs the
   N-frame average" (a meaningful proxy), and the `avg_of_n` eval row becomes
   trivially perfect — ignore it in that setting.
3. Point `data.dataset_dir` at the folder and train as usual. Verify noise
   level first: `preview` works on any conforming dataset.

## 7. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| `fatal: Memory allocation failure` then `CUDA error: unknown error` at the first backward | **Windows commit charge exhausted** (WDDM backs CUDA allocations with commit; VMs/Docker/browsers eat it). Not a VRAM issue. Halve `batch_size`, or close the VM/Docker. Check with `Get-CimInstance Win32_OperatingSystem` (FreeVirtualMemory). |
| `source X has k frames but num_steps=T needs at least T+1` | lower `schedule.num_steps` or regenerate with more `--replicas` |
| `dropped N source(s) with fewer than K replicas` warning | an interrupted generation run — usable, those sources are skipped |
| `crops need at least SxS` | source images smaller than `image_size` — lower it or use larger sources |
| `converting RGB frames to grayscale ... partially denoises` warning | the burst was generated in RGB but `channels: 1` — regenerate with `--grayscale` |
| training loss flat around 0.10–0.15 | **expected** — that is the noise floor; watch `val/psnr_pred_*` instead |
| `checkpoint ... different config; resuming with the CURRENT config` warning | you changed the YAML between runs (e.g. `max_steps`) — usually intentional |
| stale `stop` file ends runs instantly | the trainer deletes it at startup; if launching by other means, delete `run_dir/stop` |

Artifacts live under `runs/burst_diffusion/<name>/` and datasets under
`data/` — both git-ignored; never commit them.
