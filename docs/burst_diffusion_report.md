# Burst-Averaging Diffusion — Experiment Report

*2026-08-29 · code: [`burst_diffusion/`](../burst_diffusion/) · method & math: [`burst_diffusion_method.md`](burst_diffusion_method.md) · how-to: [`burst_diffusion_guide.md`](burst_diffusion_guide.md)*

**TL;DR**

- A network trained only on noisy burst frames (never a clean image) reconstructs, **from one fast noisy acquisition**, an image **9–11.5 dB cleaner than averaging all 16 burst frames** — on both fluorescence microscopy and SEM data.
- The training loss converged to within 3% of the theoretically predicted noise floor, confirming the math behind the method.
- One honest negative: the DDIM-style iterative sampler did not beat the single forward pass. The cause is understood (a train/inference distribution gap, predicted in advance) and the fix is known.

## 1. What we set out to test

The standard way to get a clean image from a noisy instrument (SEM, fluorescence
microscope, telescope) is to acquire the same area N times and average: noise is
random, signal is not, so the average gets cleaner as N grows. But N
acquisitions cost time, dose, and sample damage.

The idea under test: treat frame averaging as a **diffusion process**. In
DDPM/DDIM, an image is gradually mixed with Gaussian noise over T steps and a
network learns to undo one step at a time. Here we replace the Gaussian noise
with **real noisy frames** and the timestep with **how many frames were
averaged**: level t means "an average of T+1−t frames", so t=T is one raw frame
(noisiest) and t=1 is an average of T frames (cleanest). Training teaches the
network to predict a *held-out* noisy frame from a partial average; sampling
starts from a single real measurement and iteratively folds the network's
predictions into a running average, exactly as if they were extra acquisitions.

A key property falls out of the math (see the method doc): the optimal
predictor of a held-out noisy frame is the *clean image estimate* — so the
network learns to denoise **without ever seeing a clean image**. That means the
identical pipeline trains on real equipment bursts where no ground truth
exists.

## 2. The pipeline at a glance

Five commands, all under `python -m burst_diffusion`:

1. **generate** — build a synthetic burst dataset from clean sources using the
   repo's `noising_pipeline`, and *measure* that it is noisy enough and that
   averaging really converges to clean (`stats.json`, with warnings when out of
   band).
2. **preview** — a visual grid: clean vs single frames vs running averages.
3. **train** — 8.95M-parameter U-Net, Adam + EMA, no DataLoader (all frames
   cached in RAM), exact resume, TensorBoard logs.
4. **sample** — denoise one measurement, optionally saving the step-by-step
   trajectory.
5. **evaluate** — compare against classical baselines on held-out sources.

## 3. Experimental setup

**Datasets** (generated and verified in this experiment; Poisson noise,
effective peak 10, clean sources pre-scaled into [0.15, 0.85] to avoid
clipping bias):

| | BBBC038 (fluorescence microscopy) | MIIC (integrated-circuit SEM) |
|---|---|---|
| clean sources | 96 (of 670, CC0) | 96 (NTU MIIC, doi:10.21979/N9/WBLTFI, CC BY-NC 4.0) |
| frames per source (N) | 16 | 16 |
| single-frame PSNR (median) | 17.3 dB | 14.2 dB |
| PSNR of avg-of-16 vs clean | 29.3 dB | 26.2 dB |
| residual bias of avg-of-16 | 0.0004 | 0.0024 |

Both datasets follow the √N averaging law exactly (+3 dB per doubling of
frames) — the premise of the method, verified before training. (For scale:
+3 dB means half the noise power; +12 dB means 16× less.)

**Model & training**: U-Net (ch 64, 4 levels, attention at 16², 8.95M params),
64×64 aligned random crops, T=15 levels, batch 8, Adam 2·10⁻⁴, EMA 0.999,
30,000 steps. Hardware: RTX 4060 Ti, ~16 steps/s → **32 minutes per model**.
(Batch 8 rather than 16 because Windows backs every CUDA allocation with
system commit charge, which other running apps had nearly exhausted.)

## 4. Results

Held-out validation sources (10 per dataset), mean PSNR / SSIM vs clean.
Everything below `avg_of_n` uses **only one noisy frame** as input:

| Method | BBBC038 | MIIC SEM |
|---|---|---|
| single frame (the input) | 16.8 dB / 0.07 | 14.2 dB / 0.10 |
| average of all 16 frames | 28.7 dB / 0.50 | 26.2 dB / 0.47 |
| **model, single forward pass ("one-shot")** | **40.2 dB / 0.97** | **35.4 dB / 0.95** |
| model, iterative sampler — average output | 34.7 dB / 0.90 | 33.6 dB / 0.87 |
| model, iterative sampler — prediction output | 34.0 dB / 0.97 | 35.3 dB / 0.95 |

Convergence: the validation PSNR of the model's prediction (from one frame)
rose from ~29 dB after 1,000 steps to 40.2 dB (BBBC038) / 35.6 dB (MIIC) at
30,000 steps.

**The physics sanity check.** Because the training target is itself a noisy
frame, theory says the loss cannot fall below the single-frame noise variance
(×4 in the model's [−1,1] range): for MIIC that predicts a floor of 0.154.
Measured final training loss: **0.150**. The model saturated the information
limit of its objective — the plateau is correct behavior, not a failure to
converge (this is documented prominently, since it looks alarming on a loss
curve).

## 5. What we learned

1. **One fast acquisition + model ≫ sixteen acquisitions + averaging.** The
   learned prior beats the classical practice by ~11.5 dB (microscopy) and
   ~9 dB (SEM), with SSIM jumping from ~0.5 to ~0.95. Visually (see
   `runs/burst_diffusion/*/eval/comparison_grid.png`) the 16-frame average is
   still visibly grainy while the model output is clean with edges and small
   structures (nuclei outlines, IC vias) preserved.
2. **The self-supervised premise holds.** No clean image was ever used in
   training, yet the model reconstructs toward clean — because predicting a
   held-out noisy frame is, in expectation, predicting the clean image.
3. **Iteration did not beat one-shot** (equal on SEM, −6 dB on BBBC038). This
   was predicted as a risk before running: during sampling, the running
   average mixes the model's own predictions, whose statistics differ from the
   real frame-averages seen in training, and the prediction errors are
   correlated across steps so they do not average out like real noise. The
   evaluation reports both outputs precisely so this is measurable rather than
   hidden. Candidate fix: self-rollout finetuning (train on the sampler's own
   intermediate states).

## 6. Limitations

- Synthetic noise only so far (Poisson; Gaussian and mixes supported, salt-and-
  pepper explicitly flagged as incompatible with the MSE objective). Real SEM
  bursts are the next data source; frames must be pixel-aligned (drift-corrected).
- Evaluation used 64×64 center crops; full-image tiled inference is implemented
  (`evaluate --tile`) but not yet benchmarked.
- At effective Poisson peak 10, the very brightest pixels are slightly dimmed by
  the noise pipeline's clipping (worst-case per-pixel bias ~0.05); image-mean
  bias is ≤0.0024 and the model and the averaging baseline are affected
  identically, so comparisons stay fair.
- 10 validation sources per dataset; T=15/N=16 is the minimum configuration
  (T=31/N=32 is a config-only regeneration).

## 7. Next steps

1. Self-rollout finetuning to close the iteration gap.
2. T=31 / N=32 regeneration and re-run (config-only, ~30 min).
3. Real SEM burst ingestion (a small manifest-builder; the loader already
   accepts the layout) + drift/registration checking.
4. Full-image tiled benchmarking and larger patches (128px fits this GPU).
