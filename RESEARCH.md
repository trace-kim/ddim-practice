# Research Idea

**Burst-averaging diffusion** — a DDPM/DDIM-inspired pipeline where the noise
"epsilon" is a real noisy image and the timestep is the number of burst frames
averaged. Implemented as the standalone package `burst_diffusion/`; condensed
writeup in [`burst_diffusion/README.md`](burst_diffusion/README.md), full
documentation in [`docs/burst_diffusion_report.md`](docs/burst_diffusion_report.md)
(experiment report), [`docs/burst_diffusion_guide.md`](docs/burst_diffusion_guide.md)
(user guide), and [`docs/burst_diffusion_method.md`](docs/burst_diffusion_method.md)
(concept + mathematical derivations).

One-paragraph summary: each clean image has N noisy realizations (synthetic
via `noising_pipeline`, or N fast SEM acquisitions of the same area). Averaging
m of them approximates the clean image (`Var ∝ 1/m`), which defines a
frame-averaging degradation schedule `m(t) = T+1−t`. The network is trained to
predict a *fresh* noisy frame from a partial average — whose MSE optimum is the
posterior-mean clean estimate (a noise-level-conditioned Noise2Noise) — and
sampling starts from one real measurement and folds the network's predictions
into a cumulative average, DDIM-style, returning both the final average and
the final prediction. Clean images are never needed for training, so the same
code trains on real equipment bursts with no ground truth.

Status (2026-08-29): implemented, unit-tested (84 tests), and both reference
runs trained and evaluated (`configs/burst_diffusion/*_p10.yml`, 30k steps,
T=15, N=16, Poisson effective peak 10). Datasets match the √N averaging law
with negligible clip bias. Held-out results (mean PSNR/SSIM over 10 val
sources, 64px center crops):

| method (from 1 frame unless noted) | BBBC038 microscopy | MIIC SEM |
|---|---|---|
| single_frame (input)      | 16.8 dB / 0.07 | 14.2 dB / 0.10 |
| avg_of_n (all 16 frames)  | 28.7 dB / 0.50 | 26.2 dB / 0.47 |
| one_shot prediction       | **40.2 dB / 0.97** | **35.4 dB / 0.95** |
| iter_average              | 34.7 dB / 0.90 | 33.6 dB / 0.87 |
| iter_prediction           | 34.0 dB / 0.97 | 35.3 dB / 0.95 |

Findings: (1) the learned model from ONE fast acquisition beats classical
16-frame averaging by ~9-11.5 dB — the headline result; (2) the iterative
sampler did not improve on the one-shot prediction (equal on SEM, worse on
BBBC038) — the documented train/inference gap in action (during iteration the
running average mixes model predictions, whose statistics differ from the real
frame averages seen in training). Next knobs: self-rollout finetuning to close
that gap, t-embedding rescale, T=31/N=32 regeneration, real-burst ingestion.
