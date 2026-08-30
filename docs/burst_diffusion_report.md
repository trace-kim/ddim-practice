# Burst-Averaging Diffusion — Experiment Report

*2026-08-29 · code: [`burst_diffusion/`](../burst_diffusion/) · method & math: [`burst_diffusion_method.md`](burst_diffusion_method.md) · how-to: [`burst_diffusion_guide.md`](burst_diffusion_guide.md)*

**TL;DR**

- A network trained only on noisy burst frames (never a clean image) reconstructs, **from one fast noisy acquisition**, an image **9–11.5 dB cleaner than averaging all 16 burst frames** — on both fluorescence microscopy and SEM data.
- The training loss converged to within 3% of the theoretically predicted noise floor, confirming the math behind the method.
- One honest negative: the DDIM-style iterative sampler did not beat the single forward pass. The cause is understood (a train/inference distribution gap, predicted in advance) and the fix is known. **Update 2026-08-30: the fix (self-rollout finetuning) is implemented and worked — iteration now beats the single forward pass on both datasets (§8).**
- **Metrology repeatability (§9):** re-measuring the same area with 10 fresh frames, the model's outputs move ~10× less per pixel than 8-frame averaging, and the iterated prediction is the most repeatable output on every pixel-level and registration measure (MIIC). **Correction (2026-08-30, §9.1):** an earlier version of this section claimed one-frame CD repeatability *beat* 8-frame averaging (0.80 vs 1.43 px). That claim was withdrawn — it came from a leaked, site-pooled comparison; on content-disjoint scenes avg-of-8 is level with or ahead of the model on CD. Costs: a systematic placement bias (~0.1–0.25 px) and strong content dependence.
- **Leakage correction (§9.1):** the MIIC source corpus ships each image under several filenames (185 unique among 1050 files), so the original MIIC dataset held 68 distinct scenes among 96 sources and 6 of 10 "held-out" sources had a byte-identical training twin. Generation is now content-deduplicated, splitting is content-group-aware, a locked test split exists, and **all MIIC arms were retrained from scratch** (§9.2). BBBC038 was never affected (96/96 unique).

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
network learns to denoise **with no clean target anywhere in the training
loss**. That means the identical pipeline trains on real equipment bursts where
no ground truth exists. (Precisely: the *optimization* never sees a clean
image. These synthetic experiments still use clean images all around the loop —
validation monitoring, evaluation, CD-site selection, the registration
reference — so the honest claim is about the loss, not about the project.)

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
| clean sources | 96 (of 670, CC0) — 96 **distinct** | 96 (NTU MIIC, doi:10.21979/N9/WBLTFI, CC BY-NC 4.0) — only **68 distinct**, see §9.1 |
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
Everything below `avg_of_n` uses **only one noisy frame** as input.

> **MIIC column contaminated (§9.1):** 6 of these 10 MIIC sources have a
> byte-identical training twin, so the MIIC column measures seen-scene /
> fresh-noise performance. Retrained deduplicated MIIC results are in §9.2;
> the BBBC038 column is unaffected. Means are also skewed by a few near-flat
> crops — BBBC038 one-shot has median 37.9 dB against its 40.2 dB mean.

| Method | BBBC038 | MIIC SEM |
|---|---|---|
| single frame (the input) | 16.8 dB / 0.07 | 14.2 dB / 0.10 |
| average of all 16 frames | 28.7 dB / 0.50 | 26.2 dB / 0.47 |
| **model, single forward pass ("one-shot")** | **40.2 dB / 0.97** | **35.4 dB / 0.95** |
| model, iterative sampler — average output | 34.7 dB / 0.90 | 33.6 dB / 0.87 |
| model, iterative sampler — prediction output | 34.0 dB / 0.97 | 35.3 dB / 0.95 |

("One-shot" = the network evaluated once on the raw noisy frame at the highest
noise level — no iterative sampling. It is also, exactly, the first step of
the iterative sampler; the iterative rows differ only in continuing from
there. See the method doc, §4 "One-shot inference".)

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
   intermediate states) — since implemented and confirmed, see §8.

## 6. Limitations

- Synthetic noise only so far (Poisson; Gaussian and mixes supported, salt-and-
  pepper explicitly flagged as incompatible with the MSE objective). Real SEM
  bursts are the next data source; frames must be pixel-aligned (drift-corrected).
- Evaluation used 64×64 center crops; full-image tiled inference is implemented
  (`evaluate --tile`) but not yet benchmarked.
- At effective Poisson peak 10, the very brightest pixels are slightly dimmed by
  the noise pipeline's clipping (worst-case per-pixel bias ~0.05); image-mean
  bias is ≤0.0024 and the model and the averaging baseline are affected
  identically, so comparisons stay fair. Note the check is a *whole-image mean*,
  in which opposite-signed local biases cancel — a per-intensity-bin readout
  would bound the local bias properly (open item).
- 10 validation sources per dataset; T=15/N=16 is the minimum configuration
  (T=31/N=32 is a config-only regeneration).
- **One training seed per arm, and one split seed.** No seed-replication study
  has been run, so effect sizes below ~0.5 dB should be treated as provisional.
- **The headline compares against frame averaging only.** Missing: an
  equal-capacity single-level Noise2Noise model (which would isolate what the
  *schedule* contributes over plain N2N), a clean-supervised oracle (the upper
  bound), and a classical Poisson denoiser such as Anscombe + BM3D. The
  9–11.5 dB headline is a learned prior beating frame integration; it is not
  evidence of a *diffusion*-specific advantage, since one-shot — the source of
  that headline — is a single forward pass, not iterative diffusion.
- The `avg_of_n` reference is weak on PSNR specifically: a fixed Gaussian blur
  already beats avg-of-16 there (§9.1) while ruining CD.

## 7. Next steps

1. ~~Self-rollout finetuning to close the iteration gap.~~ Done — §8.
2. ~~Content-deduplicated MIIC regeneration + retrain of all arms.~~ Done —
   §9.1/§9.2.
3. Matched baselines: single-level Noise2Noise at equal capacity, a
   clean-supervised oracle, and Anscombe+BM3D — the comparison that would turn
   "beats frame averaging" into "beats the alternatives".
4. Multiple training and split seeds; report medians, paired CIs and
   source-clustered statistics rather than site-pooled means.
5. Real SEM burst ingestion (a small manifest-builder; the loader already
   accepts the layout) + drift/registration checking, with CD in physical units.
6. Full-image tiled benchmarking and larger patches (128px fits this GPU).
7. T=31 / N=32 regeneration and re-run (config-only, ~30 min).
8. Provenance: run the burst arms through the `ddimctl` bundle machinery
   (source snapshot, dataset/checkpoint hashes, environment lock, exact
   invocation) instead of ad-hoc run directories.

## 8. Follow-up experiment: self-rollout finetuning (2026-08-30)

The gap diagnosed in §5.3 was attacked with the opt-in `training.rollout`
stage (design and validity argument: method doc §5): each baseline was
finetuned for 10,000 further steps (30k → 40k) with half of every batch
replaced by the sampler's own intermediate pseudo-averages as network
*inputs* — targets stay real fresh frames, which is what keeps the
training grounded in the instrument's noise. Rollout states were generated
on the fly with the EMA weights (the inference distribution), at
~4.2 steps/s (~3.8× slower than baseline training; ~40 min per model on
the same RTX 4060 Ti).

Same held-out sources and protocol as §4; baseline numbers repeated for
comparison. (**MIIC columns contaminated — §9.1**; the BBBC038 columns stand,
except that the rollout-vs-one-shot margin is p = 0.0501, i.e. suggestive
rather than established.)

| Method | BBBC038 base | BBBC038 finetuned | MIIC base | MIIC finetuned |
|---|---|---|---|---|
| single frame (the input) | 16.8 | 16.8 | 14.2 | 14.2 |
| average of all 16 frames | 28.7 | 28.7 | 26.2 | 26.2 |
| one-shot | **40.2** | 39.9 | 35.4 | 35.6 |
| iterative — average output | 34.7 | 36.7 | 33.6 | 33.9 |
| iterative — prediction output | 34.0 | **40.5** | 35.3 | **35.9** |

(PSNR dB; SSIM moved in step: BBBC038 iter-prediction 0.966 → 0.970, MIIC
0.950 → 0.950.) The goal condition **iter_prediction ≥ one_shot now holds
on both datasets** (+0.7 dB BBBC038, +0.4 dB MIIC), and the iterated
prediction is the best single-frame output overall — on BBBC038 it also
beats the *baseline's* one-shot (40.5 vs 40.2).

The controlled input-distribution measurement (validation-set mean
prediction PSNR at t=1, the §5.3 probe generalized to all 10 sources):
feeding the network its own pseudo-average improved from 34.0 → 40.5 dB
(BBBC038) and 35.3 → 35.9 dB (MIIC), while feeding it a *real* 15-frame
average stayed put (44.9 → 44.9 and 39.0 → 38.9) — the finetuning fixed
the foreign-input problem without touching real-input competency. The
remaining pseudo-vs-real gap (4.3 / 3.0 dB) is mostly an information
ceiling, not a remaining defect: the sampler is deterministic, so its
final prediction is a function of the one measured frame, and no
finetuning can make one frame carry the information of fifteen.

Per-source, the headline is the disappearance of the catastrophic cases:
at baseline, iterating could *destroy* an excellent first prediction
(worst BBBC038 source: one-shot 50.1 dB → iterated 30.8 dB, −19.3 dB);
after finetuning that same source iterates to 51.2 dB — *above* its
one-shot. Across all ten BBBC038 validation sources the iterated
prediction now matches or beats one-shot within ±0.5 dB or better.
Gap-closing is uneven (sources with no one-shot-to-iteration deficit,
like the probe's default source 12, barely move; the worst deficits gain
up to +20 dB), so the single-source probe understates the effect — the
validation-set means above are the honest readout.

**Control: is it the rollout, or just 10k more steps?** Because the
finetuned models received 10,000 *additional* gradient steps, a plain
continuation of each baseline to 40k (identical config, no `rollout`
block, same resume) was trained as the **step-matched** control. Note the
label carefully: it matches gradient *steps*, not compute — a rollout step
costs ~3.8× a plain step (each runs a no-grad sampler trajectory), so the
rollout arm consumed roughly 3.8× the additional FLOPs. This control
therefore rules out "more gradient steps", not "more compute"; a
FLOP-matched control (≈38k plain steps) has not been run.

| iter_prediction (PSNR) | BBBC038 | MIIC |
|---|---|---|
| baseline, 30k | 34.00 | 35.25 |
| control: plain continuation, 40k | 33.81 | 35.27 |
| rollout finetune, 40k | **40.54** | **35.94** |

Plain continuation moves iteration by −0.19 / +0.02 dB — nothing. Paired
per-source against the control, the rollout arm gains +6.73 dB (10/10
sources, t = 3.2) on BBBC038 and +0.67 dB (8/10, t = 3.6) on MIIC: the
improvement is attributable to the rollout mechanism, not to extra
training. (Evaluation itself is deterministic — re-running the baseline
eval reproduces its numbers exactly — and no checkpoint selection was
involved anywhere: every arm is its config-fixed final step.)

Costs and effect sizes, stated plainly: one-shot on BBBC038 slipped
40.2 → 39.9 dB — the control held 40.24, so the −0.35 dB (per-source
−0.37, 1/10 positive, t = −2.5) is genuinely the price of sharing half
the gradient signal with the second input manifold (on MIIC the control
shows no one-shot effect at all, t = 0.6). And the *margin* of the
finetuned iteration over the finetuned one-shot is small — +0.67 dB
(8/10 sources, t = 2.3) on BBBC038, +0.37 dB (9/10, t = 3.2) on MIIC —
necessarily so: both outputs are deterministic functions of the same
single frame, so iteration can only read that frame somewhat better,
never add measurements. The large, unambiguous effect is the repair of
iteration itself (+6.5 dB, 10/10 sources on BBBC038); the win over
one-shot is consistent but modest. If one-shot is the only output a
deployment uses, skip the finetune or lower `rollout.fraction`; if the
iterative outputs matter (they are now the best ones), the trade is
clearly favorable.

## 9. Repeatability & metrology evaluation (2026-08-30)

> **Read §9.1 first.** An internal audit found that the MIIC dataset used in
> §4, §8 and in the tables below contains duplicated source content, so its
> "held-out" sources were partly seen in training, and that the CD headline
> below does not survive a scene-level reanalysis. The affected claims are
> corrected in §9.1 and the retrained, deduplicated results are in §9.2. The
> tables in this subsection are retained verbatim as the record of what was
> originally measured, not as current claims.

Everything above scores *accuracy* (PSNR/SSIM against clean). For metrology
the more important axis is *repeatability*: measure the same area again and
ask how much the reported number moves. This experiment simulates exactly
that. The sampler is deterministic, so its only source of run-to-run
variation is the input frame — feeding each method **10 different fresh
frames (seeds) of the same measured area** and comparing the outputs
isolates the measurement noise each method transmits. Implemented as
`python -m burst_diffusion repeatability` (guide §5); both reference
datasets, both checkpoint arms (baseline 30k, rollout-finetuned 40k), same
held-out sources and center crops as §4/§8.

**Methods and metrics.** Classical rows: the raw frame (10 independent
seeds) and *disjoint*-group averages — 8× `avg_of_2`, 4× `avg_of_4`,
2× `avg_of_8`; `avg_of_16` has one realization, so it contributes accuracy
and bias but no precision estimate. Model rows consume one seed each, so
their 10 realizations are fully independent. Reported per method: per-pixel
std across seeds (c4-debiased — comparing K=2 and K=10 sample stds without
the correction would understate the former by ~20%); **CD** measured
CD-SEM-style (16-row band profile, 50% threshold between robust profile
extremes, subpixel linear-interpolated crossings) at fixed sites
auto-selected on the clean image — pooled σ across (source, site), bias vs
the clean-image CD, success rate; **registration** as feature-center
precision at those sites plus global sub-pixel image shift vs clean
(upsampled cross-correlation). Registration is gated per source: a crop
whose clean image cannot even be registered against its own 16-frame
average (> 0.5 px — featureless area) is excluded for all methods
identically (10/10 MIIC sources pass, 7/10 BBBC038; 21 CD sites on MIIC,
27 on BBBC038).

**MIIC SEM** (10 seeds, pixel σ in [0,1]×10⁻³, CD/registration in pixels):

| method | PSNR dB | pixel σ | CD 3σ | CD bias | center σ | shift σ | shift bias |
|---|---|---|---|---|---|---|---|
| single frame | 14.2 | 194.1 | 2.590 | −0.068 | 0.416 | 1.765 | 0.264 |
| avg of 2 | 17.2 | 137.1 | 2.169 | +0.003 | 0.296 | 1.005 | 0.037 |
| avg of 4 | 20.2 | 96.9 | 1.485 | +0.035 | 0.195 | 0.393 | 0.030 |
| avg of 8 | 23.2 | 68.6 | 1.434 | +0.041 | 0.124 | 0.142 | 0.066 |
| avg of 16 | 26.2 | — | — | +0.048 | — | — | 0.055 |
| one-shot (base) | 35.6 | 8.5 | 1.003 | +0.045 | 0.219 | 0.377 | 0.121 |
| iter-avg (base) | 33.7 | 15.1 | 0.967 | +0.044 | 0.207 | 0.472 | 0.094 |
| iter-pred (base) | 35.4 | 6.5 | 0.862 | +0.042 | 0.194 | 0.163 | 0.115 |
| one-shot (rollout) | 35.7 | 8.2 | 0.938 | +0.034 | 0.213 | 0.330 | 0.115 |
| iter-avg (rollout) | 33.9 | 14.9 | 0.911 | +0.032 | 0.201 | 0.455 | 0.130 |
| **iter-pred (rollout)** | **36.1** | **6.1** | **0.797** | +0.028 | 0.186 | 0.170 | 0.134 |

**BBBC038** (same columns): single frame 146.9 / 3.97 / 1.10; avg of 8
52.1 / **1.88** / 0.59; one-shot (rollout) 7.2 / 2.65 / 0.48; iter-pred
(rollout) **5.7** / 2.56 / 0.50 (pixel σ / CD 3σ / shift σ; full tables in
`runs/burst_diffusion/*/repeatability/summary.md`).

Findings, in order of confidence:

1. **Pixel-level repeatability from one frame is ~10× better than 8-frame
   averaging** (6–8 vs 52–69 ×10⁻³ per-pixel σ), on both datasets and both
   arms. The classical ladder reproduces the √2-per-doubling law exactly
   (194 → 137 → 97 → 69), so the comparison sits on a verified baseline.
2. **Iteration's real payoff is precision, not PSNR.** §8 found the
   finetuned iterated prediction beats one-shot by a modest +0.4/+0.7 dB;
   here it beats one-shot on *every* precision metric on MIIC: pixel σ
   −26% (8.2 → 6.1), CD 3σ −15% (0.94 → 0.80), global shift σ −48%
   (0.330 → 0.170). Mechanistically: the final prediction is conditioned on
   a pseudo-average in which the seed frame carries weight 1/15, so the
   output is a much flatter function of the input noise than the direct
   t=T evaluation. The repeatability axis, invisible to PSNR, is where the
   iterative sampler earns its cost.
3. **The `iter_average` output obeys its noise budget.** The closed form
   says it retains n₁/16 of the input noise: predicted floor
   194.1/16 = 12.1×10⁻³, plus a one-shot-sized model term ⇒ ~14.7×10⁻³;
   measured 14.9. Its precision can never beat the prediction output —
   another reason `prediction` is the deployment output.
4. **CD: the model beats classical averaging on hard SEM edges, not on soft
   fluorescence edges.** MIIC: 0.80–1.00 px 3σ from one frame vs 1.43 for
   avg-of-8. BBBC038: models 2.56–2.79, avg-of-8 1.88 — the model only
   reaches avg-of-4 territory (2.99) there. Metrology conclusions from one
   content type do not transfer automatically.
5. **CD bias equals the averaging bias.** Model CD bias on MIIC
   (+0.028…+0.045 px) matches avg-of-16 (+0.048) — Theorem 1 in metrology
   units: the network converges to the same clipped-mean limit as frame
   averaging, so it inherits its (tiny) bias, while the raw frame shows a
   noise-induced −0.068 px estimator bias. CD success is 100.0% on MIIC
   for every method, 93–96% on BBBC038 (soft edges lose a crossing
   occasionally, raw frame worst).
6. **Registration: precise but not unbiased.** Seed-to-seed shift precision
   of the iterated prediction (0.17 px) sits between avg-of-8 (0.14) and
   avg-of-4 (0.39); feature-center precision (0.19 px) ≈ avg-of-4. But the
   model introduces a small systematic placement offset vs clean: 0.12–0.13
   px mean shift (3–6× its standard error, so real; per-source mean
   magnitudes ~0.24 px vs 0.16 for avg-of-16). For overlay-critical use
   this bias would need a calibration pass; classical averaging does not
   have it (0.03–0.07 px, noise-consistent).
7. **The model's residual variation is edge-concentrated.** The σ-maps
   (`sigma_maps.png`) are near-black in flat regions and light up exactly
   on feature contours (p95/mean σ ratio 2.8 for iter-pred vs 1.5 for the
   raw frame at equal K) — the opposite of averaging's spatially uniform
   noise. Pixel-mean σ therefore *flatters* the model where metrology
   looks; the honest number for measurement use is the CD/registration σ,
   which is why both are reported. Even so, the model wins CD 3σ on MIIC.
8. **Rollout finetuning helps precision too**: on MIIC it improves every
   model row over baseline (CD 3σ 0.86 → 0.80, pixel σ 6.5 → 6.1, and
   +0.8 dB PSNR on the iterated prediction) — its value is larger than the
   §8 PSNR deltas alone suggested. Caveat: the avg-of-8 precision rows rest
   on only 2 realizations × 10 sources (13–21 dof), so their σ estimates
   carry ~±15–20% uncertainty (visible as avg-of-8 ≈ avg-of-4 CD 3σ on
   MIIC); the model rows have 9 dof per site and are correspondingly
   tighter.

Practical summary for metrology: from a single fast acquisition, the
finetuned iterated prediction delivers CD repeatability better than 8-frame
averaging on SEM-like content, ~10× better pixel-level repeatability, and
registration precision at the 4–8-frame-average level — at the cost of a
~0.1–0.25 px placement bias that would need one-time calibration, and with
the explicit caveat that soft-edged content (BBBC038) keeps classical
averaging competitive on CD. Next data-side step unchanged from §7: real
SEM bursts, where fixed-pattern noise (correlated across frames, §6 of the
method doc) is the assumption most likely to move these numbers.

*(Findings 2, 4, 7 and 8 and this practical summary are superseded by §9.1;
findings 1, 3, 5 and 6 survive the reanalysis. Read on.)*

## 9.1 Audit and corrections (2026-08-30)

An internal audit of the whole experimental record found two defects that
invalidate specific claims above. Both were verified independently before
being accepted, and the fixes are in the code, not only in prose.

### Defect 1: duplicated source content in the MIIC datasets

The MIIC source corpus (`paired_miic_v0/train/target_sem`) contains **185
unique images among 1050 files** — the same picture is shipped under several
filenames. `generate` selected sources by filename, so the 96 staged MIIC
sources held only **68 distinct scenes**, and `BurstCache` split by source
*index*, so duplicated content landed on both sides. Six of the ten reported
MIIC validation sources (12, 15, 16, 24, 31, 37) had a byte-identical
training twin; only 29, 62, 72 and 88 were genuinely unseen. The noisy
frames differ, so this is *seen-scene / fresh-noise* leakage rather than
outright train-on-test, but every MIIC held-out number in §4, §8 and §9 is
affected and none of them measures unseen-content generalization.

The same defect is worse in the older `ddimctl` SEM run
(`runs/2026-08-24/...__sem-ddim-local-32-full__...`): 1050 files with 185
unique contents split by file index, so **every** validation file has an
exact training duplicate (and the upstream corpus's own nominal val/test
directories overlap its train directory completely). That run stands as a
workflow demonstration only; its validation loss is not evidence of
generalization. It was never a denoising experiment (unconditional DDPM,
Gaussian-noise prediction), so no denoising claim rests on it.

**Fixed in code, with regression tests:**

- `generate` now **content-deduplicates before selection**
  (`select_unique_sources`): candidates are visited in seeded order, the
  prepped array is hashed (so re-encoded duplicates are caught too), and a
  duplicate is skipped in favor of the next candidate. `sources.json` records
  every source's `content_sha256` and every skipped duplicate; `stats.json`
  reports `unique_contents`.
- `BurstCache` now splits by **content group**, never by index: sources whose
  clean images hash equal move together, and a duplicated dataset warns. With
  all-distinct content this reduces exactly to the previous index
  permutation, so BBBC038's split — and every BBBC038 number in this report —
  is unchanged and reproducible.
- A **locked test split** (`data.test_fraction`) is anchored at the end of
  the group permutation, so widening `val_fraction` during development can
  never move it.
- Tests assert no content crosses a split, that a duplicate-ridden corpus
  still yields a distinct-scene dataset, and that `test_fraction: 0`
  reproduces the historical split.

**BBBC038 is unaffected** (96 files, 96 unique contents), which the same
hash audit confirms; its results stand as reported.

### Defect 2: the CD headline does not survive a scene-level reanalysis

§9 finding 4 claimed the model's one-frame CD 3σ (0.797 px) beats
8-frame averaging (1.434 px) on MIIC. That comparison pools **sites**, but
sites inside one scene share the scene, the frames and the model outputs, so
they are not independent experimental units; the independent unit is the
source. Only 5 of 10 MIIC scenes yielded CD sites, and 14 of 21 sites came
from *leaked* scenes. Recomputing per scene from the saved measurements:

| scene | status | sites | avg-8 CD 3σ | model CD 3σ | winner |
|---|---|---|---|---|---|
| 16 | leaked | 6 | 2.647 | 0.972 | model |
| 24 | leaked | 4 | 0.723 | 0.643 | model |
| 29 | **unseen** | 4 | 0.432 | 0.774 | avg-8 |
| 31 | leaked | 4 | 0.375 | 0.436 | avg-8 |
| 88 | **unseen** | 3 | 0.438 | 1.001 | avg-8 |

The model wins 2 of 5 scenes, both leaked; avg-8 wins both genuinely unseen
scenes. Pooled over the unseen scenes only, avg-8 is 0.420 px vs the model's
0.875 px. The site-pooled headline was carried by leaked scene 16, where
avg-8 happened to draw a large σ on 6 degrees of freedom. An independent
re-evaluation on 32 content-disjoint MIIC scenes agrees: avg-8 0.537 px vs
rollout 0.574 px.

**The claim "one-frame CD repeatability beats 8-frame averaging" is
withdrawn.** What the evidence supports is that one-frame CD repeatability
is *comparable to* 8-frame averaging (and clearly better than 4-frame),
which is still a strong result for a single acquisition — but it is not
superiority, and it is not what was claimed.

**Fixed in code:** `repeatability` now computes and reports per-scene pooled
sigmas (`cd.scene_sigmas_px`), their median (`cd.scene_median_3sigma_px`),
and makes the **scene-level median the headline column** in `summary.md`,
with the site-pooled value retained beside it. The registration section
gained the same scene-level median.

### Other corrections applied

- **"Every repeatability metric"** (finding 2, and the earlier TL;DR) was
  false as stated: it holds on MIIC but not on BBBC038, where iteration's
  shift σ is 0.498 vs one-shot's 0.483 px. Scoped to MIIC.
- **Bias tables displayed only the signed mean**, in which opposite-sign
  sites cancel: MIIC rollout CD bias reads +0.028 px signed but 0.224 px
  mean-absolute (BBBC038: −0.049 vs 0.496). Both are now shown — they answer
  different questions (systematic offset vs typical per-site magnitude), and
  finding 5's "model bias ≈ averaging bias" is a statement about the *signed*
  column only.
- **The §8 rollout-vs-one-shot margin on BBBC038 is not significant** at the
  conventional threshold: +0.667 dB, paired 95% CI [−0.0004, 1.335],
  p = 0.0501, n = 10. Reported as suggestive, not established. (The much
  larger effect — rollout repairing iteration itself, +6.5 dB, 10/10 sources
  — is unaffected.)
- **The repeatability comparison omitted the step-matched 40k controls.**
  Adding them shows part of what §9 finding 8 attributed to rollout is plain
  continuation: MIIC iter-pred pixel σ 6.51 → 6.34 (plain) → 6.06 (rollout)
  ×10⁻³, CD site-pooled 3σ 0.862 → 0.842 → 0.797, shift σ 0.163 → 0.164 →
  0.170 (rollout marginally *worse* on shift). Rollout-specific gains are
  real but smaller than the baseline→rollout tables implied.
- **PSNR means are skewed by near-flat crops.** BBBC038 baseline one-shot has
  mean 40.2 dB but median 37.9 dB, lifted by three ~49–51 dB crops. Medians
  should accompany means.
- **Frame averaging is not a strong PSNR baseline.** A fixed Gaussian blur of
  the *single* frame (σ swept over 0.8–2.2 on the same validation center
  crops; best is σ = 2.2) reaches 32.5 dB (BBBC038) / 27.9 dB (MIIC) —
  *above* avg-of-16's 28.7 / 26.2 — while destroying exactly the edges CD
  measures. This does not threaten the model's margin (35–40 dB), but it does
  mean `avg_of_n` alone is too weak a reference for a PSNR headline: the
  honest comparison needs a denoiser-class baseline (Anscombe + BM3D) and an
  equal-capacity single-level Noise2Noise model, which would also isolate
  what the *schedule* adds over plain N2N (§7 item 3).
- **"No clean image is ever used"** overstated the self-supervision claim and
  is now "no clean target in the training loss" throughout (§2, method doc
  §3). Clean images are used for monitoring, evaluation, CD-site selection
  and the registration reference.
- **Repeatability is necessary, not sufficient, for metrological validity.** A
  model can be repeatable by stably suppressing or hallucinating structure.
  The CD sites here are oracle-selected from the clean image, the scale is
  pixels rather than physical units, and "3σ" is a convention, not a tested
  Gaussianity claim. Finding 7 (edge-concentrated residual variance) is the
  reason pixel-mean σ flatters the model, and its last sentence ("Even so,
  the model wins CD 3σ on MIIC") is withdrawn with defect 2.
