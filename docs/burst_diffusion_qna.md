# Burst-Averaging Diffusion — Questions & Answers

A record of the design-review questions asked after the first reference
experiment (2026-08), with full answers and the measurements that back them.
Each entry gives the short answer first, then the reasoning. Companion
documents: [method & derivations](burst_diffusion_method.md) ·
[experiment report](burst_diffusion_report.md) ·
[user guide](burst_diffusion_guide.md).

Most numeric claims below can be reproduced live with
`python tools/probe_burst_predictions.py` (a diagnostic that feeds the trained
model different input types and measures where its predictions land).

---

## Q1. What exactly is the "single forward pass" (one-shot)?

**Short answer:** calling the trained U-Net once on the raw noisy measurement
at the highest noise level — `model(noisy_frame, t=T)` — and using its output
directly as the denoised image. No loop.

The U-Net is a function of an image and a noise level. One-shot is one
evaluation at $t = T$ ("this input is a single raw frame"). It is not a
separate mechanism from the iterative sampler: it is *exactly the sampler's
first network call* (equivalently, the sampler run with the length-one
schedule $[T]$, which is how `evaluate` computes the `one_shot` row). The
iterative rows in the results differ only in continuing from there — folding
the prediction into a running average and re-querying at lower $t$.

Why a single call already denoises here — but not in DDPM — is Q3 and Q5.

---

## Q2. What is Noise2Noise, and how does this work relate to it?

**Short answer:** Noise2Noise (Lehtinen et al., 2018) shows a denoiser can be
trained with *noisy targets* instead of clean ones, and converges to the same
function as clean-target training. Our Theorem 1 is that argument; the burst
structure extends it with a noise-level schedule, $t$-conditioning, and a
diffusion-style sampler.

Supervised denoising wants (noisy, clean) pairs, but a clean image often
cannot be acquired (dose limits, beam damage). Noise2Noise replaces the clean
target with a *second independent noisy observation* of the same scene. Under
$L^2$ loss the optimal predictor is the conditional mean, and independent
zero-mean noise vanishes inside a conditional mean:
$\mathbb{E}[y^{\text{tgt}} \mid y^{\text{in}}] = \mathbb{E}[x_0 \mid y^{\text{in}}]$.
The noisy target is an unbiased stand-in — gradient noise grows, the optimum
does not move. (The loss must match the noise: $L^2$ for zero-mean noise;
impulse noise like salt-and-pepper needs a median-seeking loss, which is why
the generator warns about it.)

Relation to this work, precisely:

- the `fresh` target rule ($K \notin S$) **is** Noise2Noise's independence
  condition; Theorem 2 shows the degeneracy when it is violated;
- the input side is a *partial average*, so one burst yields training pairs at
  $T$ input-noise levels instead of N2N's single level;
- the network is $t$-conditioned — one model spanning the schedule;
- the diffusion-style sampler is wrapped around that family — this is the
  novel part the experiment tested (and, in v1, the part that did not add
  value over one-shot; see Q4).

Honest framing: the winning `one_shot` result is essentially a noise-level-
conditioned Noise2Noise denoiser; the experiment's open question was whether
iteration adds anything on top.

---

## Q3. The loss trains the network to predict a *noisy frame*. Why does its output end up close to the *clean* image? Was the objective even set correctly?

**Short answer:** because MSE turns "predict a random thing" into "predict its
average", and the average of `clean + zero-mean noise` is the clean image.
The objective is verified correct three independent ways below.

The network is not asked to predict "the noisiest image" — it is asked to
predict **a randomly chosen fresh frame it has never seen**. The dice analogy:
predict a fair die roll under squared error, and your best answer is 3.5 (the
mean), never an actual face. During training, the *same input* appears with
*different targets* on different steps (frame 3 today, frame 11 tomorrow);
each pulls the output toward itself; the only stationary point is the mean of
the pulls — the clean image. The network never "chooses" to denoise:
denoising is the unique loss-minimizing compromise between targets it cannot
tell apart. Two conditions make this work: the target's noise must be
**zero-mean** (assumption A1) and **independent of the input** — which is
exactly why the target frame is *excluded* from the averaged subset
(`target_mode: fresh`; the included variant provably collapses to the
identity, method doc Theorem 2).

The three receipts (from the trained BBBC038 model, held-out source):

1. **Location of the prediction.** If training had produced a noisy-image
   generator, its output would be close to noisy frames and far from clean.
   Measured, it is the reverse:

   | distance | value |
   |---|---|
   | prediction vs clean | 36–43 dB (close) |
   | prediction vs any actual noisy frame | ~17.2 dB (far — exactly that frame's own noise distance) |
   | raw frame vs clean | 17.3 dB |
   | raw frame vs another raw frame | 14.3 dB |

   Note the last row: two noisy frames are ~14 dB apart *from each other* —
   even a perfect noisy-image generator could never be closer than ~17 dB to
   a particular unseen frame, because that frame's noise is unguessable.

2. **The loss floor.** $\mathbb{E}\lVert\hat\varepsilon - y\rVert^2 =
   \mathbb{E}\lVert\hat\varepsilon - x_0\rVert^2 + \sigma^2$ exactly, so the
   loss cannot go below the target's noise variance and reaches it only when
   the output is the clean estimate. Measured: final training loss 0.150 vs
   predicted floor 0.154 (MIIC); per-image,
   $\mathrm{MSE}(\text{prediction}, \text{fresh frame}) = 0.0191$ vs
   $\sigma^2 = 0.0186$. The objective was not just correct — it was
   *saturated*.

3. **Unit tests** pin the mechanics: the fresh target is asserted to be
   excluded from the averaged subset across hundreds of draws, the subset
   averaging is checked exactly, and the sampler reproduces true burst
   averaging when fed real frames.

---

## Q4. Isn't the iteration just a cumulative average of clean images? Why would that be *worse* than one-shot?

**Short answer:** it is not an average of clean images — it is
`(1 raw noisy frame + T predictions) / (T+1)`, the predictions are *correlated*
(so their errors do not cancel), and predictions after the first are made
from inputs the network never saw in training, which measurably degrades
them.

Three corrections to the premise:

1. **The raw frame stays in.** By the sampler's closed form,
   $x^{\text{avg}} = (y_1 + \sum_t \hat\varepsilon_t)/(T+1)$: the noisy
   measurement keeps weight $1/(T+1)$ forever.
2. **The predictions are not independent.** All $T$ predictions are
   deterministic functions of the same $y_1$. Averaging them is asking the
   same witness fifteen times — correlated errors do not cancel the way
   independent frame noise does (which is the only reason real burst
   averaging works).
3. **Predictions 2..T come from out-of-distribution inputs.** In training, a
   level-$t$ input is `clean + white grainy noise of variance` $\sigma^2/m$.
   The sampler's pseudo-average is `clean +` $\sigma^2/m^2$ `of real noise +
   smooth correlated model error` — much cleaner than $t$ promises, with the
   wrong noise texture. The network's correction is calibrated per level,
   like a lab tech told "shot at ISO 6400, compensate accordingly" and handed
   an ISO 100 photo: it miscorrects.

The controlled measurement (same network, same $t = 1$, different input):

| input at $t=1$ | prediction vs clean |
|---|---|
| REAL average of 15 frames (in-distribution) | **42.8 dB** |
| the sampler's own pseudo-average | **36.0 dB** |

6.8 dB lost purely to the input distribution. Fed real $m$-frame averages the
prediction improves monotonically (36.2 → 42.8 dB for $m = 1..15$) — the
denoiser family is healthy; only the self-generated inputs are foreign.
Across the validation set, the mean *first* prediction of the trajectory
scores 40.2 dB and the mean *last* prediction 34.0 dB.

Practical corollary: the $t$-conditioning is useful today for *real*
multi-frame inputs — acquire $k$ real frames, average, and query at the
matching level (`Sampler.run(x, schedule=[T+1-k, ...])`).

---

## Q5. In DDPM the U-Net "learns the Gaussian noise". By the same conditional-mean argument, shouldn't DDPM collapse to a **zero array** (the mean of the Gaussian)? And is our collapse-to-clean because clean-image information is embedded in the noisy frame?

**Short answer:** DDPM does not collapse because its target is *inside* its
input — the conditional mean $\mathbb{E}[\varepsilon \mid x_t] \ne 0$. Your
zero-array intuition is exactly right for the variant you describe: a *fresh*
Gaussian target would collapse DDPM to zeros. And yes — the embedded clean
information is precisely why our fresh-target choice works instead of
zero-collapsing.

First, a premise correction: DDPM's U-Net does not learn the Gaussian
*distribution* (that is fixed and known). It learns to **identify the specific
noise realization inside its input**. Since
$x_t = \sqrt{\bar\alpha}\,x_0 + \sqrt{1-\bar\alpha}\,\varepsilon$, the target
is algebraically entangled with the input, and

$$\mathbb{E}[\varepsilon \mid x_t] = \frac{x_t - \sqrt{\bar\alpha}\,\mathbb{E}[x_0 \mid x_t]}{\sqrt{1-\bar\alpha}} \ne 0 .$$

The unconditional mean of $\varepsilon$ is zero, but the network never
predicts unconditionally — it sees $x_t$. And that formula shows
$\hat\varepsilon$-prediction and $\hat x_0$-prediction are the *same estimate
in two coordinate systems*: "learning the noise" in DDPM **is** learning the
clean image, through the learned image prior that separates signal from
noise.

The full symmetry — each framework is forced onto the opposite target choice
by what its $\varepsilon$ *contains*:

| | DDPM | burst diffusion |
|---|---|---|
| what $\varepsilon$ is | pure Gaussian noise — no signal, mean $0$ | a real frame — clean image + noise, mean $x_0$ |
| predict an **included** $\varepsilon$ (inside the input) | works: the weighted mixture is recoverable via the image prior | fails: identity collapse (Theorem 2) |
| predict a **fresh** $\varepsilon$ (independent of the input) | fails: **zero-array collapse** (the questioner's scenario) | works: collapses onto $\mathbb{E}[x_0 \mid \text{input}]$ — the denoiser |
| so training must target | the included realization | a fresh frame |
| what $\hat\varepsilon$ means at inference | the input's own noise (a full-strength clean estimate in disguise) | the clean-image estimate directly |

Both methods *exploit* a collapse to the conditional mean; the art is
arranging the target so the collapse lands somewhere useful.

On the embedded-clean-information question — yes, and it plays two distinct
roles:

1. **In the target frame:** the fresh frame is $x_0 + $ unpredictable noise,
   so its conditional mean is $x_0$, *not* $0$. This is exactly why our
   fresh-target training does not zero-collapse the way fresh-target DDPM
   would.
2. **In the input frame:** the input also contains $x_0$, which makes the
   prediction specific to *this* image. With no input at all, the optimum
   would be the dataset-average image — a gray blur. The input supplies the
   "which image"; the target's structure supplies the "clean".

---

## Q6. Wouldn't self-rollout finetuning make the model learn *its own* noise distribution instead of the real equipment noise? Is it still valid?

**Short answer:** no — provided one invariant holds: **model outputs enter
only as inputs; the targets remain real measured frames, always.** Theorem 1
never uses how the input was manufactured, so the optimum on pseudo-inputs is
still $\mathbb{E}[x_0 \mid \text{input}]$, graded by reality.

A self-rollout finetuning batch: run the sampler a few steps on a real frame,
take an intermediate pseudo-average as the *input*, and train against **a
real fresh frame from the same burst**. The target's noise is still real,
zero-mean, and independent of everything — including the model's own
machinations — so the conditional-mean argument goes through unchanged. The
network learns to *finish the job from its own intermediate states*, but it
is never rewarded for agreeing with itself.

The failure mode the question correctly worries about is the converse: model
outputs on the **target** side (self-distillation). Then the loss rewards
self-agreement and the model can drift toward its own artifacts with nothing
anchoring it to the instrument. (Imitation-learning analogy: this is DAgger —
train on *your own policy's states* with *expert labels*. Own states, real
labels: safe. Own labels: drift.)

Where "learning the real noise" actually lives in this architecture: the
model never generates noise; it learns a correction *calibrated to the real
noise's statistics* (variance per level, signal-dependence, spatial texture)
because it minimizes MSE under exactly that noise on the target side and on
the real-average inputs. Self-rollout changes neither — real frames stay as
every target, and real-average inputs stay in the training mix (augment,
never replace).

Open design choices (validity does not depend on them; performance will):
what $t$ means for pseudo-inputs (nominal sampler step vs matched noise
level), rollout depth distribution, and EMA-vs-live weights for generating
rollouts. Success criteria are measurable: after finetuning, the A/B curves
of `tools/probe_burst_predictions.py` should merge (baseline: 42.8 vs
36.0 dB at $t=1$), real-input validation metrics must not regress, and the
goal is `iter_prediction` $\ge$ `one_shot`.

**Outcome (2026-08-30):** implemented as the opt-in `training.rollout`
stage (decisions: nominal-step $t$, stop levels uniform over $\{1..T-1\}$,
on-the-fly EMA-weight rollouts — reasoning in the method doc §5) and run
on both baselines. The goal holds on both datasets: `iter_prediction`
40.5 vs `one_shot` 39.9 dB (BBBC038) and 35.9 vs 35.6 dB (MIIC), with
real-input metrics essentially unchanged. A **step-matched** control (each
baseline plainly continued for the same 10k extra steps, no rollout)
leaves `iter_prediction` at its baseline value, so the change is
attributable to the rollout mechanism, not to more gradient steps. (Step-
matched, not FLOP-matched: a rollout step costs ~3.8× a plain one.) One refinement to the success
criterion as originally stated: the A/B curves *cannot fully* merge,
because the deterministic sampler makes every pseudo-state a function of
the single frame $y_1$ — B is information-bounded by one measurement
while A consumes fifteen. The validation-set A−B gap at $t=1$ closed from
10.9 to 4.3 dB (BBBC038) and 3.8 to 3.0 dB (MIIC); the remainder is that
ceiling, not residual distribution mismatch. Full tables: report §8.

---

## Appendix: the diagnostic probe

`python tools/probe_burst_predictions.py` (defaults: BBBC038 checkpoint,
held-out source 12, 64 px center crop). Reference output at the 30k-step
baseline:

```
raw frame y1 vs clean:            17.31 dB
raw frame y1 vs another frame:    14.25 dB

A) network fed REAL m-frame averages (in-distribution, as in training)
    t   m   pred-vs-clean   pred-vs-a-real-fresh-frame
   15   1    36.15 dB       17.20 dB
   11   5    40.48 dB       17.23 dB
    7   9    41.77 dB       17.23 dB
    3  13    42.33 dB       17.24 dB
    1  15    42.82 dB       17.24 dB

B) network fed its OWN pseudo-averages (the iterative sampler trajectory)
    t   m   pred-vs-clean   state-vs-clean
   15   1    36.15 dB       23.15 dB
   11   5    36.18 dB       31.35 dB
    7   9    35.99 dB       33.87 dB
    3  13    35.98 dB       34.89 dB
    1  15    36.00 dB       35.18 dB

MSE(final prediction, actual fresh frame): 0.0191
single-frame noise variance (from its PSNR): 0.0186
```

Reading: **A**'s flat ~17.2 dB column = the prediction is far from every real
frame by exactly that frame's own noise — i.e. it is the clean estimate
(Q3). **A** improving down the rows = the posterior-mean family works across
levels. **B** flat at ~36 dB while **A** reaches 42.8 dB at the same $t$ =
the train/inference input gap (Q4). The final two lines = the loss-floor
signature (Q3).

*Caveat learned during the finetuning follow-up (Q6):* source 12 turned out
to be the one validation source with essentially **no headroom** — its
one-shot (36.1 dB) already saturates what a single frame supports there, so
its B column barely moves even after finetuning closes the dataset-wide gap
(validation-set mean B at $t=1$: 34.0 → 40.5 dB). Single-source probe runs
locate the mechanism, but judge gap-closing by validation-set means
(report §8), not by this one source.
