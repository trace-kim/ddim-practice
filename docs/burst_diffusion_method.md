# Burst-Averaging Diffusion — Concept and Mathematical Derivations

*Companion documents: [experiment report](burst_diffusion_report.md) ·
[user guide](burst_diffusion_guide.md) · implementation in
[`burst_diffusion/`](../burst_diffusion/).*

This document derives the method implemented in `burst_diffusion`: a
DDPM/DDIM-inspired denoiser in which the diffusion "noise" $\varepsilon$ is a
real noisy frame and the timestep is the number of burst frames averaged.
Every theorem here is enforced by a unit test in
`tests/test_burst_diffusion_*.py`, and every numeric claim was measured in the
reference experiment.

## 1. Setup and notation

A clean image $x_0 \in \mathbb{R}^d$ is observed through $N$ **burst frames**

$$y_j = x_0 + n_j, \qquad j = 1,\dots,N,$$

where the noise satisfies, conditionally on $x_0$:

- **(A1) unbiasedness:** $\mathbb{E}[n_j \mid x_0] = 0$;
- **(A2) independence:** $n_1,\dots,n_N$ are mutually independent given $x_0$;
- per-pixel variance $\sigma^2(x_0)$, possibly signal-dependent (Poisson shot
  noise: $y = \mathrm{Pois}(\lambda x_0)/\lambda$ has mean $x_0$ and variance
  $x_0/\lambda$). We write $\sigma^2 := \mathbb{E}[\sigma^2(x_0)]$ for its
  average scale.

On real equipment a "burst" is $N$ fast acquisitions of the same, pixel-aligned
area; synthetically, `noising_pipeline` draws the $n_j$ independently.
Section 6 examines what happens when (A1)–(A2) are only approximately true.

## 2. The frame-averaging forward process

For a subset $S \subseteq \{1..N\}$ with $|S| = m$, the partial average

$$\bar{y}_S \;=\; \frac{1}{m}\sum_{j\in S} y_j \;=\; x_0 + \frac{1}{m}\sum_{j\in S} n_j$$

has mean $x_0$ and per-pixel variance $\sigma^2/m$ — the **$\sqrt{N}$ law**
(+3 dB of PSNR per doubling of $m$), which the generated datasets reproduce
exactly (e.g. MIIC: 14.2 dB single frame $\to$ 17.2 / 20.2 / 23.2 / 26.2 dB at
$m = 2, 4, 8, 16$).

Index noise levels DDPM-style by $t \in \{1..T\}$, noisiest at $t = T$, with

$$m(t) \;=\; T + 1 - t, \qquad m(T) = 1,\quad m(1) = T,\quad m(0) = T+1,$$

and define the state $x_t = \bar{y}_S$ with $|S| = m(t)$, so
$\operatorname{Var}(x_t) = \sigma^2/m(t)$ — a hyperbolic variance schedule.
($m(0)$ appears only as an averaging weight in the final sampling step; the
network is never evaluated at $t=0$.) Training requires $N \ge T+1$
(Section 3).

**Contrast with DDPM.** DDPM's forward process
$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$
attenuates the signal and injects synthetic Gaussian noise; here the signal is
never attenuated and the "noise" is whatever the instrument produces. This
places the method in the cold-diffusion / variance-exploding family, with the
distinctive feature that the degradation is realized by *subsampling real
measurements* rather than by a simulator.

## 3. The training objective and its optima

Training samples a source, a level $t$, a uniformly random subset $S$ with
$|S| = m(t)$, and a target index $K$, then minimizes

$$\mathcal{L}(\theta) \;=\; \mathbb{E}\,\bigl\lVert \varepsilon_\theta(x_t, t) - y_K \bigr\rVert^2 .$$

The choice of $K$ is the load-bearing design decision. Recall the standard
$L^2$ fact: over all measurable $f$, $\mathbb{E}\lVert f(X) - Y\rVert^2$ is
minimized by $f^\ast(x) = \mathbb{E}[Y \mid X = x]$.

### Theorem 1 (fresh target ⇒ posterior-mean denoiser)

**If $K \notin S$** ("fresh": the target frame is excluded from the average),
the minimizer of $\mathcal{L}$ is

$$\varepsilon^\ast(\bar y, t) \;=\; \mathbb{E}\bigl[y_K \mid x_t = \bar y\bigr] \;=\; \mathbb{E}\bigl[x_0 \mid x_t = \bar y\bigr],$$

the minimum-MSE estimate of the **clean image** from an $m(t)$-frame average.

*Proof.* $\mathbb{E}[y_K \mid x_t] = \mathbb{E}[x_0 \mid x_t] + \mathbb{E}[n_K \mid x_t]$.
The state $x_t$ is a function of $\bigl(x_0, \{n_j\}_{j\in S}\bigr)$, and by
(A2) $n_K$ is independent of $\{n_j\}_{j\in S}$ given $x_0$, so by the tower
property and (A1)

$$\mathbb{E}[n_K \mid x_t] \;=\; \mathbb{E}\bigl[\,\mathbb{E}[n_K \mid x_0, \{n_j\}_{j\in S}]\;\big|\; x_t\bigr] \;=\; \mathbb{E}\bigl[\,\mathbb{E}[n_K \mid x_0]\;\big|\;x_t\bigr] \;=\; 0. \qquad\blacksquare$$

**Noise2Noise in one paragraph.** Noise2Noise (Lehtinen et al., 2018) is the
observation that an image-restoration network can be trained with *corrupted
targets* instead of clean ones. A training pair is two independent noisy
observations of the same scene, $(y^{\text{in}}, y^{\text{tgt}})$. Because the
$L^2$-optimal predictor is the conditional mean, and independent zero-mean
noise vanishes inside a conditional mean —
$\mathbb{E}[y^{\text{tgt}} \mid y^{\text{in}}] = \mathbb{E}[x_0 \mid y^{\text{in}}]$
— the network converges to the *same function* it would have learned from
clean targets; the noisy target is an unbiased, merely higher-variance,
stand-in. Gradient noise grows, the optimum does not move. Intuitively: the
network cannot learn to reproduce the target's noise, because that noise is
unpredictable from an independent observation — so the best it can do is
output the predictable part, the clean image. (The loss must match the noise
statistics: $L^2$ for zero-mean noise; a median-seeking $L^1$ for impulse
noise — which is exactly why salt-and-pepper breaks the MSE framework here,
§6.)

Theorem 1 is this argument, and the burst structure extends it in three ways:
(i) the **input side is a partial average**, so one burst yields training
pairs at $T$ different input-noise levels ($\sigma^2/m(t)$, $m = 1..T$)
instead of Noise2Noise's single fixed level; (ii) the network is
**$t$-conditioned**, one model spanning the whole schedule rather than one
model per level; (iii) a **diffusion-style sampler** (§4) is wrapped around
the learned family. The `fresh` requirement $K \notin S$ *is* Noise2Noise's
independence condition — input and target must not share noise realizations —
and Theorem 2 shows exactly what happens in the averaging setting when it is
violated. Practically: **training toward noisy targets converges to the
clean-image estimator, so no clean image is ever needed for training.**
Because the cleanest level averages $T$ frames and the fresh target must lie
outside that subset, $N \ge T+1$.

### Theorem 2 (included target ⇒ identity; the degenerate case)

**If $K$ is drawn uniformly from $S$** ("included"), the minimizer is the
identity map: $\varepsilon^\ast(\bar y, t) = \bar y$.

*Proof.* $K$ is independent of the frames given $S$, so

$$\mathbb{E}[y_K \mid x_t] \;=\; \frac{1}{m}\sum_{j\in S}\mathbb{E}[y_j \mid x_t] \;=\; \frac{1}{m}\,\mathbb{E}\Bigl[\sum_{j\in S} y_j \;\Big|\; x_t\Bigr] \;=\; \frac{1}{m}\, m\, x_t \;=\; x_t. \qquad\blacksquare$$

(The middle equality is linearity of conditional expectation;
$\sum_{j\in S} y_j = m\,x_t$ is measurable w.r.t. $x_t$. For a *fixed*
$k \in S$ the same conclusion follows from exchangeability of i.i.d. noise.)
The objective then carries no learning signal — the network converges to a
copy machine.

**Why DDPM does not suffer this.** In DDPM the predicted $\varepsilon$ *is*
inside $x_t$, but the mixture is weighted:
$\mathbb{E}[\varepsilon\mid x_t] = \bigl(x_t - \sqrt{\bar\alpha_t}\,\mathbb{E}[x_0\mid x_t]\bigr)/\sqrt{1-\bar\alpha_t}$,
which is non-trivial exactly because recovering it requires the learned prior
over $x_0$. An equal-weight average destroys that asymmetry. This is why
`target_mode: fresh` is the default and `included` exists only as a
documented-degenerate ablation (it warns at construction).

### Loss decomposition and the loss floor

For any $\varepsilon_\theta$, orthogonality of the conditional expectation
gives

$$\mathbb{E}\lVert \varepsilon_\theta - y_K\rVert^2 \;=\; \underbrace{\mathbb{E}\lVert \varepsilon_\theta - \varepsilon^\ast\rVert^2}_{\text{estimation error} \,\to\, 0} \;+\; \underbrace{\mathbb{E}\lVert \varepsilon^\ast - y_K\rVert^2}_{\text{irreducible}} .$$

With a fresh target, $y_K - \varepsilon^\ast = (x_0 - \mathbb{E}[x_0\mid x_t]) + n_K$,
and the cross term vanishes ($\mathbb{E}[n_K \mid x_0, x_t] = 0$), so per pixel

$$\text{irreducible} \;=\; \mathbb{E}\bigl[\operatorname{Var}(x_0 \mid x_t)\bigr] \;+\; \sigma^2 .$$

The $\sigma^2$ term is **independent of $t$** (the target is always one fresh
frame), which is why the per-level TensorBoard scalars `train/loss_by_t/*`
isolate estimation quality. The pipeline trains in model space
$u = 2v - 1 \in [-1,1]$, a linear map that multiplies variances by **4**:

$$\boxed{\;\text{loss floor} \;\approx\; 4\cdot 10^{-\mathrm{PSNR}_{\text{single}}/10}\;}$$

**Measured:** MIIC single-frame PSNR 14.16 dB $\Rightarrow$ predicted floor
$4 \cdot 10^{-1.416} = 0.154$; final training loss $\approx 0.150$. BBBC038:
predicted $0.075$; measured $\approx 0.09$–$0.14$ (richer content, larger
residual estimation error). **A plateauing loss is therefore correct behavior**;
progress is read from validation PSNR of the prediction against clean.

*Antithetic levels.* Each batch pairs $t$ with $T+1-t$ (adapted from the
legacy DDIM runner's trick), guaranteeing balanced coverage of low- and
high-noise levels per gradient step — a variance-reduction device, not a
change of objective.

## 4. Sampling: the cumulative-average recursion

The exact running-average identity for a real $(m{+}1)$-th frame is
$\bar y_{m+1} = \bar y_m + (y - \bar y_m)/(m+1)$. At inference no further real
frames exist, so the network prediction stands in for them. From level $t$
(with $m = m(t)$) to any lower level $t' < t$ (with $m' = m(t')$):

$$\boxed{\;x_{t'} \;=\; \frac{m(t)\,x_t \;+\; \bigl(m(t') - m(t)\bigr)\,\hat\varepsilon}{m(t')}\;},\qquad \hat\varepsilon = \varepsilon_\theta(x_t, t),$$

i.e. the prediction is folded in as if it were $m(t')-m(t)$ additional
acquisitions. The unit step ($t' = t-1$) is
$x \leftarrow x + (\hat\varepsilon - x)/(m(t)+1)$.

### Proposition (exact composition)

For a fixed $\hat\varepsilon$, chaining $t \to t' \to t''$ equals the direct
jump $t \to t''$:

$$\frac{m' \cdot \frac{m x + (m'-m)\hat\varepsilon}{m'} + (m''-m')\hat\varepsilon}{m''} \;=\; \frac{m x + (m''-m)\hat\varepsilon}{m''}. \qquad\blacksquare$$

Accelerated (DDIM-style) sampling over a sub-schedule is therefore **not** an
approximation of the update — only of the sequence of predictions. (Enforced
by test: any decreasing path with shared $\hat\varepsilon$ matches the direct
jump to $10^{-12}$.)

### Closed form of the average output

Starting from the measurement $x_T = y_1$ and taking unit steps with
predictions $\hat\varepsilon_T, \dots, \hat\varepsilon_1$ (where
$\hat\varepsilon_t$ is produced at level $t$), induction gives

$$x_0^{\text{avg}} \;=\; \frac{y_1 + \sum_{t=1}^{T}\hat\varepsilon_t}{T+1}.$$

*Proof sketch:* after the step at level $t$ the state is
$\bigl(y_1 + \sum_{s\ge t}\hat\varepsilon_s\bigr)/(T+2-t)$; the base case
$t=T$ gives $(y_1+\hat\varepsilon_T)/2$ and the inductive step is the unit
update. $\blacksquare$

So the sampler literally simulates a $(T{+}1)$-frame acquisition in which the
network supplies $T$ of the frames; the step sizes $1/2, 1/3, \dots, 1/(T{+}1)$
give a coarse-to-fine refinement. Two outputs are returned:

- $x_0^{\text{avg}}$ — the running average above (spec-faithful);
- $x_0^{\text{pred}} = \hat\varepsilon_1 \approx \mathbb{E}[x_0 \mid x_1]$ —
  the final prediction, theoretically the cleanest estimate.

### One-shot inference (the "single forward pass")

Throughout the report, **one-shot** means evaluating the trained network
exactly once on the raw measurement at the noisiest level, with no iteration:

$$x_0^{\text{one-shot}} \;=\; \varepsilon_\theta(y_1,\, T).$$

That a *noise predictor* used once yields a *denoised image* is a direct
consequence of Theorem 1: the network was trained to predict a fresh frame,
whose MSE-optimal value is $\mathbb{E}[x_0 \mid y_1]$ — the minimum-MSE clean
estimate from a single frame. In other words, one-shot is a $t$-conditioned
Noise2Noise denoiser applied at $t = T$. This differs sharply from DDPM, where
a single high-$t$ evaluation gives a poor $x_0$ estimate (recovering
$\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon)/\sqrt{\bar\alpha_t}$
divides by a tiny $\sqrt{\bar\alpha_t}$, amplifying error) and iteration is
essential; here the signal is never attenuated, so the first prediction is
already the full-strength estimate.

One-shot is not a separate mechanism: it is the iterative sampler's *first*
network call. Equivalently, it is the `prediction` output of the sampler run
with the length-one schedule $[T]$ — which is exactly how `evaluate` computes
it. Iteration differs only in what happens afterwards: the prediction is
folded into the running average (weight $1/2$, then $1/3$, ...) and the
network is re-queried at lower $t$. One-shot costs one network evaluation
instead of $T$; whether the extra $T-1$ evaluations help is an empirical
question — in the reference experiment they did not (§5).

### Retained-noise bound for the average output

If the predictions were perfect ($\hat\varepsilon_t \equiv x_0$),

$$x_0^{\text{avg}} = x_0 + \frac{n_1}{T+1} \;\Rightarrow\; \text{PSNR} = \text{PSNR}_{\text{input}} + 20\log_{10}(T+1),$$

an upper bound of $16.8 + 24.1 = 40.9$ dB for BBBC038 at $T=15$. The measured
34.7 dB shows prediction error, not retained input noise, is the binding
constraint — consistent with the next section.

## 5. The train/inference gap (and why iteration may not help)

At **training**, the level-$t$ input is an average of $m$ *real* frames:
real-noise variance $\sigma^2/m$. At **sampling**, the state at the same
nominal level is, by the closed form,

$$x = x_0 + \frac{n_1}{m} + \frac{1}{m}\sum_s e_s, \qquad e_s := \hat\varepsilon_s - x_0,$$

whose real-noise part has variance $\sigma^2/m^2$ — a factor $m$ *less* than
the conditioning label $t$ promises — while the prediction errors $e_s$ are
**deterministic functions of $y_1$** (the sampler is deterministic), hence
correlated across steps and *not* reduced by the averaging the way independent
noise would be. Two predicted consequences:

1. sampler states are statistically unlike training states at the same $t$;
2. iterating cannot be assumed to beat the direct estimate
   $\varepsilon_\theta(y_1, T)$.

**Empirically confirmed:** one-shot 40.2 dB vs iterative-prediction 34.0 dB
(BBBC038); 35.4 vs 35.3 dB (MIIC). Mitigations, in increasing order of effort:
prefer the `prediction` output; use fewer sampling steps (less compounding);
**self-rollout finetuning** — continue training on the sampler's own
intermediate states so the network sees its inference-time input distribution
(the natural next experiment).

## 6. When the assumptions bend: bias and correlation

### Clipping (synthetic data)

`noising_pipeline` clips to $[0,1]$ after each noise application:
$y = \mathrm{clip}(x_0 + n)$. Clipped noise violates (A1): upper truncation
dims bright pixels, lower truncation brightens dark ones, and **no amount of
averaging removes the bias** — $\bar y_N \to \mathbb{E}[y\,|\,x_0] \ne x_0$.

*Worked example (Poisson).* $y = \min(\mathrm{Pois}(\lambda x)/\lambda,\, 1)$
with $\lambda = 10$ at the brightest pre-scaled pixel $x = 0.85$: with the
normal approximation ($\mu = 8.5$, $s = \sqrt{8.5} \approx 2.92$,
$\delta = (10-8.5)/s \approx 0.51$),

$$\text{bias}(x) = -\tfrac{1}{\lambda}\mathbb{E}\bigl[(P-\lambda)^+\bigr] \approx -\tfrac{s}{\lambda}\bigl[\varphi(\delta) - \delta(1-\Phi(\delta))\bigr] \approx -0.057,$$

so the very brightest structures can be dimmed by up to $\sim$0.06 even with
the default margin. The **mitigations** implemented: (i) `generate` pre-scales
clean sources into $[\text{margin}, 1-\text{margin}]$ (default 0.15), keeping
most pixels far from the rails (Gaussian rule of thumb:
$\text{std} \le \text{margin}/2$, i.e. $\ge 2\sigma$ of headroom); (ii) the
bias is **measured**, not assumed — `stats.json` reports avg-of-N PSNR and
image-mean bias (measured: $\le 0.0024$ on both reference datasets) and warns
out of band. A fairness property worth noting: by Theorem 1 the network's
target converges to the *same* clipped-mean limit
$\mathbb{E}[y \mid x_0]$ as the averaging baseline, so model-vs-averaging
comparisons are unaffected; only absolute PSNR against the pristine clean
absorbs the bias.

### Salt-and-pepper noise

Replacing a fraction $a$ of pixels with salt/pepper values gives *exactly*
$\mathbb{E}[y \mid x_0] = (1-a)\,x_0 + a\,s$ ($s$ = salt ratio) — a contraction
toward $s$ that breaks (A1) by construction. The MSE optimum then tracks this
biased limit (impulse noise wants robust losses, not $L^2$). The generator
supports it but emits an explicit warning; quantitative runs use
Poisson/Gaussian (physically: shot + read noise).

### Real-equipment caveats

- **Fixed-pattern noise / detector artifacts** are *correlated across frames*,
  violating (A2): the model will faithfully learn them as signal. Flat-field /
  offset correction should precede ingestion.
- **Registration:** frames must be pixel-aligned; drift blurs the average and
  breaks the premise. Correct drift before building the manifest.
- **Identical distribution:** dose changes, charging, or beam damage across
  the burst make later frames differ systematically from earlier ones —
  another (A1) violation to keep small.

## 7. Relation to prior work

- **DDPM** (Ho et al., 2020): the training loop is Algorithm 1 with
  $(\sqrt{\bar\alpha}, \varepsilon)$ replaced by (partial average, fresh
  frame); antithetic $t$-sampling and the $\varepsilon$-prediction
  parameterization are inherited.
- **DDIM** (Song et al., 2021): deterministic sampling over a decreasing
  sub-schedule; here the accelerated update is *exact* by the composition
  proposition, so only the prediction sequence is approximated.
- **Noise2Noise** (Lehtinen et al., 2018): Theorem 1 is its
  conditional-expectation argument, generalized to a schedule of averaging
  levels with $t$-conditioning.
- **Cold Diffusion** (Bansal et al., 2022): diffusion with arbitrary,
  signal-preserving degradations; this method's degradation is distinctive in
  being *subsampled real measurements*, which is what makes ground-truth-free
  training possible.
- **Classical frame integration**: the baseline the method is designed to
  replace — and, per the report, beat from a single frame.

## 8. Assumptions, and how the pipeline checks them

| Assumption | Breaks when | Checked / enforced by |
|---|---|---|
| (A1) zero-mean noise | clipping near 0/1, salt-and-pepper, dose drift | `generate` margin pre-scale; `stats.json` bias + avg-of-N PSNR + warnings |
| (A2) independent frames | fixed-pattern noise, correlated readout | user-side correction; document §6 |
| frames pixel-aligned | stage drift | user-side registration; document §6 |
| $N \ge T+1$ | too few replicas | `BatchFactory` raises with the exact requirement |
| noise level learnable | too clean (>18 dB) or extreme (<10 dB) | `stats.json` single-frame PSNR band warning |
| loss floor $\approx 4\sigma^2$ | (diagnostic, not assumption) | predicted vs measured: 0.154 vs 0.150 (MIIC) |
