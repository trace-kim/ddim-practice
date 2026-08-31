# Burst Residual Diffusion — learning the increments between averaging levels

*Companion documents: [burst-averaging method](burst_diffusion_method.md) ·
[experiment report](burst_diffusion_report.md) ·
[user guide](burst_diffusion_guide.md).*

This document develops an alternative to the parameterization in
[`burst_diffusion_method.md`](burst_diffusion_method.md). There, the network
predicts a **frame** — the diffusion $\varepsilon$ is a real noisy
acquisition. Here it predicts the **increment between consecutive averaging
levels**: average $m$ frames, average $m+1$ frames, and learn the difference.

Nothing in this document is implemented yet. It is a derivation and a design,
written to the same standard as its companion: every claim is either proved,
or measured, or explicitly labelled a conjecture. The numeric checks come from
a scalar conjugate-Gaussian model where every quantity has a closed form
(3M-sample Monte Carlo, agreement to $\pm 2\cdot10^{-3}$ unless stated).

## 0. The bottom line, first

Two facts should govern how this is read, because one of them is deflationary
and hiding it would waste the reader's time.

**The MMSE optimum does not move.** Theorem R1 shows the optimal residual
predictor is the optimal frame predictor times a known constant. As
estimators of $x_0$ the two parameterizations are *the same function*, related
by an invertible affine map. **Switching to residuals buys no denoising
accuracy by itself**, and any claim that it does is wrong. This is the
$\varepsilon$- vs $x_0$- vs $v$-prediction situation in DDPM: same optimum,
different conditioning.

**What the residual view does buy** is everything downstream of it, and that
turns out to be a lot:

| | payoff | status |
|---|---|---|
| §3 | the naive residual loss silently reweights levels by $1/(m+1)^2$ — a 64× downweight of the cleanest level at $T=15$ — with an exact preconditioning fix | derived, measured |
| §4 | the burst forward process is **exactly a Brownian motion** in the time variable $\tau = 1/m$, so this is a variance-exploding diffusion and the whole score-based toolkit applies | proved, measured |
| §5 | the existing sampler is the reverse-SDE **drift with the noise deleted** — it is *not* the probability-flow ODE, which needs a factor $\tfrac12$. Full step gives the posterior *mean*; half step gives a sharp sample | proved, measured |
| §5 | **posterior sampling**: draw plausible clean images rather than one blurry mean — impossible in the current MMSE-only framework | derived |
| §6 | **calibrated per-pixel error bars** from noisy data only, with no ground truth | derived |
| §5 | the sampling schedule decouples from integer frame counts | derived |
| §2 | a new degeneracy theorem (R2) that kills the most natural-sounding variant of this idea | proved, measured |

Theorem R2 deserves the early warning. "Learn the residual between the
successive images" has two readings, and one of them is fatal: differencing
the successive *averages* works (R1), while differencing the successive
*denoised estimates* collapses to the zero array (R2). The distinction is not
obvious from the phrasing.

## 1. The residual and its exact statistics

Notation follows the companion document: $y_j = x_0 + n_j$ for $j = 1..N$,
with (A1) $\mathbb{E}[n_j \mid x_0] = 0$, (A2) mutual independence given
$x_0$, and per-pixel variance $\sigma^2$. Write $\bar y_m$ for the average of
the first $m$ frames and $\bar n_m := \bar y_m - x_0$ for its noise. The
level index of the companion doc maps in as $m = m(t) = T + 1 - t$.

Define the **step residual**

$$\Delta_m \;:=\; \bar y_{m+1} - \bar y_m .$$

The exact running-average identity $\bar y_{m+1} = \bar y_m + (y_{m+1} - \bar y_m)/(m+1)$
gives the form that will be used throughout:

$$\boxed{\;\Delta_m \;=\; \frac{y_{m+1} - \bar y_m}{m+1} \;=\; \frac{n_{m+1} - \bar n_m}{m+1}\;}$$

The second equality is the first structural fact: **the residual is
signal-free**. Both $x_0$ terms cancel, so unlike $\varepsilon = y_K$ — which
carries the clean image inside it, the fact that drove Theorems 1 and 2 —
$\Delta_m$ is pure noise, with $\mathbb{E}[\Delta_m] = 0$ unconditionally.

Its scale follows from (A2):

$$\operatorname{Var}(\Delta_m) \;=\; \frac{1}{(m+1)^2}\Bigl(\sigma^2 + \frac{\sigma^2}{m}\Bigr) \;=\; \frac{\sigma^2}{m(m+1)},$$

consistent with the telescoped variances,
$\operatorname{Var}(\bar y_m) - \operatorname{Var}(\bar y_{m+1}) = \sigma^2/m - \sigma^2/(m+1) = \sigma^2/(m(m+1))$
(measured: $m = 1, 2, 4, 8, 16$ give $0.17999, 0.05998, 0.01800, 0.00500, 0.00132$
against predictions $0.18, 0.06, 0.018, 0.005, 0.001324$).
So the residual scale decays as $\sigma_\Delta(m) = \sigma/\sqrt{m(m+1)} \sim \sigma/m$.

**The telescoping series.** Summing increments from the seed frame,

$$\bar y_M \;=\; y_1 + \sum_{m=1}^{M-1}\Delta_m \qquad\text{and, since } \bar y_M \to x_0, \qquad \boxed{\;x_0 \;=\; y_1 + \sum_{m=1}^{\infty}\Delta_m\;}$$

This is the whole method in one line: **the clean image is the seed frame
plus the sum of all future residuals.** Denoising becomes summing a series
whose terms are not observed, and the network's job is to supply them. It
reframes inference from "iterate a fixed point" to "sum a series," which is
what makes §5's convergence questions answerable.

**A remark on state.** Carrying only the running average as the sampler state
is not an approximation: $\sum_{j\le m} y_j$ is a *sufficient statistic* for
$x_0$ under both Gaussian and Poisson noise (both are exponential families in
which the sum is sufficient). Nothing is lost by discarding the individual
frames.

## 2. What the optimal residual predictor is

### Theorem R1 (raw increment ⇒ scaled correction)

Train a network on $\mathcal{L} = \mathbb{E}\lVert \delta_\theta(\bar y_m, m) - \Delta_m\rVert^2$
with $\Delta_m$ built from a frame $y_{m+1}$ *outside* the averaged subset.
The minimizer is

$$\boxed{\;\delta^\ast(x, m) \;=\; \mathbb{E}[\Delta_m \mid \bar y_m = x] \;=\; \frac{\mathbb{E}[x_0 \mid \bar y_m = x] \;-\; x}{m+1}\;}$$

*Proof.* By the $L^2$ fact, the minimizer is the conditional mean. Then

$$\mathbb{E}[\Delta_m \mid \bar y_m] \;=\; \frac{\mathbb{E}[y_{m+1}\mid \bar y_m] - \bar y_m}{m+1} \;=\; \frac{\mathbb{E}[x_0 \mid \bar y_m] - \bar y_m}{m+1},$$

where the second equality is Theorem 1 of the companion document: $y_{m+1}$
is *fresh* (excluded from the subset that formed $\bar y_m$), so
$\mathbb{E}[n_{m+1}\mid\bar y_m] = 0$ and $\mathbb{E}[y_{m+1}\mid \bar y_m] = \mathbb{E}[x_0\mid\bar y_m]$. $\blacksquare$

**The reference toy**, used for every measurement in this document: scalar
$x_0\sim\mathcal{N}(\mu, \tau_p^2)$ and $n_j \sim \mathcal{N}(0,\sigma^2)$
i.i.d., with $\mu = 0.35$, prior s.d. $\tau_p = 0.40$, $\sigma = 0.60$. Write
$w_m := \tau_p^2/(\tau_p^2 + \sigma^2/m)$ for the posterior shrinkage factor.
Everything is jointly Gaussian, so each conditional mean is affine in $x$ and
the check is on its slope:

| $m$ | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| slope, Monte Carlo | $-0.34613$ | $-0.17646$ | $-0.07220$ | $-0.02431$ | $-0.00725$ |
| slope, Theorem R1 | $-0.34615$ | $-0.17647$ | $-0.07200$ | $-0.02439$ | $-0.00725$ |

Name the two derived quantities, used from here on:

$$D^\ast(x,m) := \mathbb{E}[x_0\mid \bar y_m = x] \quad\text{(the denoiser)},\qquad c^\ast(x,m) := D^\ast(x,m) - x \quad\text{(the correction)},$$

so that $\delta^\ast = c^\ast/(m+1)$.

**Corollary (exact reparameterization).** The three targets

$$\underbrace{y_{m+1}}_{\varepsilon\text{-form}}, \qquad \underbrace{y_{m+1} - \bar y_m}_{\text{correction form}}, \qquad \underbrace{\frac{y_{m+1}-\bar y_m}{m+1}}_{\text{residual form}}$$

differ by adding a function of the input and multiplying by a constant known
from $m$. Their optima are therefore in bijection:

$$\hat\varepsilon \;=\; x + (m+1)\,\hat\delta, \qquad \hat\delta \;=\; \frac{\hat\varepsilon - x}{m+1},$$

and a trained network in one form converts to the other **exactly, with no
retraining**. This is the deflationary half of §0: at the level of "what
function is learned," the residual method is the existing method.

What is *not* invariant: the implicit weighting across levels (§3), the
numerical conditioning of the regression, and — the substantive part — the
sampler that the parameterization suggests (§5).

**Why the parameterization is still worth having.** Writing
$D_\theta(x,m) = x + (m+1)\,\delta_\theta(x,m)$ builds the identity map into
the architecture as a skip connection, so the network only ever learns the
*correction*. This is the DnCNN residual-learning argument, and it bites
hardest exactly where this method is weakest: at large $m$ the input is
already clean, $c^\ast$ is tiny, and asking a network to emit the whole image
means computing a large output that must nearly cancel its own input.

### Theorem R2 (estimate increment ⇒ zero; the new degenerate case)

There is a second, very natural reading of "learn the residual between the
two images," and it does not work. Let $\mathcal{F}_m := \sigma(y_1,\dots,y_m)$
and define the **posterior-mean process** and its increment

$$Z_m := \mathbb{E}[x_0 \mid \mathcal{F}_m], \qquad \zeta_m := Z_{m+1} - Z_m .$$

$Z_m$ is the sequence of *best current estimates* — precisely what a trained
denoiser outputs at each level — so "difference the successive denoised
images and learn that" means regressing $\zeta_m$. Then:

**The minimizer of $\mathbb{E}\lVert f(\bar y_m, m) - \zeta_m\rVert^2$ is
$f^\ast \equiv 0$.**

*Proof.* $Z_m$ is a Doob martingale: by the tower property,
$\mathbb{E}[Z_{m+1}\mid\mathcal{F}_m] = \mathbb{E}\bigl[\mathbb{E}[x_0\mid\mathcal{F}_{m+1}]\;\big|\;\mathcal{F}_m\bigr] = \mathbb{E}[x_0\mid\mathcal{F}_m] = Z_m$.
Hence $\mathbb{E}[\zeta_m \mid \mathcal{F}_m] = 0$, and since
$\sigma(\bar y_m)\subseteq\mathcal{F}_m$, one more application of the tower
property gives $\mathbb{E}[\zeta_m \mid \bar y_m] = 0$. $\blacksquare$

Measured: regressing $\zeta_m$ on $\bar y_m$ returns slopes
$+1.1\cdot10^{-5}$, $+6\cdot10^{-6}$, $-1.4\cdot10^{-4}$, $+6.4\cdot10^{-5}$
at $m = 1,2,4,8$ — zero to Monte-Carlo precision.

**Why it fails, in words.** A martingale's increments are by definition
unpredictable from the present. $Z_{m+1}$ differs from $Z_m$ only because
frame $m+1$ arrived, and that frame's noise is independent of everything
observed so far; the *update* it causes is unforecastable. There is no
signal to regress. This is the exact mirror of the companion document's
Theorem 2 — that objective collapsed to the identity because the target's
noise was fully inside the input; this one collapses to zero because the
target is fully *outside* it.

The three-way pattern, which is the cleanest summary of both documents:

| target | its relation to the input | MSE optimum | usable? |
|---|---|---|---|
| fresh frame $y_K$, $K\notin S$ | independent noise, contains $x_0$ | $\mathbb{E}[x_0\mid x_t]$ | yes — the denoiser |
| included frame $y_K$, $K \in S$ | noise leaked into the input | $x_t$ (identity) | no — copy machine |
| raw increment $\Delta_m$ | signal-free, partially inside | $c^\ast/(m+1)$ | yes — R1 |
| estimate increment $\zeta_m$ | martingale difference | $0$ | no — zero array |

**Practical consequence.** $\zeta$ is not useless, it is just not *regressable*.
It can only be **sampled**, and §5's stochastic sampler is exactly the machine
that samples it. Its second moment is also perfectly estimable, which is what
§6 exploits. The rule is: predict the raw increment, sample the estimate
increment, never regress the estimate increment.

## 3. Preconditioning: the implicit reweighting, and its fix

Because the three parameterizations of the Corollary share an optimum but not
a scale, their loss floors differ — and the loss floor at each level is
exactly the weight that level carries in the total gradient.

Two variances are needed and must not be confused, since §6 estimates the
first and the loss floors involve the second:

$$V_m(x) := \operatorname{Var}\bigl(x_0 \mid \bar y_m = x\bigr) \quad\text{(pointwise, per pixel)}, \qquad \bar V_m := \mathbb{E}\bigl[V_m(\bar y_m)\bigr] \quad\text{(its average)}.$$

Reusing the companion document's decomposition, the irreducible loss per pixel
is:

| target | prediction | loss floor at level $m$ | implicit level weight |
|---|---|---|---|
| $y_{m+1}$ | $\varepsilon_\theta$ | $\bar V_m + \sigma^2$ | $1$ |
| $y_{m+1} - \bar y_m$ | $c_\theta$ | $\bar V_m + \sigma^2$ | $1$ |
| $\Delta_m$ | $\delta_\theta$ | $\dfrac{\bar V_m + \sigma^2}{(m+1)^2}$ | $\dfrac{1}{(m+1)^2}$ |

(The first two agree because they differ by a quantity that is a function of
the input, which cancels from prediction and target alike.) Measured floors,
same toy:

| $m$ | 1 | 2 | 4 | 8 | 15 |
|---|---|---|---|---|---|
| $\varepsilon$ / correction | $0.4708$ | $0.4447$ | $0.4176$ | $0.3951$ | $0.3809$ |
| raw residual | $0.1177$ | $0.0494$ | $0.0167$ | $0.0049$ | $0.0015$ |
| relative weight | $0.250$ | $0.111$ | $0.040$ | $0.0124$ | $0.0039$ |

**The naive residual objective is badly miscalibrated, and in the wrong
direction.** At $T = 15$ the cleanest level ($m = 15$) receives $0.0156\times$
the gradient weight of the noisiest ($m=1$) — a 64-fold downweight of exactly
the levels that do the fine-detail work, and the levels the report's §5 gap
analysis identifies as decisive. Trained naively, a residual model would be
strictly worse than the existing one despite sharing its optimum. This is a
real trap, and it is invisible unless the floors are worked out.

**The fix: unit-variance targets.** Normalize by the known residual scale
$\sigma_\Delta(m) = \sigma/\sqrt{m(m+1)}$. Train $F_\theta$ against

$$\tilde\Delta_m \;:=\; \frac{\Delta_m}{\sigma_\Delta(m)} \;=\; \frac{1}{\sigma}\sqrt{\frac{m}{m+1}}\;\bigl(y_{m+1} - \bar y_m\bigr), \qquad \operatorname{Var}(\tilde\Delta_m) = 1 \;\;\forall m,$$

and reconstruct

$$\hat\delta \;=\; \frac{\sigma\,F_\theta}{\sqrt{m(m+1)}}, \qquad \hat x_0 \;=\; D_\theta(x,m) \;=\; x + \sigma\sqrt{\frac{m+1}{m}}\;F_\theta(x,m).$$

The normalized floor is $\frac{m}{m+1}\cdot\frac{\bar V_m+\sigma^2}{\sigma^2}$,
measured as $0.654, 0.824, 0.928, 0.976, 0.992$ for $m = 1,2,4,8,15$ — level
weights now within a factor of $1.5$ instead of $64$, and monotone toward $1$.
This is the same reasoning as EDM's preconditioning (Karras et al., 2022),
specialized to the burst schedule.

**Estimating $\sigma$ without ground truth.** The normalization needs
$\sigma$, which is *free* here — the burst measures it directly:

$$\hat\sigma^2 \;=\; \frac{1}{N-1}\sum_{j=1}^{N}\bigl(y_j - \bar y_N\bigr)^2,$$

computed per pixel (Poisson noise is signal-dependent, so a per-pixel or
per-intensity-bin estimate is the honest one) and pooled to a scalar scale
where a scalar is wanted. `stats.json` already reports the closely related
quantities, so this costs nothing new. Note this makes $\sigma$ a *dataset
constant*, not a learned parameter — it must be stored in the checkpoint,
because sampling needs it.

## 4. The hidden Brownian motion

This is where the residual view stops being bookkeeping. Adopt the
**reciprocal time**

$$\tau \;:=\; \frac{1}{m}, \qquad \tau = 1 \text{ at the raw frame},\quad \tau \to 0 \text{ as } m\to\infty .$$

Then $\operatorname{Var}(\bar y_m) = \sigma^2/m = \sigma^2\tau$ — variance
*linear in $\tau$*, the signature of Brownian motion.

### Theorem R3 (the forward process is a Brownian motion in $\tau = 1/m$)

For any $m \le m'$,

$$\operatorname{Cov}\bigl(\bar n_m,\, \bar n_{m'}\bigr) \;=\; \frac{1}{mm'}\operatorname{Cov}\Bigl(\sum_{j\le m} n_j,\; \sum_{k \le m'} n_k\Bigr) \;=\; \frac{m\,\sigma^2}{mm'} \;=\; \frac{\sigma^2}{m'} \;=\; \sigma^2\min(\tau,\tau'),$$

which is exactly the covariance kernel of a Brownian motion of diffusion
coefficient $\sigma^2$ evaluated at times $\tau, \tau'$. Consequently
increments over disjoint $\tau$-intervals are uncorrelated, and the
$\tau$-increment matches the variance:
$\tau_m - \tau_{m+1} = \frac{1}{m} - \frac{1}{m+1} = \frac{1}{m(m+1)}$, so
$\operatorname{Var}(\Delta_m) = \sigma^2\,(\tau_m - \tau_{m+1})$. $\blacksquare$

Measured: $\operatorname{Cov}(\bar n_m, \bar n_{m'})$ for $(m,m') =
(1,4), (2,8), (3,16), (5,5)$ gives $0.0900, 0.0450, 0.0225, 0.0720$ against
predictions $0.0900, 0.0450, 0.0225, 0.0720$; cross-covariances of increments
over disjoint intervals are $\le 5\cdot10^{-5}$.

**Scope of the claim, stated precisely.** The covariance identity needs only
(A1), (A2) and a common variance — it holds *exactly* for Poisson, Gaussian,
or any i.i.d. noise. Gaussianity of the increments (hence "is a Brownian
motion" rather than "has Brownian second-order structure") additionally needs
Gaussian $n_j$; for Poisson it is the CLT approximation, good at moderate
$m$ and at moderate dose, and poor in the photon-starved limit.

So: reading the sequence of burst averages from clean to noisy is a
**variance-exploding diffusion** $\mathrm{d}x = \sigma\,\mathrm{d}B_\tau$
started at $x_0$, and $\Delta_m$ are precisely its (negated) Brownian
increments. The burst schedule is not merely *analogous* to a diffusion; it
is one, with the unusual property that the Brownian path is realized by
physical measurement rather than simulated.

### Theorem R4 (the residual target is a score-matching target)

For Gaussian noise, $\bar y_m = x_0 + \sqrt{\sigma^2\tau}\,\epsilon$, so
Tweedie's formula applies exactly:

$$\mathbb{E}[x_0\mid \bar y_m = x] \;=\; x + \sigma^2\tau\,\nabla_x\log p_m(x), \qquad\text{i.e.}\qquad c^\ast(x,m) = \frac{\sigma^2}{m}\nabla_x\log p_m(x),$$

where $p_m$ is the marginal law of $\bar y_m$. Substituting into Theorem R1,

$$\boxed{\;\mathbb{E}[\Delta_m \mid \bar y_m = x] \;=\; \frac{\sigma^2}{m(m+1)}\,\nabla_x\log p_m(x) \;=\; \operatorname{Var}(\Delta_m)\cdot\nabla_x\log p_m(x)\;}$$

**The optimal residual is its own variance times the score.** Measured:
slopes $-0.34613, -0.17646, -0.07220, -0.02431$ against the score-form
predictions $-0.34615, -0.17647, -0.07200, -0.02439$ at $m = 1,2,4,8$.

Two consequences. First, training on real burst residuals *is* denoising
score matching — the method estimates $\nabla\log p_\tau$ of the real
measurement distribution, using real noise instead of simulated noise, which
is a genuinely unusual position to be in. Second, everything the score-based
literature provides — probability-flow ODEs, predictor–corrector samplers,
exact likelihoods, guidance — becomes formally available, which §5 uses.

## 5. The sampler, rewritten as an SDE solver

### The schedule and its step sizes

Let the sampling schedule be any increasing sequence of frame counts
$1 = m_0 < m_1 < \dots < m_K$ (equivalently, decreasing $\tau_k = 1/m_k$).
The exact averaging identity for a jump is

$$\bar y_{m_{k+1}} \;=\; \frac{m_k\,\bar y_{m_k} + (m_{k+1}-m_k)\,\bar y^{\text{new}}}{m_{k+1}}, \qquad \mathbb{E}\bigl[\bar y^{\text{new}} \mid \bar y_{m_k}\bigr] = D^\ast(\bar y_{m_k}, m_k),$$

so the optimal jump is

$$\boxed{\;x_{k+1} \;=\; x_k + \eta_k\,c^\ast(x_k, m_k) \;=\; (1-\eta_k)\,x_k + \eta_k\,\hat x_0, \qquad \eta_k := 1 - \frac{m_k}{m_{k+1}}\;}$$

The sampler is an **exponential moving average toward the running denoised
estimate**, with the schedule supplying the mixing rates $\eta_k \in [0,1)$.
This is algebraically the same update as `sample_step` in
`burst_diffusion/schedule.py`, rewritten so the step size is explicit; it
inherits the exact-composition property proved there. Unit steps give
$\eta = 1/(m+1)$, and the classic ladder $1/2, 1/3, \dots, 1/(T+1)$ is
recovered.

### Proposition R5 (the existing sampler is the drift, not the flow)

Write the reverse-time SDE for the VE process of Theorem R3 (Anderson's
formula, $f = 0$, $g = \sigma$), integrated with $\tau$ decreasing:

$$\mathrm{d}x \;=\; -\sigma^2\nabla_x\log p_\tau(x)\,\mathrm{d}\tau \;+\; \sigma\,\mathrm{d}\bar B_\tau, \qquad\text{probability-flow ODE:}\qquad \frac{\mathrm{d}x}{\mathrm{d}\tau} = -\tfrac{1}{2}\sigma^2\nabla_x\log p_\tau(x).$$

Taking one Euler–Maruyama step over $\lvert\mathrm{d}\tau\rvert = \frac{1}{m(m+1)}$
gives drift $\frac{\sigma^2}{m(m+1)}\nabla\log p$, which by Theorem R4 is
*exactly* $\mathbb{E}[\Delta_m\mid \bar y_m]$. Therefore:

**The unit-step mean-residual recursion follows the reverse-SDE drift with
the diffusion term deleted. It is not the probability-flow ODE — that
requires half the step.**

This is not a pedantic distinction; the two limits are different estimators,
and both are correct for different objectives. In the conjugate-Gaussian
model both integrals are closed-form: with drift factor $\kappa$, the map
from $\tau = 1$ to $\tau = 0$ scales $(x - \mu)$ by
$\bigl(\tau_p^2/(\tau_p^2+\sigma^2)\bigr)^{\kappa}$, giving

- $\kappa = \tfrac12$: multiplier $\tau_p/\sqrt{\tau_p^2+\sigma^2}$, mapping
  $p_{\tau=1}$ onto $p_{\tau=0}$ — **marginals preserved**;
- $\kappa = 1$: multiplier $\tau_p^2/(\tau_p^2+\sigma^2) = w_1$, which is
  precisely the posterior-mean shrinkage — **the MMSE estimator**.

Measured on a fine geometric $\tau$ grid (4000 steps, exact score, 400k
samples), terminal variance:

| sampler | terminal variance | target |
|---|---|---|
| half step ($\kappa = \tfrac12$, PF-ODE) | $0.15986$ | prior variance $0.16000$ |
| full step ($\kappa = 1$, existing update) | $0.04918$ | MMSE variance $0.04923$ |

So the existing sampler is not "an approximate ODE solver with a bug" — it is
an exact and deliberate MMSE accumulator, which is why it maximizes PSNR.
The half-step variant is the distribution-preserving map, which will produce
sharper images and **lower PSNR**. That is the distortion–perception
trade-off, and it must be reported as such: a PSNR drop from the $\kappa=\frac12$
sampler is the expected behavior, not a regression.

### Three samplers

All share the network and differ only in the update.

**(a) MMSE ladder** ($\kappa = 1$, deterministic) — the existing sampler.
$x \leftarrow x + \eta_k\,c_\theta$. Converges to $D_\theta$; maximizes PSNR;
blurs where the posterior is genuinely uncertain. Iterating to large $M$
sends the retained seed-noise weight $1/M \to 0$, but the limit is just
$D_\theta$ itself — i.e. the existing `prediction` output. **No free lunch
here**, and the companion document's advice to prefer `prediction` stands.

**(b) Probability-flow ODE** ($\kappa = \tfrac12$, deterministic).
$x \leftarrow x + \tfrac{1}{2}\eta_k\,c_\theta$. Deterministic in the seed,
correct marginals, sharper output, lower PSNR. Note this preserves marginals,
not the *posterior* — being deterministic it has no spread given $y_1$, in
the same way DDIM differs from DDPM.

**(c) Ancestral sampler** (stochastic) — the genuinely new capability. The
exact reverse transition is

$$x_{k+1} \;=\; x_k + \eta_k\,c_\theta(x_k,m_k) \;+\; \frac{\sqrt{V_{m_k}(x_k) + \sigma^2}}{m_k+1}\;z, \qquad z\sim\mathcal{N}(0,I),$$

for unit steps, since
$\operatorname{Var}(\Delta_m\mid\bar y_m = x) = \bigl(V_m(x) + \sigma^2\bigr)/(m+1)^2$
— the *pointwise* posterior variance of §3, which is exactly what §6's
uncertainty head estimates. Iterated to convergence this draws from
$p(x_0 \mid y_1)$:
running the exact recursion from a fixed $y_1 = 1.10$ gives mean $0.58052$
and variance $0.11080$ against the analytic posterior's $0.58077$ and
$0.11077$, while the mean-only ladder (a) collapses to variance $0$ — a point
mass at the MMSE estimate, as it should.

**A useful check on (c).** The law of total variance,
$\mathbb{E}[\operatorname{Var}(\Delta\mid\bar y)] + \operatorname{Var}(\mathbb{E}[\Delta\mid\bar y]) = \operatorname{Var}(\Delta)$,
holds to $10^{-6}$ in the toy, and implies the exact ancestral noise is
*strictly smaller* than the naive Euler–Maruyama noise $\sigma/\sqrt{m(m+1)}$
— measured ratios $0.809, 0.908, 0.963, 0.988$ at $m = 1,2,4,8$. The gap is
exactly the information the drift already supplied. Using EM noise instead of
the exact transition variance over-noises the early steps by 20%; this is the
same reason DDPM's ancestral sampler beats naive EM.

**Why (c) matters for microscopy.** An MMSE denoiser answers "what is the
most likely value of each pixel," and where the data cannot decide, it
answers with a blur that looks like a confident smooth structure. A posterior
sampler answers "what could the specimen be," and running it twenty times
shows *where the answers disagree*. For a scientific instrument, the second
question is often the one that matters — and the variance across draws is a
hallucination detector, since a feature invented by the prior varies across
draws while a feature supported by the measurement does not.

### The schedule decouples from integer frame counts

At training, $m$ must be an integer — one can only average whole frames. At
*sampling*, the state is characterized entirely by its noise variance
$\sigma^2\tau$, so nothing forces $\tau$ to lie in $\{1, \tfrac12, \tfrac13, \dots\}$.
Conditioning the network on continuous $\tau$ instead of integer $t$ lets the
sampler use any decreasing $\tau$-schedule — for instance geometric with 100
steps rather than the 15 the burst provides.

This matters more than it sounds, because **the integer grid is savagely
non-uniform**: $\tau$ runs over $1, \tfrac12, \tfrac13, \dots$, so the single
step $m = 1 \to 2$ traverses *half the entire time interval* $[0,1]$, and
$m=1\to4$ traverses three quarters. Every discretization error and every
Gaussian-transition approximation is concentrated in the first one or two
steps. This is, incidentally, a clean explanation for the companion report's
§5 finding that iteration did not help: the trajectory's first step is not a
small refinement, it is most of the journey.

To fill the continuum, the level between two real frame counts can be reached
by adding synthetic Gaussian noise to a real average: from $\bar y_m$,

$$x(\tau) \;=\; \bar y_m + \sigma\sqrt{\tau - \tfrac{1}{m}}\;z, \qquad \tau \ge \tfrac1m,$$

has exactly the right marginal variance. Two honest caveats. This
reintroduces simulated noise, which is the thing the method exists to avoid,
so it should be used only to *interpolate between* real levels and never to
replace them — the real-noise grounding at the integer levels is what makes
the model transfer. And the cleanest reachable state is bounded by the burst:
$\tau \ge 1/T$ always, since noise can be added but not removed.

## 6. Uncertainty for free

Theorem R2's martingale is not only a warning; it is also an exact accounting
identity. Since $x_0 - Z_M = \sum_{m\ge M}\zeta_m$ and martingale increments
are conditionally orthogonal,

$$\boxed{\;\mathbb{E}\bigl[\lVert x_0 - Z_M\rVert^2 \;\big|\; \mathcal{F}_M\bigr] \;=\; \sum_{m \ge M} \mathbb{E}\bigl[\lVert\zeta_m\rVert^2 \;\big|\; \mathcal{F}_M\bigr]\;}$$

**The posterior uncertainty is exactly the energy budget of the remaining
residuals** — a "denoising ladder" in which each rung's mean square is the
portion of the final error that rung resolves. Cross-terms vanish, so there
is no double counting.

This is directly estimable from noisy data. For a fresh frame $y_K$,

$$\mathbb{E}\bigl[(y_K - D^\ast)^2 \;\big|\; \bar y_m = x\bigr] \;=\; V_m(x) + \sigma^2,$$

which is the companion document's loss-floor decomposition read per pixel
rather than as a scalar. So a second network head $u_\theta(x,m)$ trained on
the squared residual of the first — target $(y_K - D_\theta)^2$, with
$D_\theta$ detached — estimates $V_m(x) + \sigma^2$, and

$$\hat V_m(x) \;=\; \max\bigl(u_\theta(x,m) - \hat\sigma^2(x),\; 0\bigr)$$

is a **per-pixel error bar on the denoised output, learned with no clean
images at any point**. It feeds sampler (c) directly, which closes the design:
the uncertainty head and the stochastic sampler are the same machinery, and
each validates the other (a miscalibrated $\hat V$ shows up as posterior draws
with visibly wrong spread).

Calibration is checkable without ground truth too: over held-out sources the
empirical mean of $(y_K - D_\theta)^2$ must match $\hat V + \hat\sigma^2$
bin-by-bin in predicted variance. With clean images available in the
synthetic datasets, the stronger check — that $(x_0 - D_\theta)^2$ averages
to $\hat V$ — is also available and should be the reported metric.

## 7. Implementation recipe

**The data layer needs no change.** `BatchFactory.sample_batch` already
returns exactly $(\bar y_{m(t)},\, y_K,\, t)$ with $K$ fresh, and every target
in this document is an arithmetic function of those three. The residual method
is a loss-and-sampler change, not a pipeline change. `min_replicas` stays
$T+1$.

**Network.** Keep `UNet.forward(x, t)`. Add an optional second output group
for $\log u_\theta$ (§6) — a channel-count change in the final projection
only. For continuous $\tau$ (§5), the timestep embedding takes a float rather
than an integer index; the existing sinusoidal embedding handles this without
modification.

**Loss.** With $m = $ `frames_at(t, T)` and $\hat\sigma$ from §3:

```
target_norm = sqrt(m / (m + 1)) * (y_K - x) / sigma      # unit variance, all m
loss_main   = mse(F_theta(x, t), target_norm)
x0_hat      = x + sigma * sqrt((m + 1) / m) * F_theta(x, t)
loss_var    = mse(u_theta(x, t), (y_K - x0_hat.detach()) ** 2)    # optional head
```

Antithetic $t$-pairing, EMA, and the rollout stage carry over unchanged; note
the §5 rollout invariant still applies (model outputs on the input side only),
and Theorem R1's proof, like Theorem 1's, never uses how the input was made.

**Sampler.** Extend `sample_step` with a drift scale and an optional noise
term, defaulting to today's behavior:

```
def sample_step(x, eps_hat, t, t_next, num_steps, *, drift=1.0, noise_var=None)
    # drift=1.0, noise_var=None  -> current MMSE ladder, bit-identical
    # drift=0.5                  -> probability-flow ODE
    # noise_var=V_hat + sigma^2  -> ancestral posterior sampling
```

**Tests to write**, in the style the repo already uses (numpy toys, no GPU,
no network):

1. R1 in a linear-Gaussian toy: the fitted residual optimum equals
   $c^\ast/(m+1)$.
2. R2: regressing $\zeta$ on $\bar y_m$ returns slope $\approx 0$.
3. R3: cached burst averages have covariance $\hat\sigma^2/\max(m,m')$.
4. **Equivalence**: `drift=1.0, noise_var=None` reproduces the current
   `sample_step` bit-exactly, and an $F_\theta$-parameterized model converts
   to an $\varepsilon_\theta$ one with identical outputs (the Corollary).
5. Preconditioning: $\tilde\Delta_m$ has unit variance at every level.
6. PF-ODE: the half-step recursion maps prior to prior in the Gaussian toy;
   the full step maps to the MMSE shrinkage.
7. Ancestral: transition variance $\le$ the Euler–Maruyama variance, with
   equality iff the prior is flat.

**Suggested order.** Preconditioning (§3) and the equivalence test first —
they are cheap and derisk everything else. Then the variance head (§6), which
is independently useful even if no new sampler ships. Then samplers (b) and
(c). The continuous-$\tau$ schedule last, since it is the only piece that
reintroduces synthetic noise.

## 8. What would falsify this

Predictions, in decreasing order of confidence:

1. **The equivalence holds.** A residual-parameterized model converted to
   $\varepsilon$-form matches an $\varepsilon$-trained model's optimum. If
   residual training produces *better* one-shot PSNR than the existing model
   by more than preconditioning explains, one of the two implementations is
   wrong — Theorem R1 leaves no room for a real gap.
2. **Naive residual training underperforms** the existing model, by roughly
   the amount §3's weighting predicts, and preconditioning removes the gap.
   This is the sharpest single test of §3.
3. **The $\kappa = \tfrac12$ sampler lowers PSNR and raises perceptual
   sharpness.** If it *raises* PSNR, Proposition R5's identification is wrong.
4. **Posterior draws disagree most where the existing model blurs.** If the
   spread is uniform across the image, either $\hat V$ is uncalibrated or the
   diagonal-Gaussian transition is too crude.
5. **Ancestral sampling is worst at $m = 1$.** The reverse transition is
   Gaussian only up to $O(V_m/\sigma^2)$, and $V_m$ is largest there — the
   same place the $\tau$-step is largest. Expect artifacts concentrated in
   the first step, mitigated by starting from a real multi-frame average when
   one is available.

## 9. Assumptions and failure modes

Inherits every assumption of the companion document's §6 and §8 — clipping
bias, salt-and-pepper, fixed-pattern noise, registration, dose drift — plus:

| new assumption | needed for | breaks when |
|---|---|---|
| Gaussian $n_j$ | R3's "is a Brownian motion", R4's Tweedie form | photon-starved Poisson; the *covariance* structure survives, the increment law does not |
| $\bar y_m$ Markov / sufficient | carrying only the running average as state | correlated frames (A2 violation) |
| reverse transition $\approx$ diagonal Gaussian | ancestral sampler (c) | small $m$, and any strongly correlated posterior — a diagonal covariance cannot express structured uncertainty |
| $\hat\sigma^2$ accurate | the §3 normalization and (c)'s noise scale | signal-dependent noise estimated as a scalar; use per-pixel |
| $\hat V$ calibrated | (c)'s spread, §6's error bars | under-trained variance head; check against held-out squared error |

The diagonal-covariance limitation is the most consequential and deserves
naming plainly: sampler (c) will produce correct per-pixel spread but
independent per-pixel *noise* in its draws, where the true posterior has
spatially coherent ambiguity (an edge that could be in one of two places).
Fixing that properly means a structured or learned covariance, or enough
sampling steps that the accumulated small steps generate coherence the way a
diffusion model does — the latter is the standard answer, and is the main
reason the continuous-$\tau$ schedule of §5 is worth building rather than
optional.

## 10. Relation to prior work

- **DDPM / DDIM** (Ho et al., 2020; Song et al., 2021): the increment
  parameterization is $\varepsilon$- vs $x_0$- vs $v$-prediction, and R5's
  full/half distinction is ancestral-vs-DDIM, both specialized to a schedule
  whose "noise" is measured rather than simulated.
- **Score-based SDEs** (Song et al., 2021): Theorem R3 places the burst
  process exactly in the VE family with $g^2 = \sigma^2$ and $\tau = 1/m$,
  making the reverse SDE and probability-flow ODE available verbatim.
- **EDM preconditioning** (Karras et al., 2022): §3 is that argument for this
  schedule; the $1/(m+1)^2$ weighting is the concrete pathology it prevents.
- **Noise2Noise** (Lehtinen et al., 2018): still the engine underneath — R1
  reduces to it, since the residual target is a fresh frame minus a known
  input.
- **DnCNN residual learning** (Zhang et al., 2017): the skip-connection
  argument for predicting the correction rather than the image.
- **Doob martingales / Lévy's upward theorem**: R2 and §6's orthogonal
  decomposition are textbook consequences; the contribution here is noticing
  that the burst-averaging posterior-mean process *is* such a martingale, and
  that this forbids one of the two natural residual objectives.
