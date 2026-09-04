# Legacy SEM training loss and checkpoint diagnostics

This note applies to SEM diffusion runs launched through the legacy
`python main.py` interface. It documents how to interpret the training loss,
why sample quality can improve after that loss plateaus, and how the legacy
runner handles learning-rate changes and checkpoints.

## Observed 512 x 512 run

The investigated remote run had the following confirmed properties:

- 3,600 source images stored as RGB PNGs whose three channels contain the
  same values;
- `data.channels`, `model.in_channels`, and `model.out_ch` all set to `1`;
- `data.image_size: 512`;
- `model.ch_mult: [1, 1, 2, 2, 4, 4]`;
- global batch size 64 across four H100 GPUs;
- an initial learning rate of `2e-4`, subsequently intended to be changed to
  `2e-5`;
- more than 500,000 optimizer steps.

Because `data.channels` is one, `SEMImageDataset` converts every source image
to PIL mode `L` before `ToTensor`. An RGB pixel with `R = G = B` retains the
same grayscale intensity. The fact that the PNG files have three identical
stored channels therefore has no training or output effect in this run.

## TensorBoard chart scaling

TensorBoard's **Ignore outliers in chart scaling** option can make this loss
look much less stable than it is. When enabled, TensorBoard excludes the very
large initial losses when choosing the vertical scale. It then expands the
late-loss band to fill the chart, visually exaggerating its ordinary
batch-to-batch variation.

With that option disabled, the observed curve has two phases:

1. a large loss at initialization followed by a steep decline during roughly
   the first 800 steps; and
2. a long, noisy plateau with no obvious downward slope in the aggregate
   scalar.

That second phase is a plateau, not evidence that the optimizer is diverging.
Convergence to a stationary region does not require individual stochastic
loss values to become constant.

TensorBoard smoothing is also much shorter-term than it may appear. An
exponential smoothing factor of `0.90` has an effective window of roughly ten
logged values, while `0.99` has an effective window of roughly one hundred.
If the runner logs every ten optimizer steps, those correspond to only about
100 and 1,000 optimizer steps. Both are short windows relative to a
500,000-step run. Long-term trends should instead be checked with
non-overlapping means over at least 10,000 to 50,000 steps.

## Why samples improve after aggregate loss plateaus

The legacy objective is an unweighted epsilon-prediction MSE:

```text
mean_over_batch(sum_over_channels_and_pixels((epsilon - predicted_epsilon)^2))
```

Each batch uses newly generated Gaussian noise and uniformly sampled diffusion
timesteps. The resulting single scalar combines prediction problems with very
different noise levels and difficulty.

This objective can become insensitive to perceptually important progress:

- A 512 x 512 image contributes 262,144 pixel terms. Large uniform regions
  and locally predictable texture dominate the total, while relatively sparse
  SEM edges and pattern geometry have less influence.
- The network can learn basic local noise statistics quickly, producing the
  initial loss collapse, and then improve global structure much more slowly.
- Small improvements at sampling-critical timesteps can substantially improve
  a full reverse trajectory without producing a visible change in the
  uniformly averaged training loss.
- Sampling uses EMA parameters when EMA is enabled, whereas the logged loss is
  from the current training parameters. The EMA model can improve while the
  instantaneous loss continues to occupy the same noisy band.

In the investigated run, a checkpoint near step 530,000 produced substantially
more realistic samples than the checkpoint at step 80,000. This is direct
evidence that useful model improvement continued after the aggregate loss had
lost a visible downward trend. It also exposed a sampling-script error: a
fixed default `sampling.ckpt_id` had repeatedly selected step 80,000.

This result does not by itself prove better generalization. With 3,600 source
images, a global batch of 64, and hundreds of thousands of updates, memorization
must be checked against held-out images.

## Multi-GPU logging caveat

When the current legacy runner is launched with `torchrun`, global batch 64 is
split into 16 images per rank across four ranks. DDP synchronizes and averages
the gradients correctly, but the TensorBoard scalar is rank zero's local loss;
the code does not all-reduce the scalar before logging it. The displayed curve
therefore represents a local batch of 16 rather than the global batch of 64.

When `main.py` is launched as one process and uses `torch.nn.DataParallel`, the
logged loss covers the whole global batch. The exact launch command therefore
matters when interpreting the graph.

Resumed runs can introduce another display problem. The legacy code opens a
new `SummaryWriter` in the same directory without a `purge_step`. If training
resumes from an older checkpoint, old and new event files can contain
overlapping steps, and TensorBoard may display both histories. Never interpret
a merged curve until event-file step ranges have been checked for overlap.

## Learning-rate changes during resume

Editing `optim.lr` in the YAML is not enough to change the learning rate when
resuming a legacy checkpoint. The runner constructs an optimizer from the
current config and then loads the saved optimizer state:

```python
optimizer.load_state_dict(states[1])
```

The load restores the old checkpoint learning rate. The legacy resume path
explicitly replaces only the saved Adam `eps` value.

Verify the actual saved learning rate with:

```python
import torch

state = torch.load("ckpt.pth", map_location="cpu")
print(state[1]["param_groups"][0]["lr"])
```

To make the configured learning rate take effect after a resume, override it
after loading the optimizer state:

```python
optimizer.load_state_dict(states[1])
for group in optimizer.param_groups:
    group["lr"] = self.config.optim.lr
```

A fresh run started with `2e-5` does not have this problem. The override is
needed only when loading an existing optimizer state.

## Legacy checkpoint behavior

The legacy runner does not calculate validation loss and does not choose a
best checkpoint. Although it creates a validation dataset and accepts a
`validation_freq` setting, the training loop never evaluates that dataset.

At every snapshot it writes:

- `ckpt_<step>.pth`, a numbered checkpoint; and
- `ckpt.pth`, an overwritten copy of the most recently saved checkpoint.

The sampling path uses `sampling.ckpt_id` as follows:

```yaml
sampling:
  ckpt_id: 80000
```

This always loads `ckpt_80000.pth`. To sample the latest saved checkpoint,
remove `ckpt_id` or set it to null:

```yaml
sampling:
  ckpt_id: null
```

This loads `ckpt.pth`. "Latest" is not equivalent to "best"; the legacy run
has no metric from which it could determine the latter. In particular, never
select the checkpoint corresponding to the lowest individual training-loss
point. That point is strongly affected by its randomly selected images,
timesteps, and noise.

## Selecting a checkpoint responsibly

Evaluate a sequence of numbered checkpoints with an identical inference
protocol. At minimum:

1. persist one bank of at least 64 to 128 initial noise tensors;
2. use the same DDIM timestep sequence, `eta`, EMA setting, and output
   postprocessing for every checkpoint;
3. compare realism and failure rate across those identical latent inputs;
4. compare generated intensity and noise distributions with a held-out SEM
   split; and
5. perform nearest-neighbor checks against the training set to detect
   memorization.

For future runs, add a deterministic validation pass using fixed validation
images, fixed Gaussian noise, and fixed timesteps. Report both the overall
per-pixel loss and loss in timestep bins. Also log the pre-clipping gradient
norm and whether gradient clipping activated. A best-checkpoint pointer can
then be updated using the deterministic validation metric rather than a noisy
training batch.

## Practical conclusion

The corrected TensorBoard view and the improved step-530,000 samples do not
support the earlier interpretation of catastrophic non-convergence. The
aggregate epsilon loss learned its easy, high-volume signal quickly and then
became a poor indicator of slower perceptual progress. The next decision
should be based on fixed-seed, held-out checkpoint evaluation rather than on
waiting for the aggregate training-loss curve to resume falling.
