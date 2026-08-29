"""Diagnostic probe: where do a trained burst_diffusion model's predictions sit?

For one held-out source, measures the prediction's PSNR against (a) the clean
image and (b) an actual fresh noisy frame, when the network is fed:

  A. REAL m-frame averages (in-distribution inputs, as in training), and
  B. the sampler's own pseudo-averages (the iterative trajectory).

Interpretation (see docs/burst_diffusion_method.md S5 and
docs/burst_diffusion_qna.md):
- pred-vs-clean high while pred-vs-frame ~= the frame's own noise distance
  proves the network outputs the clean estimate, not "a noisy image";
- the gap between A and B at the same t isolates the train/inference input-
  distribution gap. After self-rollout finetuning the two curves should merge.

Usage (from the repo root, package installed):
  python tools/probe_burst_predictions.py
  python tools/probe_burst_predictions.py --dataset data/MIIC-burst-p10 `
      --checkpoint runs/burst_diffusion/miic_p10/ckpt_latest.pt --source-index 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from burst_diffusion.data import resolve_burst_dir
from burst_diffusion.metrics import psnr
from burst_diffusion.sample import Sampler
from burst_diffusion.schedule import frames_at, sample_step


def _load01(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="data/BBBC038-burst-p10")
    parser.add_argument("--checkpoint", default="runs/burst_diffusion/bbbc038_p10/ckpt_latest.pt")
    parser.add_argument("--source-index", type=int, default=12)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sampler = Sampler.from_checkpoint(args.checkpoint, device=args.device)
    num_steps = sampler.num_steps
    burst_dir = resolve_burst_dir(args.dataset)
    clean = _load01(burst_dir / "clean" / f"{args.source_index:05d}.png")
    frame_paths = sorted((burst_dir / "noisy").glob(f"{args.source_index:05d}_*.png"))
    if len(frame_paths) < num_steps + 1:
        raise SystemExit(
            f"source {args.source_index} has {len(frame_paths)} frames; "
            f"the probe needs at least T+1 = {num_steps + 1}"
        )
    frames = [_load01(path) for path in frame_paths]

    height, width = clean.shape
    top, left = (height - args.size) // 2, (width - args.size) // 2
    window = np.s_[top : top + args.size, left : left + args.size]
    clean_c = clean[window]
    frames_c = [frame[window] for frame in frames]
    fresh = frames_c[num_steps]  # never fed to the network below

    model = sampler.model
    device = sampler.device

    def to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((array * 2.0 - 1.0).astype(np.float32))[None, None].to(device)

    def to_image(tensor: torch.Tensor) -> np.ndarray:
        return ((tensor[0, 0].clamp(-1, 1) + 1.0) / 2.0).cpu().numpy().astype(np.float64)

    levels = sorted(
        {num_steps, (3 * num_steps) // 4, num_steps // 2, num_steps // 4, 1} - {0},
        reverse=True,
    )

    print(f"T = {num_steps} | source {args.source_index} | crop {args.size}px")
    print(f"raw frame y1 vs clean:            {psnr(clean_c, frames_c[0]):.2f} dB")
    print(f"raw frame y1 vs another frame:    {psnr(fresh, frames_c[0]):.2f} dB")
    print()
    print("A) network fed REAL m-frame averages (in-distribution, as in training)")
    print("    t   m   pred-vs-clean   pred-vs-a-real-fresh-frame")
    with torch.no_grad():
        for t in levels:
            m = frames_at(t, num_steps)
            real_average = np.mean(frames_c[:m], axis=0)
            pred = to_image(model(to_tensor(real_average), torch.tensor([float(t)], device=device)))
            print(f"   {t:2d}  {m:2d}   {psnr(clean_c, pred):6.2f} dB      {psnr(fresh, pred):6.2f} dB")
    print()
    print("B) network fed its OWN pseudo-averages (the iterative sampler trajectory)")
    print("    t   m   pred-vs-clean   state-vs-clean")
    x = to_tensor(frames_c[0])
    with torch.no_grad():
        for t in range(num_steps, 0, -1):
            pred_t = model(x, torch.tensor([float(t)], device=device))
            x = sample_step(x, pred_t, t, t - 1, num_steps)
            if t in levels:
                print(
                    f"   {t:2d}  {frames_at(t, num_steps):2d}   "
                    f"{psnr(clean_c, to_image(pred_t)):6.2f} dB      "
                    f"{psnr(clean_c, to_image(x)):6.2f} dB"
                )
    print()
    mse_pred_fresh = float(np.mean((to_image(pred_t) - fresh) ** 2))
    sigma_sq = 10 ** (-psnr(clean_c, frames_c[0]) / 10.0)
    print(f"MSE(final prediction, actual fresh frame): {mse_pred_fresh:.4f}")
    print(f"single-frame noise variance (from its PSNR): {sigma_sq:.4f}")
    print("(these two matching is the loss-floor signature: the prediction is as close")
    print(" to a noisy target as mathematically possible -- i.e. it is the clean estimate)")


if __name__ == "__main__":
    main()
