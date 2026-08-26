"""Manually display averages of noisy images from one source image."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Edit this path if your pipeline output is elsewhere.
NOISY_FOLDER = Path(r"E:\PythonProjects\ddim\data\BBBC038-noisy\noisy")
CLEAN_FOLDER = NOISY_FOLDER.parent / "clean"
SOURCE_INDEX = 2
AVERAGE_COUNTS = (1, 10, 20, 50, 100)


def main() -> None:
    files = sorted(NOISY_FOLDER.glob(f"{SOURCE_INDEX:05d}_*.png"))
    clean_path = CLEAN_FOLDER / f"{SOURCE_INDEX:05d}.png"
    required = max(AVERAGE_COUNTS)
    if len(files) < required:
        raise RuntimeError(
            f"Need {required} noisy images for source {SOURCE_INDEX}, "
            f"but found {len(files)} in {NOISY_FOLDER}."
        )
    if not clean_path.is_file():
        raise RuntimeError(f"Clean image not found: {clean_path}")

    with Image.open(clean_path) as image:
        clean = np.asarray(image).copy()

    running_sum = None
    image_dtype = None
    image_shape = None
    averages = {}

    for number, path in enumerate(files[:required], start=1):
        with Image.open(path) as image:
            array = np.asarray(image).copy()

        if running_sum is None:
            image_dtype = array.dtype
            image_shape = array.shape
            if clean.shape != image_shape:
                raise RuntimeError(
                    f"Clean image shape {clean.shape} does not match "
                    f"noisy image shape {image_shape}."
                )
            running_sum = np.zeros(array.shape, dtype=np.float64)
        elif array.shape != image_shape:
            raise RuntimeError(f"Image shape changed at {path}.")

        running_sum += array
        if number in AVERAGE_COUNTS:
            averages[number] = running_sum / number

    differences = {
        count: np.abs(averages[count] - clean.astype(np.float64))
        for count in AVERAGE_COUNTS
    }
    differences = {
        count: difference.mean(axis=2) if difference.ndim == 3 else difference
        for count, difference in differences.items()
    }
    difference_max = max(
        float(difference.max()) for difference in differences.values()
    )
    if difference_max == 0:
        difference_max = 1

    _, axes = plt.subplots(
        2,
        len(AVERAGE_COUNTS) + 1,
        figsize=(22, 8),
    )

    if clean.ndim == 2:
        axes[0, 0].imshow(clean, cmap="gray")
    else:
        axes[0, 0].imshow(clean)
    axes[0, 0].set_title("Clean")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for column, count in enumerate(AVERAGE_COUNTS, start=1):
        displayed = np.rint(averages[count]).astype(image_dtype)
        if displayed.ndim == 2:
            axes[0, column].imshow(displayed, cmap="gray")
        else:
            axes[0, column].imshow(displayed)
        axes[0, column].set_title(f"Average n={count}")
        axes[0, column].axis("off")

        axes[1, column].imshow(
            differences[count],
            cmap="magma",
            vmin=0,
            vmax=difference_max,
        )
        axes[1, column].set_title(f"|Average n={count} - clean|")
        axes[1, column].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
