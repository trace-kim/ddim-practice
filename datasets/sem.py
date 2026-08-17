import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class SEMImageDataset(Dataset):
    """Load SEM images directly from a filesystem directory.

    By default, the dataset keeps only file paths in memory so it remains
    compatible with PyTorch DataLoader workers on Windows and Linux.
    """

    DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        root,
        transform=None,
        channels=1,
        recursive=False,
        extensions=None,
        cache_in_memory=False,
    ):
        try:
            configured_root = os.fspath(root)
        except TypeError as error:
            raise TypeError("SEM dataset root must be a path-like value") from error

        if not isinstance(configured_root, str) or not configured_root.strip():
            raise ValueError("SEM dataset root must not be empty")

        configured_root = os.path.expandvars(os.path.expanduser(configured_root))
        self.root = Path(configured_root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(
                "SEM dataset directory does not exist or is not a directory: {}".format(
                    self.root
                )
            )

        if channels not in (1, 3):
            raise ValueError(
                "SEMImageDataset supports 1 or 3 channels, got {}".format(channels)
            )

        selected_extensions = (
            self.DEFAULT_EXTENSIONS if extensions is None else extensions
        )
        if isinstance(selected_extensions, str):
            selected_extensions = (selected_extensions,)

        self.extensions = self._normalize_extensions(selected_extensions)
        self.transform = transform
        self.channels = channels
        self.recursive = recursive
        self.cache_in_memory = cache_in_memory

        iterator = self.root.rglob("*") if recursive else self.root.glob("*")
        self.files = sorted(
            (
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in self.extensions
            ),
            key=lambda path: path.relative_to(self.root).as_posix().casefold(),
        )
        if not self.files:
            raise RuntimeError(
                "No SEM images with extensions {} were found in {}".format(
                    ", ".join(self.extensions), self.root
                )
            )

        self.cached_images = None
        if self.cache_in_memory:
            self.cached_images = [self._load_image(path) for path in self.files]

    @staticmethod
    def _normalize_extensions(extensions):
        try:
            extensions = tuple(extensions)
        except TypeError as error:
            raise TypeError(
                "SEM image extensions must be a string or iterable"
            ) from error

        normalized = []
        for extension in extensions:
            if not isinstance(extension, str) or not extension.strip():
                raise ValueError("SEM image extensions must be non-empty strings")

            extension = extension.strip().lower()
            if extension.startswith("*."):
                extension = extension[1:]
            elif not extension.startswith("."):
                extension = ".{}".format(extension)

            if extension not in normalized:
                normalized.append(extension)

        if not normalized:
            raise ValueError("At least one SEM image extension must be configured")

        return tuple(normalized)

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        try:
            with Image.open(path) as image:
                image = image.convert("L" if self.channels == 1 else "RGB")
                if self.transform is not None:
                    image = self.transform(image)
        except Exception as error:
            raise RuntimeError("Failed to load SEM image: {}".format(path)) from error

        return image

    def __getitem__(self, index):
        if self.cached_images is None:
            image = self._load_image(self.files[index])
        else:
            image = self.cached_images[index]

        return image, 0
