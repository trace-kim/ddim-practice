import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.lsun import LSUN
from datasets.sem import SEMImageDataset
from torch.utils.data import Subset
import numpy as np


def _get_sem_data_path(data_config):
    """Return the configured SEM directory without coupling it to args.exp."""

    configured_paths = []
    for field in ("data_path", "data_dir", "path"):
        value = getattr(data_config, field, None)
        if value is not None:
            configured_paths.append((field, value))

    if not configured_paths:
        raise ValueError(
            "SEM requires a local image directory in data.data_path "
            "(data.data_dir is also supported)"
        )

    normalized_paths = []
    for field, value in configured_paths:
        try:
            value = os.fspath(value)
        except TypeError as error:
            raise TypeError(
                "data.{} must be a path-like value".format(field)
            ) from error

        if not isinstance(value, str) or not value.strip():
            raise ValueError("data.{} must not be empty".format(field))

        normalized_paths.append(
            (
                field,
                os.path.abspath(os.path.expandvars(os.path.expanduser(value))),
            )
        )

    selected_field, selected_path = normalized_paths[0]
    selected_comparison = os.path.normcase(os.path.normpath(selected_path))
    for field, path in normalized_paths[1:]:
        if os.path.normcase(os.path.normpath(path)) != selected_comparison:
            raise ValueError(
                "Conflicting SEM directories configured in data.{} and data.{}".format(
                    selected_field, field
                )
            )

    return selected_path


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataset(args, config):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    if config.data.dataset == "CIFAR10":
        dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10"),
            train=True,
            download=True,
            transform=tran_transform,
        )
        test_dataset = CIFAR10(
            os.path.join(args.exp, "datasets", "cifar10_test"),
            train=False,
            download=True,
            transform=test_transform,
        )

    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(args.exp, "datasets", "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(args.exp, "datasets", "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )

    elif config.data.dataset == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        if config.data.random_flip:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ToTensor(),
                    ]
                ),
            )
        else:
            dataset = LSUN(
                root=os.path.join(args.exp, "datasets", "lsun"),
                classes=[train_folder],
                transform=transforms.Compose(
                    [
                        transforms.Resize(config.data.image_size),
                        transforms.CenterCrop(config.data.image_size),
                        transforms.ToTensor(),
                    ]
                ),
            )

        test_dataset = LSUN(
            root=os.path.join(args.exp, "datasets", "lsun"),
            classes=[val_folder],
            transform=transforms.Compose(
                [
                    transforms.Resize(config.data.image_size),
                    transforms.CenterCrop(config.data.image_size),
                    transforms.ToTensor(),
                ]
            ),
        )

    elif config.data.dataset == "FFHQ":
        if config.data.random_flip:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()]
                ),
                resolution=config.data.image_size,
            )
        else:
            dataset = FFHQ(
                path=os.path.join(args.exp, "datasets", "FFHQ"),
                transform=transforms.ToTensor(),
                resolution=config.data.image_size,
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9) :],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    elif config.data.dataset == "SEM":
        data_path = _get_sem_data_path(config.data)
        recursive = getattr(config.data, "recursive", False)
        extensions = getattr(config.data, "extensions", None)
        cache_in_memory = getattr(config.data, "cache_in_memory", False)
        if cache_in_memory and config.data.random_flip:
            raise ValueError(
                "data.cache_in_memory cannot be combined with random_flip because "
                "the transformed result is cached"
            )

        sem_test_transform = transforms.Compose(
            [
                transforms.Resize(
                    (config.data.image_size, config.data.image_size)
                ),
                transforms.ToTensor(),
            ]
        )
        if config.data.random_flip:
            sem_train_transform = transforms.Compose(
                [
                    transforms.Resize(
                        (config.data.image_size, config.data.image_size)
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        else:
            sem_train_transform = sem_test_transform

        dataset = SEMImageDataset(
            root=data_path,
            transform=sem_train_transform,
            channels=config.data.channels,
            recursive=recursive,
            extensions=extensions,
            cache_in_memory=cache_in_memory,
        )

        validation_split = getattr(config.data, "validation_split", 0.1)
        if not 0.0 <= validation_split < 1.0:
            raise ValueError(
                "data.validation_split must be in [0, 1), got {}".format(
                    validation_split
                )
            )

        num_items = len(dataset)
        indices = np.arange(num_items)
        random_state = np.random.RandomState(
            getattr(config.data, "split_seed", 2019)
        )
        random_state.shuffle(indices)

        num_test_items = int(num_items * validation_split)
        if validation_split > 0 and num_items > 1:
            num_test_items = max(1, min(num_items - 1, num_test_items))

        if num_test_items:
            test_indices = indices[-num_test_items:].tolist()
            train_indices = indices[:-num_test_items].tolist()
        else:
            test_indices = []
            train_indices = indices.tolist()

        if config.data.random_flip and test_indices:
            test_base_dataset = SEMImageDataset(
                root=data_path,
                transform=sem_test_transform,
                channels=config.data.channels,
                recursive=recursive,
                extensions=extensions,
                cache_in_memory=cache_in_memory,
            )
        else:
            test_base_dataset = dataset

        test_dataset = Subset(test_base_dataset, test_indices)
        dataset = Subset(dataset, train_indices)
    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
