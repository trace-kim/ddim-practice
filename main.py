import argparse
import logging
import os
import sys
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.tensorboard as tb
import yaml

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


_NAMED_CONFIG_OVERRIDES = {
    "image_size": (("data", "image_size"),),
    "batch_size": (("training", "batch_size"),),
    "learning_rate": (("optim", "lr"),),
    "max_steps": (("training", "max_steps"), ("training", "n_iters")),
    "data_path": (("data", "data_path"), ("data", "data_dir"), ("data", "path")),
    "num_workers": (("data", "num_workers"),),
    "diffusion_steps": (("diffusion", "num_diffusion_timesteps"),),
    "model_ch": (("model", "ch"),),
}


def distributed_process_info(
    environ: Mapping[str, str] | None = None,
    local_rank_override: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(rank, local_rank, world_size)`` from the torchrun environment."""

    values = os.environ if environ is None else environ
    try:
        rank = int(values.get("RANK", "0"))
        local_rank = (
            int(local_rank_override)
            if local_rank_override is not None
            else int(values.get("LOCAL_RANK", "0"))
        )
        world_size = int(values.get("WORLD_SIZE", "1"))
    except (TypeError, ValueError) as error:
        raise ValueError("RANK, LOCAL_RANK, and WORLD_SIZE must be integers") from error

    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("RANK must be between 0 and WORLD_SIZE - 1")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be nonnegative")
    return rank, local_rank, world_size


def _config_path_exists(config: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _coerce_config_override(current: Any, value: Any, path: tuple[str, ...]) -> Any:
    label = ".".join(path)
    if isinstance(current, dict):
        raise ValueError(f"{label} is a section; override one of its leaf values")
    if current is None:
        return value
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{label} expects a boolean")
        return value
    if isinstance(current, int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label} expects an integer")
        return value
    if isinstance(current, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{label} expects a number")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ValueError(f"{label} expects a string")
        return value
    if isinstance(current, list):
        if not isinstance(value, list):
            raise ValueError(f"{label} expects a YAML list")
        return value
    if not isinstance(value, type(current)):
        raise ValueError(f"{label} expects {type(current).__name__}")
    return value


def _set_config_override(
    config: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    if not path or any(not key for key in path):
        raise ValueError("config override paths must not contain empty components")
    current: Any = config
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"unknown config key: {'.'.join(path)}")
        current = current[key]
    leaf = path[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise ValueError(f"unknown config key: {'.'.join(path)}")
    current[leaf] = _coerce_config_override(current[leaf], value, path)


def apply_config_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[str, ...]:
    """Apply typed convenience flags and repeated dotted YAML overrides."""

    applied: list[str] = []
    for argument, candidate_paths in _NAMED_CONFIG_OVERRIDES.items():
        value = getattr(args, argument, None)
        if value is None:
            continue
        path = next(
            (candidate for candidate in candidate_paths if _config_path_exists(config, candidate)),
            None,
        )
        if path is None:
            choices = " or ".join(".".join(candidate) for candidate in candidate_paths)
            raise ValueError(f"--{argument.replace('_', '-')} requires config key {choices}")
        _set_config_override(config, path, value)
        applied.append(f"{'.'.join(path)}={value!r}")

    for override in getattr(args, "config_overrides", ()):
        key, separator, raw_value = override.partition("=")
        if not separator or not key.strip():
            raise ValueError("--set must use SECTION.KEY=VALUE syntax")
        path = tuple(part.strip() for part in key.split("."))
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid YAML value for {key.strip()}: {error}") from error
        _set_config_override(config, path, value)
        applied.append(f"{'.'.join(path)}={value!r}")
    return tuple(applied)


def parse_args_and_config(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--image-size", type=int, help="Override data.image_size")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size")
    parser.add_argument("--learning-rate", type=float, help="Override optim.lr")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override training.max_steps or legacy training.n_iters",
    )
    parser.add_argument("--data-path", type=str, help="Override the configured dataset path")
    parser.add_argument("--num-workers", type=int, help="Override data.num_workers")
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        help="Override diffusion.num_diffusion_timesteps",
    )
    parser.add_argument("--model-ch", type=int, help="Override model.ch")
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override any existing YAML leaf; may be repeated",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args(argv)
    try:
        args.rank, args.local_rank, args.world_size = distributed_process_info(
            local_rank_override=args.local_rank
        )
    except ValueError as error:
        parser.error(str(error))
    args.distributed = args.world_size > 1
    args.is_main_process = args.rank == 0
    if args.distributed and (args.test or args.sample):
        parser.error("torchrun multi-process execution is supported for training only")
    if args.is_main_process:
        logging.warning(
            "python main.py is deprecated; use the typed 'ddimctl' workflow for new SEM runs"
        )
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(os.path.join("configs", args.config), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        parser.error("config file must contain a YAML mapping")
    try:
        args.applied_config_overrides = apply_config_overrides(config, args)
    except ValueError as error:
        parser.error(str(error))
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    new_config.tb_logger = None
    if not args.test and not args.sample:
        if args.is_main_process:
            if not args.resume_training:
                if os.path.exists(args.log_path):
                    raise FileExistsError(
                        "Refusing to overwrite existing legacy run directory: {}".format(
                            args.log_path
                        )
                    )
                os.makedirs(args.log_path)

                with open(
                    os.path.join(args.log_path, "config.yml"), "w", encoding="utf-8"
                ) as f:
                    yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

            new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
            # setup logger
            level = getattr(logging, args.verbose.upper(), None)
            if not isinstance(level, int):
                raise ValueError("level {} not supported".format(args.verbose))

            handler1 = logging.StreamHandler()
            handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
            formatter = logging.Formatter(
                "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
            )
            handler1.setFormatter(formatter)
            handler2.setFormatter(formatter)
            logger = logging.getLogger()
            logger.addHandler(handler1)
            logger.addHandler(handler2)
            logger.setLevel(level)
    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    raise FileExistsError(
                        "Refusing to overwrite existing sample directory: {}".format(
                            args.image_folder
                        )
                    )

    # add device
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun DDP training requires CUDA")
        if args.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "LOCAL_RANK {} is not available; found {} visible CUDA device(s)".format(
                    args.local_rank, torch.cuda.device_count()
                )
            )
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.is_main_process:
        logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    process_seed = args.seed + args.rank
    torch.manual_seed(process_seed)
    np.random.seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    if args.is_main_process:
        logging.info("Writing log file to {}".format(args.log_path))
        logging.info("Exp instance id = {}".format(os.getpid()))
        logging.info("Exp comment = {}".format(args.comment))
        if args.applied_config_overrides:
            logging.info(
                "Config overrides = {}".format(", ".join(args.applied_config_overrides))
            )

    exit_code = 0
    try:
        if args.distributed:
            dist.init_process_group(backend="nccl", init_method="env://")
            dist.barrier()
        runner = Diffusion(args, config, device=config.device)
        if args.sample:
            runner.sample()
        elif args.test:
            runner.test()
        else:
            runner.train()
    except Exception:
        logging.error("rank %s failed:\n%s", args.rank, traceback.format_exc())
        exit_code = 1
    finally:
        if config.tb_logger is not None:
            config.tb_logger.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
