# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Three layers, in order of how "supported" they are:

1. **`ddimctl/`** — the supported, reproducible training workflow for SEM (scanning electron microscope image) DDIM experiments. This is where new work should happen.
2. **Legacy DDIM research code** (`main.py`, `runners/`, `models/`, `functions/`, `datasets/`) — the original Song/Meng/Ermon DDIM implementation. Still used for sampling and as the actual training engine underneath `ddimctl` in some paths, but `python main.py` directly is deprecated for launching new training and only kept for compatibility.
3. **`noising_pipeline/`** — a standalone, portable utility for generating paired clean/noisy microscopy image datasets (e.g. from BBBC038). Unrelated to the diffusion training loop itself; only shares the repo.

Read `docs/training_workflow.md` before making any change that touches training, run bundles, machine profiles, or executors — it is the authoritative spec for that system and is kept in sync with the code. `TRAINING_GUIDE.md` is the condensed EN/KR quick-start for the same workflow. `docs/sem_dataset_migration.md` explains the SEM directory loader and a specific DataLoader-worker-restart perf pitfall (see Gotchas below).

## Commands

```
python -m pip install -e ".[dev]"        # install package + ddimctl entry point + pytest
python -m pip install -e ".[tracking]"   # add MLflow/psutil/nvidia-ml-py (only needed on the tracking-UI host)

python -m pytest                          # full suite (testpaths = tests/, see pyproject.toml)
python -m pytest tests/test_ddimctl_cli.py -q     # one module
python -m pytest tests/test_ddimctl_cli.py::test_cli_enforces_the_single_active_experiment_config  # one test

ddimctl doctor --machine <id> --exercise-executor   # validate a training target without launching
ddimctl train wizard --machine <id>                 # interactive plan + launch for the supported workflow
ddimctl train plan --machine <id> ...                # non-interactive: prints the resolved plan + canonical launch command, writes nothing
ddimctl run status|logs|stop|resume <run_dir>
ddimctl track serve / track publish <run_dir>        # local-only MLflow UI/publish
```

`python main.py --config {DATASET}.yml --exp {PATH} --doc {NAME} ...` is legacy: retained for sampling (`--sample --fid`, `--interpolation`, `--sequence`) and for compatibility, emits a deprecation warning, and lacks typed planning, immutable run bundles, source snapshotting, executor integration, and tracking. `main.py` does support ad-hoc YAML overrides without editing config files: convenience flags (`--image-size`, `--batch-size`, `--learning-rate`, `--max-steps`, `--data-path`, `--num-workers`, `--diffusion-steps`, `--model-ch`) plus repeatable `--set SECTION.KEY=VALUE` (YAML-typed values); unknown keys/incompatible types are rejected, and resolved config is saved to `<exp>/logs/<doc>/config.yml`.

There is no configured linter/formatter — match the style of nearby code.

## Architecture: the `ddimctl` reproducible workflow

The core design idea: **machine-specific operational facts** (executor, paths, GPU) live in a **machine profile** stored *outside* the repo (`%APPDATA%\ddimctl\profiles` on Windows, `$XDG_CONFIG_HOME/ddimctl/profiles` on Linux), while **experiment recipe settings** (batch size, LR, steps, schedule) live in a versioned YAML (`configs/sem.yml`) and CLI overrides. This separation is enforced by distinct Pydantic schemas — never let operational values leak into `TrainingSpec` or vice versa.

Module responsibilities (`ddimctl/`):
- `schemas.py` — Pydantic models (`MachineProfile`, `TrainingSpec`, `RunManifest`, `SlurmResources`, `ReproducibilityMode`, `ExecutorType`) with strict validation (unknown keys, duplicate scalar CLI options, and incompatible types are rejected rather than silently accepted).
- `profiles.py` — read/write machine profiles to the per-user config dir.
- `bundles.py` — the run bundle: dataset fingerprinting, deterministic source snapshotting (tars the exact tracked + dirty working-tree source so a queued run cannot be changed by later edits), atomic JSON/text writes, GPU isolation (`isolate_cuda_gpu`, sets `CUDA_VISIBLE_DEVICES` before the worker imports PyTorch), and run-directory layout/creation.
- `backends.py` — the four executors: `ForegroundBackend` (blocking, tests/short runs only), `WindowsTaskBackend` (Task Scheduler, survives terminal close), `SlurmBackend` (`sbatch`/`sbatch --test-only`), `ExternalHPCBackend` (writes `worker.sh`/`probe.sh` for manual submission through a corporate portal). `get_backend`/`detect_backend` select by profile.
- `training.py` — `ModernTrainingRunner` and `train_from_manifest`: the actual training loop driven by an immutable `RunManifest`, wraps the legacy `runners`/`models` code with resumable sampling, EMA, reproducibility seeding, and a `StopController` for graceful stop.
- `worker.py` — the scheduler-safe entry point that loads a manifest JSON and calls into `training.py`; defines the process exit-code contract (`EXIT_SUCCESS`, `EXIT_CUDA_OOM`, `EXIT_INTERRUPTED`, etc.) that backends/status inspection rely on.
- `checkpoints.py`, `run_logging.py` — atomic checkpoint save/load with fallback to the previous checkpoint on corruption; append-only `metrics.jsonl`, heartbeat, and `state.json`.
- `tracking.py` — optional, explicit, idempotent MLflow publication of a completed run bundle (`ddimctl track publish`); training itself never depends on MLflow or W&B.
- `hpc_probe.py`, `offline.py` — HPC compute-node probing (CUDA/cuDNN/GPU facts, checkpoint roundtrip) and offline dependency wheelhouse build/verify for air-gapped HPC targets.
- `cli.py` — the Typer app wiring all of the above into `machine`, `train`, `run`, `track`, `environment` subcommands; also enforces duplicate-option and single-active-config invariants at the argv level before Typer/Click parsing.

A run bundle (`<runs-root>/YYYY-MM-DD/<timestamp>__<label>__<hash>/`) is the single source of truth — TensorBoard and MLflow are just views over it. It is never overwritten; `resume` creates a new numbered `attempts/NNN/` directory and requires a valid `checkpoints/latest.json`. There are no automatic retries by design (would loop on config errors/OOM) — failures stay visible in `state.json`/backend metadata for a human to inspect before an explicit `run resume`.

## Architecture: legacy DDIM code

`main.py` parses args/config (`dict2namespace`), then hands off to `runners/diffusion.py`'s `Diffusion` class, whose methods are the actual entry points: `train`, `sample` (dispatches to `sample_fid`/`sample_sequence`/`sample_interpolation`), and `test`. `models/diffusion.py` is the U-Net; `functions/` holds the beta schedule, denoising step, and loss. `datasets/__init__.py::get_dataset` is the single dispatch point for all datasets (`CIFAR10`, `CELEBA`, `LSUN`, `FFHQ`, `SEM`) — adding a dataset means adding a branch here plus a loader module in `datasets/`.

`datasets/sem.py` (`SEMImageDataset`) loads SEM PNGs directly from a configured directory (`data.data_dir` in YAML, forward slashes even on Windows) with a deterministic train/validation split (seeded shuffle, not by filename). `ddimctl`'s `ModernTrainingRunner` reuses this same dataset/model code — it does not reimplement the diffusion math, only the orchestration around it.

## Gotchas

- **Duplicate source content = silent validation leakage.** Real image corpora
  ship the same picture under several filenames — the MIIC corpus at
  `SEMImageAI/data/datasets/paired_miic_v0/train/target_sem` has **185 unique
  images among 1050 files**, and its own nominal val/test directories overlap
  its train directory completely. Selecting or splitting by *filename/index*
  therefore puts byte-identical content on both sides (fresh noise, seen
  scene) and quietly invalidates every held-out number; this happened and cost
  a full retrain (report §9.1). `burst_diffusion` now deduplicates by content
  hash at generation (`select_unique_sources`) and splits by content group
  (`BurstCache`), with regression tests. **Any new dataset path must hash
  content before splitting** — the legacy `datasets/sem.py` + `get_dataset`
  index split does *not* do this and will leak on that corpus.
- **DataLoader worker restart trap** (see `docs/sem_dataset_migration.md`): the legacy runner creates a new DataLoader iterator every epoch. On a small dataset (few batches/epoch) with `num_workers > 0` and no `persistent_workers` support in this runner, workers get torn down and recreated almost every step — a measured 3.3s/batch vs 17ms/batch. Fix is `num_workers: 0` plus `cache_in_memory: true` on `SEMImageDataset`, not adding persistent-workers support. `cache_in_memory` cannot be combined with `random_flip` (the transform result would be cached pre-augmentation).
- The legacy runner ignores `training.n_iters`; **`n_epochs` is what controls when training stops**. Recalculate it when dataset size or desired update count changes.
- Multi-GPU: `ddimctl` isolates one GPU per run via `gpu_index` before the worker imports PyTorch (single-GPU-per-run by design, v1). The legacy `main.py` path instead wraps the model in `torch.nn.DataParallel` across every visible GPU — set `CUDA_VISIBLE_DEVICES` explicitly when benchmarking a single GPU through `main.py`.
- Never embed credentials in an MLflow tracking URI or a machine profile — `tracking.py`/schema validation rejects them; supply auth at publish time through the environment instead.
- Machine profiles, absolute local paths, and generated artifacts (`runs/`, `experiments/`, `output/`, checkpoints, datasets) must never be committed — see `.gitignore` for the full generated-artifact list.

## Coding style & conventions

Python 3.10+, four-space indent, PEP 8 (`snake_case` functions/modules, `PascalCase` classes/Pydantic models, `UPPER_SNAKE_CASE` constants). Type new public APIs; keep filesystem/process code cross-platform (this is developed on Windows but must also run on Linux HPC). Group imports stdlib / third-party / local.

Tests live in `tests/test_<area>.py`, functions named `test_*`, using `tmp_path`/`monkeypatch`/mocks for filesystem, scheduler, network, and GPU behavior — never hit real network/GPU/scheduler in a test. Every bug fix should include a regression test.

Commits favor concise, imperative subjects with `<type>: <summary>` prefixes (`feat:`, `fix:`, `docs:`). PRs should explain problem + solution, list verification commands, and call out config/compatibility changes; never attach datasets, credentials, checkpoints, or run bundles.
