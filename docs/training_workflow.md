# Reliable SEM training workflow

`ddimctl` is the supported interface for new SEM experiments. It keeps one stable recipe in `configs/sem.yml`, validates experiment choices, prints a fully explicit launch command, and records every run in a dated, portable bundle. Machine-specific paths and executor details live in per-user machine profiles, not in training configuration files.

The run bundle is the source of truth. TensorBoard and optional MLflow are views of the data in that bundle; training never depends on either MLflow or Weights & Biases.

## Install on each machine

Use Python 3.10 or newer in a dedicated environment. Install a CUDA-enabled PyTorch build that matches the machine and its driver, then install this repository:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -e .
```

On Linux:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

The H100 system has no internet access, so stage an approved wheelhouse or a prebuilt environment containing PyTorch and this project's dependencies before connecting. Do not let `pip` contact the public internet there. The `python_executable` recorded in the machine profile must exist on the compute node, not only on the login node.

To create a wheelhouse, use a connected Linux builder with the same CPU architecture, Python minor version, and intended CUDA/PyTorch build as the H100 target (a Windows-built wheelhouse is not compatible):

```sh
ddimctl environment bundle --output h100-wheelhouse --yes
```

The build uses that machine's configured pip indexes, so point pip at the corporate-approved mirror and PyTorch wheel source before running it; never place credentials in a committed command or config.

Transfer the complete directory through the approved route. Before installing on the offline side, run its standard-library verifier, then use the generated installer inside an activated environment:

```sh
python h100-wheelhouse/verify.py
cd h100-wheelhouse
sh install.sh
```

After installation, `ddimctl environment verify h100-wheelhouse` performs the same manifest/checksum audit. The bundle command never overwrites an existing directory, and the manifest records every exact wheel, the builder platform, architecture, and Python tag. Build separate wheelhouses for Windows and Linux, and rebuild whenever the target Python/CUDA/PyTorch combination changes.

MLflow is optional and only needs to be installed on the workstation that will host the local catalog:

```powershell
python -m pip install -e ".[tracking]"
```

## Configure a machine once

Profiles contain operational facts only: executor, runs directory, dataset aliases and paths, Python executable, time zone, and expected GPU. They intentionally contain no learning rate, batch size, model size, or other experiment conditions.

Run the interactive configurator on each target machine:

```text
ddimctl machine configure --id local-4060ti
ddimctl machine configure --id corp-a6000
ddimctl machine configure --id h100
```

Recommended answers are:

| Target | Executor | Expected GPU | Important paths |
| --- | --- | --- | --- |
| Local Windows PC | `windows-task` | `RTX 4060 Ti` | A local runs directory and the local SEM image directory |
| Corporate Windows workstation | `windows-task` | `RTX A6000` | Workstation-local or approved shared paths |
| H100 cluster with Slurm CLI access | `slurm` | `H100` | Paths visible from compute nodes; enter the site's partition/account/QoS/resources |
| H100 through a proprietary portal | `external-hpc` | `H100` | Paths visible from the portal's compute job |

Use dataset alias `sem` unless there is a concrete reason to introduce another alias. For the nested-SSH system, run configuration in the actual target login environment; paths from the local PC or an intermediate jump host are not usable by a compute job.

Profiles are stored outside the repository. On Windows they are under `%APPDATA%\ddimctl\profiles`; on Linux they are under `$XDG_CONFIG_HOME/ddimctl/profiles`, or `~/.config/ddimctl/profiles` when `XDG_CONFIG_HOME` is unset.

Use `ddimctl machine list` to see configured IDs and `ddimctl machine show <id>` to review the exact operational values. Re-running `configure` asks before replacing an existing profile.

## Verify the target before training

The doctor validates Python, dataset and runs paths, a real CUDA allocation/kernel, the expected GPU name, and the selected executor:

```text
ddimctl doctor --machine local-4060ti --exercise-executor
ddimctl doctor --machine corp-a6000 --exercise-executor
ddimctl doctor --machine h100 --exercise-executor --export-hpc-probe h100-login-probe.json
```

`--exercise-executor` performs a harmless Windows Task Scheduler acceptance run on Windows. With a Slurm profile, it renders the requested resources and calls `sbatch --test-only`, which validates the request without submitting a job. Use it before the first long experiment and after relevant IT or scheduler-policy changes.

An HPC login node commonly has no GPU, so a doctor run there may intentionally report CUDA as failed. Probe the real compute context through an allocation or the approved portal:

```sh
python -m ddimctl.hpc_probe --output h100-compute-probe.json \
  --path dataset=/path/visible/on/compute/node \
  --path runs=/path/visible/on/compute/node
```

An `external-hpc` launch also creates `probe.sh` beside `worker.sh`; submit the probe through the same route as a training job when that is the only way to reach a compute node. Keep the JSON report and its `.sha256` sidecar with the deployment notes.

The compute probe performs a real CUDA forward/backward operation and a checkpoint save/load roundtrip in addition to recording Torch, CUDA, cuDNN, GPU model/memory, scheduler variables, container-runtime availability, paths, and free space.

## Plan and launch

For the lowest-error interactive path, use the wizard. It validates each high-impact setting, shows the complete resolved plan and canonical PowerShell/POSIX commands, then asks for confirmation:

```text
ddimctl train wizard --machine local-4060ti
```

For automation or review, run `plan` first. It performs no writes and launches nothing:

```text
ddimctl train plan --machine local-4060ti --config configs/sem.yml --label sem32-baseline --dataset sem --image-size 32 --model-ch 64 --ch-mult 1,2,2,2 --diffusion-steps 100 --beta-start 0.001 --beta-end 0.2 --ema-rate 0.999 --max-steps 20000 --batch-size 7 --learning-rate 0.0002 --checkpoint-every 2500 --checkpoint-minutes 30 --validation-every 2500 --sample-every 2500 --seed 1234 --reproducibility seeded --num-workers 0 --no-cache-in-memory
```

Review the displayed plan, especially dataset, GPU, estimated checkpoint size, intervals, seed, and runs directory. Copy the fully explicit canonical `ddimctl train launch ...` command printed by `plan`; add `--yes` only for an already-reviewed noninteractive submission. Duplicate scalar options are rejected rather than silently taking the last value.

Executor behavior is machine-specific:

- `foreground` blocks the current terminal and is intended only for short tests.
- `windows-task` registers and starts an on-demand Task Scheduler job, protecting it from accidental terminal closure. It does not protect against shutdown, sleep, logout restrictions, or site IT policy; the acceptance probe is the authority for that workstation.
- `slurm` retains `attempts/NNN/job.sbatch`, submits it with `sbatch`, and records the job ID.
- `external-hpc` writes `worker.sh` and `probe.sh` and leaves the run in `prepared`. Submit `worker.sh` through the approved corporate scheduler or portal; `ddimctl` cannot infer that site-specific step.

Version 1 intentionally supports one training process and exactly one visible CUDA GPU. Request one scheduler GPU. If an environment exposes several GPUs, restrict visibility using the site's supported mechanism before launching; multi-GPU/DDP training is not implemented.

## What a run contains

Runs are never overwritten. Their paths include both local date and a precise timestamp:

```text
<runs-root>/YYYY-MM-DD/YYYYMMDDTHHMMSS+ZZZZ__<label>__<hash>/
```

Important contents are:

- `manifest.json`, `argv.json`, `resolved_config.yml`, `command.sh`, and `command.ps1`: exact machine, dataset fingerprint, settings, and replayable command.
- `source.tar.gz` and `source/`: a checksummed snapshot of tracked and non-ignored untracked source, including dirty working-tree changes. The worker runs the verified snapshot, so later checkout edits cannot change a queued run.
- `dataset.json` and `environment.json`: dataset content fingerprint and launch-host information.
- `state.json` and `metrics.jsonl`: authoritative state and append-only metrics.
- `tensorboard/`, `samples/`, and `checkpoints/`: visualizations, fixed-noise samples, and atomic latest/best recovery data.
- `attempts/001/`, `attempts/002/`, and so on: stdout, stderr, raw worker argv/commands, launch request, backend metadata, and retained `windows-task.xml` or `job.sbatch` definitions for the original attempt and each resume.
- Top-level executor conveniences such as `worker.sh`/`probe.sh` for an `external-hpc` portal, plus `backend.json` with the exact submission identity.

Use a new descriptive `--label` for a materially different question. Do not copy and edit `configs/sem.yml` into experiment-specific variants; keep fixed recipe choices there and put varying choices in the command.

## Operate and recover runs

All run-management commands take the run bundle path:

```text
ddimctl run status <run>
ddimctl run logs <run> --lines 120
ddimctl run logs <run> --stream stdout --follow
ddimctl run stop <run>
ddimctl run resume <run>
```

`stop` writes a graceful stop request so the worker can checkpoint and exit. Use `ddimctl run stop <run> --force` only when graceful stopping cannot work; it asks the executor to cancel the job and may lose work since the last checkpoint. `resume` requires a valid `checkpoints/latest.json`, refuses to resume an active run, preserves the original manifest/settings, and creates a new numbered attempt. Add `--yes` only for a reviewed noninteractive resume.

There are no blind automatic retries: configuration errors and out-of-memory failures would otherwise loop. Slurm `TIMEOUT`, preemption, scheduler exit code/signal, Task Scheduler failure, stale heartbeat, and worker exceptions remain visible in state/backend metadata; inspect them, correct only operational causes, then resume the unchanged run explicitly.

For live scalar and image inspection:

```text
tensorboard --logdir <run>/tensorboard
```

On the isolated HPC, either use an approved multi-hop port-forwarding method or transfer the complete bundle to a workstation and point TensorBoard there. `metrics.jsonl` remains usable even if TensorBoard was unavailable during training.

## Local MLflow without W&B

Host MLflow on a machine where a browser is permitted. The built-in launcher binds only to loopback:

```text
ddimctl track serve --data-dir D:\ddim-mlflow --port 5000
```

Then publish a copied run bundle:

```text
ddimctl track publish <run> --tracking-uri http://127.0.0.1:5000 --experiment ddim-sem
```

Publication is explicit, idempotent, and separate from training failure handling. It replays parameters, metrics, metadata, source/command artifacts, and samples into MLflow; checkpoints stay in the portable bundle. The corporate A6000 and offline H100 do not need browser access, an MLflow client during training, a W&B account, or a W&B network connection.

Do not embed credentials in an MLflow URI or machine profile; the schema/publisher rejects them. If an internal service requires authentication, supply it at publication time through the corporate-approved environment or secret mechanism so it never enters the run bundle.

For offline transfer, wait for completion or a clean graceful stop, then copy the entire dated run directory through the approved corporate route without flattening or renaming its contents. Publish only after the copy is complete. Retain the original bundle even after publication because it is the authoritative recovery and audit artifact.

## Site-specific acceptance checklist

Resolve these items with the workstation/HPC administrators before a long run:

- Whether Windows Task Scheduler jobs may run for the required duration and what logout, sleep, patching, and antivirus policies apply.
- Whether the H100 site exposes Slurm commands or requires `external-hpc`, and the exact portal submission procedure.
- Slurm partition, account, QoS, wall-time, CPU, memory, and GPU request values.
- Compute-node-visible repository, dataset, runs, Python environment, and scratch paths.
- Required environment modules, container policy, storage quota, checkpoint size limits, and job preemption/signalling behavior.
- The approved route for transferring source/dependencies inward and complete run bundles outward through the nested SSH boundary.

Do not guess these values in a training command. Record the confirmed operational values in the machine profile, rerun `doctor`/the compute probe, and perform a short end-to-end run before committing H100 time.

## Legacy entry point

`python main.py` remains available for historical sampling and compatibility, but it emits a deprecation warning and is not the supported interface for new SEM training. It does not provide the typed planning, immutable bundles, source snapshot, executor integration, or portable tracking workflow described here.
