import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import TensorDataset

from ddimctl.checkpoints import build_checkpoint, load_checkpoint, save_checkpoint
from ddimctl.bundles import fingerprint_dataset
from ddimctl.run_logging import MetricLogger, StateStore, read_json
from ddimctl.training import ModernTrainingRunner, StopController, select_device
from ddimctl.worker import EXIT_USAGE, run_worker, verify_dataset_fingerprint


class TinyNoiseModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.bias = torch.nn.Parameter(torch.tensor(-0.1))

    def forward(self, images, timesteps):
        return images * self.scale + self.bias + timesteps[:, None, None, None] * 0.001


def tiny_spec(max_steps=4):
    return {
        "label": "tiny",
        "dataset_alias": "sem",
        "image_size": 4,
        "channels": 1,
        "model_type": "simple",
        "diffusion_steps": 4,
        "beta_schedule": "linear",
        "beta_start": 0.001,
        "beta_end": 0.02,
        "batch_size": 2,
        "max_steps": max_steps,
        "checkpoint_every": 2,
        "validation_every": 2,
        "sample_every": 0,
        "checkpoint_minutes": 10_000,
        "lr": 0.001,
        "grad_clip": 1.0,
        "seed": 2468,
        "reproducibility": "strict",
        "num_workers": 0,
        "ema": True,
        "ema_rate": 0.9,
    }


def tiny_datasets(args, config):
    values = torch.linspace(0.05, 0.8, steps=8 * 4 * 4).reshape(8, 1, 4, 4)
    labels = torch.zeros(8, dtype=torch.long)
    return (
        TensorDataset(values[:6].clone(), labels[:6].clone()),
        TensorDataset(values[6:].clone(), labels[6:].clone()),
    )


class RunLoggingTests(unittest.TestCase):
    def test_state_updates_are_atomic_and_preserve_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            state = StateStore(Path(directory) / "state.json", {"status": "prepared"})
            state.update(status="running", global_step=3)
            state.update(heartbeat_at="now")
            loaded = read_json(Path(directory) / "state.json")
            self.assertEqual(loaded["status"], "running")
            self.assertEqual(loaded["global_step"], 3)
            self.assertEqual(loaded["heartbeat_at"], "now")


class CheckpointTests(unittest.TestCase):
    def _payload(self, step):
        model = TinyNoiseModel(None)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return build_checkpoint(
            model=model,
            optimizer=optimizer,
            ema_state=None,
            global_step=step,
            epoch=0,
            batch_in_epoch=step,
            sampler_state={"epoch": 0, "start_index": step * 2},
            config_sha256="abc",
            run_id="run",
        )

    def test_corrupt_latest_falls_back_to_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_checkpoint(root, self._payload(1), milestone=False)
            latest = save_checkpoint(root, self._payload(2), milestone=False)
            latest.write_bytes(b"corrupt")

            payload, selected = load_checkpoint(root)
            self.assertEqual(payload["global_step"], 1)
            self.assertEqual(selected, first)


class ModernTrainingTests(unittest.TestCase):
    def test_implicit_device_refuses_accidental_cpu_training(self):
        with mock.patch("ddimctl.training.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU-bound"):
                select_device()
            self.assertEqual(select_device("cpu").type, "cpu")

    def test_worker_retains_single_isolated_gpu_invariant(self):
        with mock.patch("ddimctl.training.torch.cuda.is_available", return_value=True), mock.patch(
            "ddimctl.training.torch.cuda.device_count", return_value=4
        ):
            with self.assertRaisesRegex(RuntimeError, "expected one isolated CUDA GPU"):
                select_device()

    def test_worker_rejects_dataset_changed_after_bundle_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"original")
            manifest = {
                "training": {"extensions": [".png"], "recursive": False},
                "dataset": fingerprint_dataset(root, [".png"]).model_dump(mode="json"),
            }
            verify_dataset_fingerprint(manifest)
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "dataset no longer matches"):
                verify_dataset_fingerprint(manifest)

    def _runner(self, root, run_id, stop=None, progress=None):
        metrics = MetricLogger(Path(root) / "metrics.jsonl", None)
        runner = ModernTrainingRunner(
            spec=tiny_spec(),
            dataset_path=root,
            run_dir=root,
            run_id=run_id,
            config_sha256="same-config",
            device="cpu",
            stop_controller=stop,
            metric_logger=metrics,
            progress_callback=progress,
            model_factory=TinyNoiseModel,
            dataset_factory=tiny_datasets,
        )
        return runner, metrics

    def test_exact_max_steps_and_resume_match_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as interrupted_dir, tempfile.TemporaryDirectory() as full_dir:
            stop = StopController(Path(interrupted_dir) / "stop.request")

            def stop_at_two(progress):
                if progress["global_step"] == 2:
                    stop.request("test stop")

            first_runner, first_metrics = self._runner(
                interrupted_dir, "resumed", stop=stop, progress=stop_at_two
            )
            try:
                first = first_runner.run()
            finally:
                first_metrics.close()
            self.assertEqual(first.status, "interrupted")
            self.assertEqual(first.global_step, 2)

            resumed_runner, resumed_metrics = self._runner(interrupted_dir, "resumed")
            try:
                resumed = resumed_runner.run(resume=Path(interrupted_dir) / "checkpoints")
            finally:
                resumed_metrics.close()
            self.assertEqual(resumed.status, "completed")
            self.assertEqual(resumed.global_step, 4)

            full_runner, full_metrics = self._runner(full_dir, "full")
            try:
                full = full_runner.run()
            finally:
                full_metrics.close()
            self.assertEqual(full.global_step, 4)

            resumed_payload, _ = load_checkpoint(Path(interrupted_dir) / "checkpoints")
            full_payload, _ = load_checkpoint(Path(full_dir) / "checkpoints")
            for name, value in resumed_payload["model_state"].items():
                self.assertTrue(torch.equal(value, full_payload["model_state"][name]), name)

    def test_worker_returns_nonzero_for_missing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(run_worker(directory), EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
