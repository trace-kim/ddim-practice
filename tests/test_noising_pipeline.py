from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from noising_pipeline import create_noisy_dataset
from noising_pipeline import pipeline


def _save_image(path: Path, array: np.ndarray, *, format: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format=format)


def _read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def _manifest_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def _rgba_png_bytes(value: int) -> bytes:
    stream = io.BytesIO()
    rgba = np.full((3, 4, 4), value, dtype=np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba).save(stream, format="PNG")
    return stream.getvalue()


def _bbbc038_zip(*, unsafe_name: str | None = None) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("b/images/b.png", _rgba_png_bytes(200))
        archive.writestr("a/images/a.png", _rgba_png_bytes(100))
        archive.writestr("a/masks/mask.png", _rgba_png_bytes(255))
        archive.writestr("__MACOSX/a/images/._a.png", b"metadata")
        archive.writestr("notes.txt", b"ignored")
        if unsafe_name is not None:
            archive.writestr(unsafe_name, b"unsafe")
    return stream.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.payload), max(1, chunk_size // 2)):
            yield self.payload[start : start + max(1, chunk_size // 2)]

    def close(self) -> None:
        self.closed = True


def test_creates_clean_and_noisy_files_and_manifest_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _save_image(source / "nested" / "b.PNG", np.full((5, 7), 80, dtype=np.uint8))
    _save_image(source / "a.tif", np.full((4, 6), 3000, dtype=np.uint16))
    (source / "ignored.txt").write_text("not an image", encoding="utf-8")

    manifest = create_noisy_dataset(
        source,
        tmp_path / "result",
        n=3,
        steps=1,
        noise_type="GAUSSIAN",
        seed=12,
        source_license="CC BY 4.0",
    )
    rows = _manifest_rows(manifest)

    assert manifest == tmp_path / "result" / "manifest.jsonl"
    assert len(list((tmp_path / "result" / "clean").glob("*.png"))) == 2
    assert len(list((tmp_path / "result" / "noisy").glob("*.png"))) == 6
    assert len(rows) == 6
    assert [row["source_path"] for row in rows[::3]] == ["a.tif", "nested/b.PNG"]
    assert [row["replica_index"] for row in rows[:3]] == [0, 1, 2]
    assert {row["clean_path"] for row in rows[:3]} == {"clean/00000.png"}
    assert rows[0]["dataset"] == "local"
    assert rows[0]["license"] == "CC BY 4.0"
    assert rows[0]["noise_types"] == ["gaussian"]


def test_reproducible_outputs_have_independent_replicas(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _save_image(source / "image.png", np.full((24, 20), 128, dtype=np.uint8))

    first_manifest = create_noisy_dataset(
        source, tmp_path / "first", 3, 2, "gaussian", seed=99
    )
    second_manifest = create_noisy_dataset(
        source, tmp_path / "second", 3, 2, "gaussian", seed=99
    )

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    first_images = sorted((tmp_path / "first" / "noisy").glob("*.png"))
    second_images = sorted((tmp_path / "second" / "noisy").glob("*.png"))
    assert [path.read_bytes() for path in first_images] == [
        path.read_bytes() for path in second_images
    ]
    assert len({path.read_bytes() for path in first_images}) == 3
    rows = _manifest_rows(first_manifest)
    assert len({row["sample_seed"] for row in rows}) == 3


def test_fused_steps_apply_each_noise_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _save_image(source / "image.png", np.full((4, 4), 64, dtype=np.uint8))
    calls: list[tuple[str, dict[str, float]]] = []
    original = pipeline._apply_noise

    def recording_apply(image, noise_type, params, rng):
        calls.append((noise_type, dict(params)))
        return original(image, noise_type, params, rng)

    monkeypatch.setattr(pipeline, "_apply_noise", recording_apply)
    manifest = create_noisy_dataset(
        source,
        tmp_path / "result",
        n=1,
        steps=3,
        noise_type=["Gaussian", "POISSON"],
        noise_params={
            "GAUSSIAN": {"mean": 0.0, "std": 0.0},
            "poisson": {"peak": 250.0},
        },
    )

    assert [name for name, _ in calls] == ["gaussian", "poisson"]
    assert calls[0][1] == {"mean": 0.0, "std": 0.0}
    assert calls[1][1] == {"peak": pytest.approx(250.0 / 3.0)}
    row = _manifest_rows(manifest)[0]
    assert row["noise_types"] == ["gaussian", "poisson"]
    assert row["step_mode"] == "fused"
    assert row["noise_params"] == {
        "gaussian": {"mean": 0.0, "std": 0.0},
        "poisson": {"peak": 250.0},
    }
    assert row["effective_noise_params"]["poisson"]["peak"] == pytest.approx(
        250.0 / 3.0
    )


def test_iterative_mode_retains_per_step_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _save_image(source / "image.png", np.full((4, 4), 64, dtype=np.uint8))
    calls: list[str] = []
    original = pipeline._apply_noise

    def recording_apply(image, noise_type, params, rng):
        calls.append(noise_type)
        return original(image, noise_type, params, rng)

    monkeypatch.setattr(pipeline, "_apply_noise", recording_apply)
    manifest = create_noisy_dataset(
        source,
        tmp_path / "result",
        n=1,
        steps=3,
        noise_type=["gaussian", "poisson"],
        noise_params={
            "gaussian": {"mean": 0.0, "std": 0.0},
            "poisson": {"peak": 250.0},
        },
        step_mode="iterative",
    )

    assert calls == ["gaussian", "poisson"] * 3
    assert _manifest_rows(manifest)[0]["step_mode"] == "iterative"


def test_fused_parameter_math() -> None:
    fused = pipeline._fuse_noise_params(
        {
            "gaussian": {"mean": 0.1, "std": 0.2},
            "poisson": {"peak": 100.0},
            "salt_pepper": {"amount": 0.1, "salt_ratio": 0.4},
        },
        steps=4,
    )

    assert fused["gaussian"] == {
        "mean": pytest.approx(0.4),
        "std": pytest.approx(0.4),
    }
    assert fused["poisson"] == {"peak": pytest.approx(25.0)}
    assert fused["salt_pepper"] == {
        "amount": pytest.approx(1.0 - 0.9**4),
        "salt_ratio": pytest.approx(0.4),
    }


def test_progress_reports_counts_rate_and_eta(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    _save_image(source / "a.png", np.zeros((2, 2), dtype=np.uint8))
    _save_image(source / "b.png", np.ones((2, 2), dtype=np.uint8))

    create_noisy_dataset(
        source,
        tmp_path / "result",
        n=2,
        steps=50,
        noise_type="gaussian",
        progress_interval=60.0,
    )
    progress_output = capsys.readouterr().err

    assert "[noising_pipeline] Starting 2 clean images x 2 replicas" in progress_output
    assert "1/4 noisy images" in progress_output
    assert "images/s" in progress_output
    assert "ETA" in progress_output
    assert "Completed 4/4 noisy images" in progress_output


def test_progress_can_be_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source"
    _save_image(source / "image.png", np.zeros((2, 2), dtype=np.uint8))
    create_noisy_dataset(
        source,
        tmp_path / "result",
        1,
        1,
        "gaussian",
        progress=False,
    )

    assert capsys.readouterr().err == ""


def test_steps_accumulate_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _save_image(source / "black.png", np.zeros((3, 5), dtype=np.uint8))
    create_noisy_dataset(
        source,
        tmp_path / "result",
        1,
        2,
        "gaussian",
        noise_params={"gaussian": {"mean": 0.1, "std": 0.0}},
    )

    actual = _read_image(tmp_path / "result" / "noisy" / "00000_00000.png")
    np.testing.assert_array_equal(actual, np.full((3, 5), 51, dtype=np.uint8))


def test_noise_combinations_follow_supplied_order(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _save_image(source / "gray.png", np.full((4, 5), 128, dtype=np.uint8))
    params = {
        "gaussian": {"mean": -0.25, "std": 0.0},
        "salt_pepper": {"amount": 1.0, "salt_ratio": 1.0},
    }
    create_noisy_dataset(
        source,
        tmp_path / "salt_first",
        1,
        1,
        ["salt_pepper", "gaussian"],
        noise_params=params,
    )
    create_noisy_dataset(
        source,
        tmp_path / "salt_last",
        1,
        1,
        ["gaussian", "salt_pepper"],
        noise_params=params,
    )

    salt_first = _read_image(
        tmp_path / "salt_first" / "noisy" / "00000_00000.png"
    )
    salt_last = _read_image(
        tmp_path / "salt_last" / "noisy" / "00000_00000.png"
    )
    np.testing.assert_array_equal(salt_first, np.full((4, 5), 191, dtype=np.uint8))
    np.testing.assert_array_equal(salt_last, np.full((4, 5), 255, dtype=np.uint8))


def test_gaussian_poisson_and_salt_pepper_behavior(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _save_image(source / "gray.png", np.full((32, 32), 128, dtype=np.uint8))

    create_noisy_dataset(
        source,
        tmp_path / "gaussian",
        1,
        1,
        "gaussian",
        noise_params={"gaussian": {"mean": 0.25, "std": 0.0}},
    )
    gaussian = _read_image(
        tmp_path / "gaussian" / "noisy" / "00000_00000.png"
    )
    np.testing.assert_array_equal(gaussian, np.full((32, 32), 192, dtype=np.uint8))

    create_noisy_dataset(
        source,
        tmp_path / "poisson",
        1,
        1,
        "poisson",
        noise_params={"poisson": {"peak": 2.0}},
    )
    poisson = _read_image(tmp_path / "poisson" / "noisy" / "00000_00000.png")
    assert not np.array_equal(poisson, np.full((32, 32), 128, dtype=np.uint8))
    assert np.unique(poisson).size > 1

    for name, ratio, expected in (("salt", 1.0, 255), ("pepper", 0.0, 0)):
        create_noisy_dataset(
            source,
            tmp_path / name,
            1,
            1,
            "salt_pepper",
            noise_params={
                "salt_pepper": {"amount": 1.0, "salt_ratio": ratio}
            },
        )
        actual = _read_image(tmp_path / name / "noisy" / "00000_00000.png")
        np.testing.assert_array_equal(
            actual, np.full((32, 32), expected, dtype=np.uint8)
        )


def test_preserves_16_bit_grayscale_and_8_bit_rgb_without_resizing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    gray16 = np.array(
        [[0, 1, 255, 256], [1024, 32767, 32768, 65535], [9, 99, 999, 9999]],
        dtype=np.uint16,
    )
    rgb8 = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    _save_image(source / "a.tif", gray16)
    _save_image(source / "b.png", rgb8)

    manifest = create_noisy_dataset(
        source,
        tmp_path / "result",
        1,
        1,
        "gaussian",
        noise_params={"gaussian": {"mean": 0.0, "std": 0.0}},
    )

    for index, expected in enumerate((gray16, rgb8)):
        clean = _read_image(tmp_path / "result" / "clean" / f"{index:05d}.png")
        noisy = _read_image(
            tmp_path / "result" / "noisy" / f"{index:05d}_00000.png"
        )
        np.testing.assert_array_equal(clean, expected)
        np.testing.assert_array_equal(noisy, expected)
    rows = _manifest_rows(manifest)
    assert [(row["shape"], row["bit_depth"]) for row in rows] == [
        ([3, 4], 16),
        ([5, 7, 3], 8),
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n": 0}, "n must be"),
        ({"n": 1.5}, "n must be"),
        ({"steps": 0}, "steps must be"),
        ({"seed": -1}, "seed must be"),
        ({"overwrite": "yes"}, "overwrite must be"),
        ({"step_mode": "unknown"}, "step_mode must be"),
        ({"progress": "yes"}, "progress must be"),
        ({"progress_interval": 0}, "progress_interval must be"),
        ({"download": "unsupported"}, "download must be None or 'bbbc038'"),
        ({"noise_type": "speckle"}, "unknown noise type"),
        ({"noise_type": []}, "at least one"),
        (
            {"noise_params": {"gaussian": {"std": -0.1}}},
            "gaussian.std",
        ),
        ({"noise_type": "poisson", "noise_params": {"poisson": {"peak": 0}}}, "poisson.peak"),
        (
            {
                "noise_type": "salt_pepper",
                "noise_params": {"salt_pepper": {"amount": 1.1}},
            },
            "salt_pepper.amount",
        ),
        (
            {"noise_params": {"gaussian": {"variance": 1.0}}},
            "unknown gaussian parameter",
        ),
        (
            {"noise_params": {"poisson": {"peak": 10.0}}},
            "unselected noise type",
        ),
    ],
)
def test_rejects_invalid_counts_names_and_parameters(
    tmp_path: Path, kwargs: dict[str, object], match: str
) -> None:
    arguments: dict[str, object] = {
        "source_dir": tmp_path / "missing-source",
        "output_dir": tmp_path / "output",
        "n": 1,
        "steps": 1,
        "noise_type": "gaussian",
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        create_noisy_dataset(**arguments)


def test_rejects_empty_source_and_overlapping_paths(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no supported images"):
        create_noisy_dataset(empty, tmp_path / "empty-output", 1, 1, "gaussian")
    assert not (tmp_path / "empty-output").exists()

    source = tmp_path / "source"
    _save_image(source / "image.png", np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="must not overlap"):
        create_noisy_dataset(source, source / "generated", 1, 1, "gaussian")
    with pytest.raises(ValueError, match="must not overlap"):
        create_noisy_dataset(source, tmp_path, 1, 1, "gaussian")


def test_existing_output_prompts_and_decline_preserves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _save_image(source / "image.png", np.zeros((2, 2), dtype=np.uint8))
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    prompts: list[str] = []

    def decline(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", decline)
    with pytest.raises(FileExistsError, match="overwrite declined"):
        create_noisy_dataset(source, existing, 1, 1, "gaussian")

    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert len(prompts) == 1
    assert str(existing) in prompts[0]
    assert "[y/N]" in prompts[0]


def test_existing_output_confirmed_yes_reuses_exact_requested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
    _save_image(source / "image.png", expected)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "old-result.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda prompt: "YES")

    manifest = create_noisy_dataset(
        source,
        existing,
        1,
        1,
        "gaussian",
        noise_params={"gaussian": {"mean": 0.0, "std": 0.0}},
    )

    assert manifest == existing / "manifest.jsonl"
    assert not (existing / "old-result.txt").exists()
    np.testing.assert_array_equal(
        _read_image(existing / "clean" / "00000.png"), expected
    )
    assert not list(tmp_path.glob(".existing.staging-*"))
    assert not list(tmp_path.glob(".existing.backup-*"))


def test_overwrite_flags_skip_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    _save_image(source / "image.png", np.zeros((2, 2), dtype=np.uint8))
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "old.txt").write_text("old", encoding="utf-8")

    def unexpected_prompt(prompt: str) -> str:
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr("builtins.input", unexpected_prompt)
    create_noisy_dataset(source, existing, 1, 1, "gaussian", overwrite=True)
    marker = existing / "preserve.txt"
    marker.write_text("new", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        create_noisy_dataset(source, existing, 1, 1, "gaussian", overwrite=False)
    assert marker.read_text(encoding="utf-8") == "new"


def test_failed_confirmed_overwrite_leaves_partial_output_at_requested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    rgba = np.full((3, 4, 4), 100, dtype=np.uint8)
    rgba[..., 3] = 10
    _save_image(source / "invalid.png", rgba)
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("old output", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    with pytest.raises(ValueError, match="non-opaque alpha"):
        create_noisy_dataset(source, existing, 1, 1, "gaussian")

    assert not sentinel.exists()
    assert existing.is_dir()
    assert (existing / "clean").is_dir()
    assert (existing / "noisy").is_dir()
    assert (existing / "manifest.jsonl").read_text(encoding="utf-8") == ""
    assert not list(tmp_path.glob(".existing.staging-*"))


def test_interrupted_overwrite_preserves_completed_images_and_manifest_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _save_image(source / "image.png", np.full((8, 8), 100, dtype=np.uint8))
    existing = tmp_path / "existing"
    existing.mkdir()
    old_file = existing / "old.txt"
    old_file.write_text("old output", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    original_save = pipeline._save_png
    noisy_save_count = 0

    def interrupt_before_fourth_noisy(array, bit_depth, path):
        nonlocal noisy_save_count
        if path.parent.name == "noisy":
            noisy_save_count += 1
            if noisy_save_count == 4:
                raise KeyboardInterrupt
        return original_save(array, bit_depth, path)

    monkeypatch.setattr(pipeline, "_save_png", interrupt_before_fourth_noisy)
    with pytest.raises(KeyboardInterrupt):
        create_noisy_dataset(
            source,
            existing,
            n=10,
            steps=20,
            noise_type="gaussian",
            progress=False,
        )

    assert not old_file.exists()
    assert existing.is_dir()
    assert len(list((existing / "clean").glob("*.png"))) == 1
    assert len(list((existing / "noisy").glob("*.png"))) == 3
    rows = _manifest_rows(existing / "manifest.jsonl")
    assert len(rows) == 3
    assert [row["replica_index"] for row in rows] == [0, 1, 2]
    assert not list(tmp_path.glob(".existing.staging-*"))
    assert not list(tmp_path.glob(".existing.backup-*"))


def test_refuses_to_overwrite_current_working_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _save_image(source / "image.png", np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="protected directory"):
        create_noisy_dataset(
            source, Path.cwd(), 1, 1, "gaussian", overwrite=True
        )


def test_rejects_multiframe_tiff(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    frames = [Image.fromarray(np.full((3, 4), value, dtype=np.uint16)) for value in (1, 2)]
    frames[0].save(source / "stack.tif", save_all=True, append_images=frames[1:])

    with pytest.raises(ValueError, match="multi-frame"):
        create_noisy_dataset(source, tmp_path / "output", 1, 1, "gaussian")


def test_rejects_nonopaque_alpha(tmp_path: Path) -> None:
    source = tmp_path / "source"
    rgba = np.full((3, 4, 4), 100, dtype=np.uint8)
    rgba[..., 3] = 254
    _save_image(source / "transparent.png", rgba)

    with pytest.raises(ValueError, match="non-opaque alpha"):
        create_noisy_dataset(source, tmp_path / "output", 1, 1, "gaussian")


def test_bbbc038_download_checksum_extraction_and_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _bbbc038_zip()
    response = _FakeResponse(payload)
    calls: list[tuple[object, ...]] = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return response

    monkeypatch.setattr(pipeline, "BBBC038_IMAGE_COUNT", 2)
    monkeypatch.setattr(pipeline, "BBBC038_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(pipeline.requests, "get", fake_get)
    source = tmp_path / "download"

    first_manifest = create_noisy_dataset(
        source,
        tmp_path / "first",
        1,
        1,
        "gaussian",
        download="bbbc038",
        noise_params={"gaussian": {"mean": 0.0, "std": 0.0}},
    )
    second_manifest = create_noisy_dataset(
        source,
        tmp_path / "second",
        1,
        1,
        "gaussian",
        download="bbbc038",
        noise_params={"gaussian": {"mean": 0.0, "std": 0.0}},
    )

    assert calls == [(pipeline.BBBC038_URL, True, (10, 120))]
    assert response.closed
    assert sorted(
        path.relative_to(source).as_posix()
        for path in source.glob("*/images/*.png")
    ) == [
        "a/images/a.png",
        "b/images/b.png",
    ]
    assert not (source / "__MACOSX").exists()
    assert not (source / "a" / "masks").exists()
    assert (source / pipeline._BBBC038_ARCHIVE_NAME).read_bytes() == payload
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    rows = _manifest_rows(first_manifest)
    assert [row["source_path"] for row in rows] == [
        "a/images/a.png",
        "b/images/b.png",
    ]
    assert all(row["shape"] == [3, 4, 3] for row in rows)
    assert all(row["dataset"] == "BBBC038v1-stage1-train" for row in rows)
    assert all(row["license"] == "CC0" for row in rows)
    np.testing.assert_array_equal(
        _read_image(tmp_path / "first" / "clean" / "00000.png"),
        np.full((3, 4, 3), 100, dtype=np.uint8),
    )


def test_bbbc038_download_rejects_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _bbbc038_zip()
    response = _FakeResponse(payload)
    monkeypatch.setattr(pipeline, "BBBC038_SHA256", "0" * 64)
    monkeypatch.setattr(pipeline.requests, "get", lambda *args, **kwargs: response)
    source = tmp_path / "download"

    with pytest.raises(ValueError, match="checksum mismatch"):
        create_noisy_dataset(
            source,
            tmp_path / "output",
            1,
            1,
            "gaussian",
            download="bbbc038",
        )

    assert response.closed
    assert not (source / pipeline._BBBC038_ARCHIVE_NAME).exists()
    assert not (source / (pipeline._BBBC038_ARCHIVE_NAME + ".part")).exists()
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape.png", "images/../../escape.png", "C:/escape.png", "/escape.png"],
)
def test_bbbc038_download_rejects_unsafe_zip_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_name: str
) -> None:
    payload = _bbbc038_zip(unsafe_name=unsafe_name)
    monkeypatch.setattr(pipeline, "BBBC038_IMAGE_COUNT", 2)
    monkeypatch.setattr(pipeline, "BBBC038_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        pipeline.requests, "get", lambda *args, **kwargs: _FakeResponse(payload)
    )

    with pytest.raises(ValueError, match="unsafe path"):
        create_noisy_dataset(
            tmp_path / "download",
            tmp_path / "output",
            1,
            1,
            "gaussian",
            download="bbbc038",
        )

    assert not (tmp_path / "escape.tif").exists()
    assert not (tmp_path / "output").exists()
