import pytest

from main import distributed_process_info
from runners.diffusion import per_rank_batch_size


def test_distributed_process_info_reads_torchrun_environment() -> None:
    assert distributed_process_info(
        {"RANK": "2", "LOCAL_RANK": "2", "WORLD_SIZE": "4"}
    ) == (2, 2, 4)


def test_distributed_process_info_accepts_torchrun_local_rank_argument() -> None:
    assert distributed_process_info(
        {"RANK": "1", "LOCAL_RANK": "0", "WORLD_SIZE": "4"},
        local_rank_override=1,
    ) == (1, 1, 4)


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        ({"RANK": "4", "LOCAL_RANK": "0", "WORLD_SIZE": "4"}, "RANK"),
        ({"RANK": "0", "LOCAL_RANK": "-1", "WORLD_SIZE": "4"}, "LOCAL_RANK"),
        ({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "0"}, "WORLD_SIZE"),
        ({"RANK": "zero", "LOCAL_RANK": "0", "WORLD_SIZE": "4"}, "integers"),
    ),
)
def test_distributed_process_info_rejects_invalid_values(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        distributed_process_info(environment)


def test_per_rank_batch_size_preserves_global_batch() -> None:
    assert per_rank_batch_size(192, 4) == 48


def test_per_rank_batch_size_requires_even_partition() -> None:
    with pytest.raises(ValueError, match="divisible"):
        per_rank_batch_size(191, 4)
