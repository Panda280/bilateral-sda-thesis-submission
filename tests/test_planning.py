import numpy as np
import pytest

from sda import plan_sample_execution


def test_plan_uses_largest_whole_sample_chunk_within_budget():
    one = plan_sample_execution(
        emitters=4, consumers=2, samples=10, dtype=np.float64, budget_bytes=10_000
    ).bytes_per_sample
    plan = plan_sample_execution(
        emitters=4, consumers=2, samples=10, dtype=np.float64, budget_bytes=one * 3 + one // 2
    )

    assert plan.chunk_size == 3
    assert plan.chunks == 4
    assert plan.estimated_peak_bytes == one * 3
    assert plan.estimated_peak_bytes <= plan.budget_bytes


def test_plan_caps_chunk_at_available_samples_and_records_metadata():
    plan = plan_sample_execution(
        emitters=3, consumers=2, samples=2, dtype="float32", budget_bytes=1_000_000
    )

    assert plan.chunk_size == 2
    assert plan.chunks == 1
    assert plan.metadata == {
        "emitters": 3,
        "consumers": 2,
        "samples": 2,
        "dtype": "float32",
        "budget_bytes": 1_000_000,
        "bytes_per_sample": plan.bytes_per_sample,
        "chunk_size": 2,
        "chunks": 1,
        "estimated_peak_bytes": plan.estimated_peak_bytes,
    }


def test_float64_requires_twice_float32_memory():
    args = dict(emitters=7, consumers=3, samples=1, budget_bytes=1_000_000)
    small = plan_sample_execution(dtype=np.float32, **args)
    large = plan_sample_execution(dtype=np.float64, **args)
    assert large.bytes_per_sample == 2 * small.bytes_per_sample


def test_plan_refuses_budget_that_cannot_fit_one_sample():
    with pytest.raises(ValueError, match="cannot fit one sample"):
        plan_sample_execution(
            emitters=100, consumers=10, samples=1, dtype=np.float64, budget_bytes=100
        )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"emitters": 0}, "emitters"),
        ({"consumers": -1}, "consumers"),
        ({"samples": 0}, "samples"),
        ({"budget_bytes": 0}, "budget_bytes"),
        ({"dtype": np.int64}, "floating-point"),
    ],
)
def test_plan_rejects_invalid_inputs(changes, message):
    args = dict(emitters=2, consumers=1, samples=1, dtype=np.float64, budget_bytes=10_000)
    args.update(changes)
    with pytest.raises(ValueError, match=message):
        plan_sample_execution(**args)
