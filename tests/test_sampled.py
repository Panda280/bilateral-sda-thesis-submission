import numpy as np
import pytest

from sda import (
    CANONICAL_EFFECTS, EffectDimensions, EffectResult, LabelSet, SDAResult,
    ValidationReport, plan_sample_execution, run_chunked_samples,
)


def _result(value, *, emitter_labels=("e0", "e1"), passed=True):
    emitter = LabelSet("emitter", emitter_labels)
    driver = LabelSet("driver", ("d0", "d1"))
    consumer = LabelSet("consumer", ("c0",))
    effects = {}
    for name in CANONICAL_EFFECTS:
        dims = EffectDimensions(emitter) if name == "D_v" else EffectDimensions(emitter, consumer if name == "D_y" else driver)
        shape = (2,) if dims.column is None else (2, len(dims.column.labels))
        effects[name] = EffectResult(name, np.full(shape, value, dtype=float), dims)
    report = ValidationReport(passed, {"mandatory": passed})
    if not passed:
        # Construction itself correctly blocks invalid results; emulate a result
        # whose report becomes invalid at the callback boundary.
        report = ValidationReport(True, {"mandatory": True})
    return SDAResult(effects, np.full(2, value, dtype=float), report)


def _plan(samples=5, chunk=2):
    base = plan_sample_execution(emitters=2, consumers=1, samples=samples, budget_bytes=10_000)
    return type(base)(**{**base.metadata, "chunk_size": chunk, "chunks": (samples + chunk - 1) // chunk,
                         "estimated_peak_bytes": chunk * base.bytes_per_sample})


def test_chunked_coordinator_accumulates_without_retaining_samples():
    seen = []
    def compute(start, stop):
        seen.append((start, stop))
        return (_result(float(index)) for index in range(start, stop))

    output = run_chunked_samples(_plan(), compute)
    assert seen == [(0, 2), (2, 4), (4, 5)]
    assert output.total_change.count == 5
    np.testing.assert_allclose(output.effects["D_C"].mean, 2.0)
    np.testing.assert_allclose(output.effects["D_C"].variance, 2.5)
    assert output.metadata["samples"] == 5


def test_chunked_coordinator_refuses_changed_labels():
    def compute(start, stop):
        for index in range(start, stop):
            yield _result(index, emitter_labels=("changed", "e1") if index == 2 else ("e0", "e1"))
    with pytest.raises(ValueError, match="failed validation"):
        run_chunked_samples(_plan(samples=3), compute)


@pytest.mark.parametrize("mode", ["too_few", "too_many", "failure"])
def test_chunked_coordinator_refuses_incomplete_or_failed_chunks(mode):
    def compute(start, stop):
        if mode == "failure":
            raise RuntimeError("sample kernel failed")
        count = stop - start + (1 if mode == "too_many" else -1)
        return (_result(index) for index in range(count))
    with pytest.raises(ValueError, match="chunk"):
        run_chunked_samples(_plan(samples=2), compute)
