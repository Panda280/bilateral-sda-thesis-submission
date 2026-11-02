import numpy as np
import pytest

from sda.output import StreamingStatistics


def test_streaming_statistics_match_numpy_for_arbitrary_effect_shape():
    rng = np.random.default_rng(17)
    samples = rng.normal(loc=1e8, scale=0.25, size=(37, 2, 3))
    stats = StreamingStatistics.from_samples(samples)

    assert stats.count == samples.shape[0]
    np.testing.assert_allclose(stats.mean, samples.mean(axis=0), rtol=1e-15, atol=1e-8)
    np.testing.assert_allclose(stats.variance, samples.var(axis=0, ddof=1), rtol=3e-8)
    np.testing.assert_array_equal(stats.minimum, samples.min(axis=0))
    np.testing.assert_array_equal(stats.maximum, samples.max(axis=0))


def test_chunk_statistics_merge_and_restore():
    samples = np.arange(120, dtype=float).reshape(10, 3, 4)
    left = StreamingStatistics.from_samples(samples[:4])
    right = StreamingStatistics.from_samples(samples[4:])
    left.merge(right)
    expected = StreamingStatistics.from_samples(samples)

    np.testing.assert_allclose(left.mean, expected.mean)
    np.testing.assert_allclose(left.M2, expected.M2)
    restored = StreamingStatistics.restore(
        left.metadata, mean=left.mean, M2=left.M2,
        minimum=left.minimum, maximum=left.maximum,
    )
    assert restored.metadata == left.metadata
    np.testing.assert_array_equal(restored.maximum, left.maximum)


def test_statistics_reject_shape_and_non_finite_samples():
    stats = StreamingStatistics.empty((2,))
    with pytest.raises(ValueError, match="shape"):
        stats.update(np.zeros(3))
    with pytest.raises(ValueError, match="non-finite"):
        stats.update([1.0, np.nan])
