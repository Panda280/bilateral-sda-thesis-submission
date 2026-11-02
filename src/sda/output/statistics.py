"""Numerically stable streaming statistics for sampled effect arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class StreamingStatistics:
    """Elementwise Welford statistics over arrays of one fixed shape."""

    count: int
    mean: NDArray[np.float64]
    M2: NDArray[np.float64]
    minimum: NDArray[np.float64]
    maximum: NDArray[np.float64]

    @classmethod
    def empty(cls, shape: tuple[int, ...]) -> "StreamingStatistics":
        if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
            raise ValueError("statistics shape dimensions must be positive integers")
        return cls(
            count=0,
            mean=np.zeros(shape, dtype=np.float64),
            M2=np.zeros(shape, dtype=np.float64),
            minimum=np.full(shape, np.inf, dtype=np.float64),
            maximum=np.full(shape, -np.inf, dtype=np.float64),
        )

    @classmethod
    def from_samples(cls, samples: ArrayLike) -> "StreamingStatistics":
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim < 1 or values.shape[0] == 0:
            raise ValueError("samples must have a non-empty leading sample axis")
        result = cls.empty(values.shape[1:])
        for sample in values:
            result.update(sample)
        return result

    def update(self, sample: ArrayLike) -> None:
        value = np.asarray(sample, dtype=np.float64)
        if value.shape != self.mean.shape:
            raise ValueError(f"sample shape {value.shape} does not match statistics shape {self.mean.shape}")
        if not np.isfinite(value).all():
            raise ValueError("sample contains non-finite values")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.M2 += delta * (value - self.mean)
        np.minimum(self.minimum, value, out=self.minimum)
        np.maximum(self.maximum, value, out=self.maximum)

    def merge(self, other: "StreamingStatistics") -> None:
        """Merge an independently accumulated chunk without loss of stability."""

        if not isinstance(other, StreamingStatistics):
            raise TypeError("other must be StreamingStatistics")
        if other.mean.shape != self.mean.shape:
            raise ValueError("cannot merge statistics with different shapes")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean[...] = other.mean
            self.M2[...] = other.M2
            self.minimum[...] = other.minimum
            self.maximum[...] = other.maximum
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.M2 += other.M2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total
        np.minimum(self.minimum, other.minimum, out=self.minimum)
        np.maximum(self.maximum, other.maximum, out=self.maximum)

    @property
    def variance(self) -> NDArray[np.float64]:
        if self.count < 2:
            return np.full(self.mean.shape, np.nan, dtype=np.float64)
        return self.M2 / (self.count - 1)

    @property
    def metadata(self) -> dict[str, Any]:
        """JSON-safe metadata; numeric arrays can be stored alongside it."""

        return {
            "algorithm": "welford",
            "count": self.count,
            "shape": list(self.mean.shape),
            "dtype": "float64",
            "variance_ddof": 1,
            "array_fields": ["mean", "M2", "minimum", "maximum"],
        }

    @classmethod
    def restore(cls, metadata: Mapping[str, Any], *, mean: ArrayLike, M2: ArrayLike,
                minimum: ArrayLike, maximum: ArrayLike) -> "StreamingStatistics":
        if metadata.get("algorithm") != "welford" or metadata.get("dtype") != "float64":
            raise ValueError("unsupported statistics metadata")
        shape = tuple(metadata.get("shape", ()))
        result = cls.empty(shape)
        arrays = [np.asarray(value, dtype=np.float64) for value in (mean, M2, minimum, maximum)]
        if any(value.shape != shape for value in arrays):
            raise ValueError("serialized statistics arrays do not match metadata shape")
        count = metadata.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("statistics count must be a non-negative integer")
        result.count = count
        result.mean[...] = arrays[0]
        result.M2[...] = arrays[1]
        result.minimum[...] = arrays[2]
        result.maximum[...] = arrays[3]
        return result
