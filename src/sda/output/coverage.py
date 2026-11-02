"""Immutable coverage manifests for production SDA run sets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Mapping

from sda.hierarchy import CANONICAL_EFFECTS

RUN_STATUSES = ("pending", "running", "valid", "failed")
_TRANSITIONS = {
    "pending": frozenset({"running"}),
    "running": frozenset({"valid", "failed"}),
    "failed": frozenset({"running"}),
    "valid": frozenset(),
}


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _hash(name: str, value: str) -> str:
    value = _nonempty(name, value).lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    return value


@dataclass(frozen=True, order=True)
class RunSpec:
    """Identity of one required database/year/flow/impact comparison."""

    database: str
    previous_year: int
    current_year: int
    flow: str
    impact: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "database", _nonempty("database", self.database))
        object.__setattr__(self, "flow", _nonempty("flow", self.flow))
        object.__setattr__(self, "impact", _nonempty("impact", self.impact))
        if isinstance(self.previous_year, bool) or not isinstance(self.previous_year, int):
            raise ValueError("previous_year must be an integer")
        if isinstance(self.current_year, bool) or not isinstance(self.current_year, int):
            raise ValueError("current_year must be an integer")
        if self.current_year <= self.previous_year:
            raise ValueError("current_year must be later than previous_year")

    @property
    def run_id(self) -> str:
        raw = f"{self.database}|{self.previous_year}|{self.current_year}|{self.flow}|{self.impact}"
        return sha256(raw.encode("utf-8")).hexdigest()[:16]

    def metadata(self) -> dict[str, str | int]:
        return {
            "run_id": self.run_id, "database": self.database,
            "previous_year": self.previous_year, "current_year": self.current_year,
            "flow": self.flow, "impact": self.impact,
        }


@dataclass(frozen=True)
class RunCoverage:
    spec: RunSpec
    expected_effects: tuple[str, ...] = CANONICAL_EFFECTS
    status: str = "pending"
    code_hash: str = ""
    input_hash: str = ""
    group_hash: str = ""
    execution_plan: Mapping[str, Any] = field(default_factory=dict)
    validated_effects: tuple[str, ...] = ()
    failure: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        expected = tuple(self.expected_effects)
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("expected_effects must be non-empty and unique")
        unknown = set(expected) - set(CANONICAL_EFFECTS)
        if unknown:
            raise ValueError(f"unknown canonical effects: {sorted(unknown)}")
        if self.status not in RUN_STATUSES:
            raise ValueError(f"invalid run status: {self.status}")
        validated = tuple(self.validated_effects)
        if len(set(validated)) != len(validated) or set(validated) - set(expected):
            raise ValueError("validated_effects must be unique members of expected_effects")
        for name in ("code_hash", "input_hash", "group_hash"):
            value = getattr(self, name)
            if value:
                object.__setattr__(self, name, _hash(name, value))
        if self.status in {"running", "valid", "failed"}:
            for name in ("code_hash", "input_hash", "group_hash"):
                if not getattr(self, name):
                    raise ValueError(f"{name} is required once a run starts")
        if self.status == "valid" and set(validated) != set(expected):
            raise ValueError("valid status requires every expected effect to be validated")
        if self.status == "failed" and not self.failure:
            raise ValueError("failed status requires a failure message")
        if self.status != "failed" and self.failure is not None:
            raise ValueError("failure is only allowed for failed runs")
        try:
            json.dumps(dict(self.execution_plan), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("execution_plan must contain finite JSON-compatible metadata") from exc
        object.__setattr__(self, "expected_effects", expected)
        object.__setattr__(self, "validated_effects", validated)
        object.__setattr__(self, "execution_plan", MappingProxyType(dict(self.execution_plan)))

    def transition(
        self, status: str, *, validated_effects: tuple[str, ...] | None = None,
        failure: str | None = None, code_hash: str | None = None,
        input_hash: str | None = None, group_hash: str | None = None,
        execution_plan: Mapping[str, Any] | None = None, updated_at: str | None = None,
    ) -> "RunCoverage":
        if status not in _TRANSITIONS.get(self.status, frozenset()):
            raise ValueError(f"invalid status transition: {self.status} -> {status}")
        timestamp = updated_at or datetime.now(timezone.utc).isoformat()
        return RunCoverage(
            spec=self.spec, expected_effects=self.expected_effects, status=status,
            code_hash=code_hash if code_hash is not None else self.code_hash,
            input_hash=input_hash if input_hash is not None else self.input_hash,
            group_hash=group_hash if group_hash is not None else self.group_hash,
            execution_plan=execution_plan if execution_plan is not None else self.execution_plan,
            validated_effects=(validated_effects if validated_effects is not None else self.validated_effects),
            failure=failure, updated_at=timestamp,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **self.spec.metadata(), "status": self.status,
            "expected_effects": list(self.expected_effects),
            "validated_effects": list(self.validated_effects), "code_hash": self.code_hash,
            "input_hash": self.input_hash, "group_hash": self.group_hash,
            "execution_plan": dict(self.execution_plan), "failure": self.failure,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class CoverageManifest:
    runs: tuple[RunCoverage, ...]

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError("coverage manifest requires at least one run")
        ids = [run.spec.run_id for run in self.runs]
        if len(ids) != len(set(ids)):
            raise ValueError("coverage manifest contains duplicate run specifications")

    @property
    def complete(self) -> bool:
        return all(run.status == "valid" for run in self.runs)

    def replace(self, updated: RunCoverage) -> "CoverageManifest":
        matches = [run.spec.run_id == updated.spec.run_id for run in self.runs]
        if not any(matches):
            raise KeyError(updated.spec.run_id)
        return CoverageManifest(tuple(updated if match else run for run, match in zip(self.runs, matches)))

    def summary(self) -> dict[str, Any]:
        counts = {status: sum(run.status == status for run in self.runs) for status in RUN_STATUSES}
        return {"schema_version": 1, "complete": self.complete, "counts": counts,
                "runs": [run.metadata() for run in sorted(self.runs, key=lambda r: r.spec.run_id)]}

    def to_json(self) -> str:
        return json.dumps(self.summary(), indent=2, sort_keys=True) + "\n"
