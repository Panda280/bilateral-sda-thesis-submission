#!/usr/bin/env python3
"""Verify every checksum and contract in one published canonical run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sda.output import load_result  # noqa: E402


def verify_run(run_dir: str | Path) -> dict[str, object]:
    path = Path(run_dir)
    result = load_result(path)
    result.validation.raise_if_invalid()
    return {
        "run_dir": str(path),
        "passed": True,
        "effects": sorted(result.effects),
        "effect_shapes": {name: list(effect.baseline.shape) for name, effect in result.effects.items()},
        "total_change_shape": list(result.total_change.shape),
        "checks": dict(result.validation.checks),
        "max_abs_error": result.validation.max_abs_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    summary = verify_run(args.run_dir)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
