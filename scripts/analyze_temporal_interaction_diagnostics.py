#!/usr/bin/env python
"""Build temporal diagnostics from validated compact interaction summaries.

The input is ``annual_group_metrics.csv`` produced by
``analyze_sourcing_group_trends.py``. Results remain separate by effect and
resolution. Robust MAD flags are anomaly screens, not formal structural-break
tests. All arithmetic uses the input's Mt CO2 fields; their upstream conversion
is kg / 1e9 = Mt CO2.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EFFECTS = ("D_C", "D_YF")
RESOLUTIONS = ("2013", "2016", "R3")
PARTITIONS = {
    "relationship": "relationship",
    "source_group": "source_group",
    "destination_group": "destination_group",
}
METRICS = ("net_mt", "positive_mt", "negative_mt", "gross_mt")
MAD_SCALE = 0.6744897501960817


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _robust_scores(values: pd.Series) -> tuple[pd.Series, float, float, bool]:
    median = float(values.median())
    mad = float((values - median).abs().median())
    if not np.isfinite(mad) or mad == 0.0:
        return pd.Series(0.0, index=values.index), median, mad, True
    return MAD_SCALE * (values - median) / mad, median, mad, False


def load_and_validate(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"resolution": str})
    required = {
        "effect", "resolution", "year_previous", "year_current",
        "dimension_level", "relationship", "source_group", "destination_group",
        *METRICS, "cancellation_ratio", "cell_count",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    numeric = [*METRICS, "cancellation_ratio", "cell_count"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("numeric fields contain non-finite values")
    if set(frame["effect"]) != set(EFFECTS):
        raise ValueError(f"effect inventory is {sorted(frame['effect'].unique())}")
    if set(frame["resolution"]) != set(RESOLUTIONS):
        raise ValueError(f"resolution inventory is {sorted(frame['resolution'].unique())}")
    if not (frame["year_current"] == frame["year_previous"] + 1).all():
        raise ValueError("input contains non-adjacent transitions")
    # Expected transition coverage is derived from the input itself (not
    # hardcoded to 2000-2014) so a temporally-restricted panel -- e.g. the
    # 2001-2012 first-commitment-period window -- validates the same way: every
    # effect/resolution series must still cover the SAME set of transitions.
    overall = frame.loc[frame["dimension_level"] == "overall"]
    expected_years = set(overall["year_previous"].unique())
    n_expected = len(expected_years)
    for effect in EFFECTS:
        for resolution in RESOLUTIONS:
            subset = overall.loc[
                (overall["effect"] == effect) & (overall["resolution"] == resolution)
            ]
            if set(subset["year_previous"]) != expected_years or len(subset) != n_expected:
                raise ValueError(f"incomplete overall series for {effect}/{resolution}")
    # Every reporting partition must reconcile to the overall signed and gross totals.
    parent = overall.set_index(
        ["effect", "resolution", "year_previous", "year_current"]
    )[["net_mt", "gross_mt"]]
    for dimension in PARTITIONS:
        grouped = (
            frame.loc[frame["dimension_level"] == dimension]
            .groupby(["effect", "resolution", "year_previous", "year_current"])[
                ["net_mt", "gross_mt"]
            ]
            .sum()
        )
        if not np.allclose(grouped.loc[parent.index], parent, rtol=1e-10, atol=1e-8):
            raise ValueError(f"{dimension} partition does not reconcile to overall totals")
    return frame


def build_annual_overall(frame: pd.DataFrame) -> pd.DataFrame:
    overall = frame.loc[frame["dimension_level"] == "overall", [
        "effect", "resolution", "year_previous", "year_current",
        "net_mt", "positive_mt", "negative_mt", "gross_mt", "cancellation_ratio",
    ]].copy()
    overall["absolute_net_mt"] = overall["net_mt"].abs()
    overall["net_materiality_share"] = np.where(
        overall["gross_mt"] > 0, overall["absolute_net_mt"] / overall["gross_mt"], np.nan
    )
    return overall.sort_values(["effect", "resolution", "year_previous"])


def build_reversals(
    annual: pd.DataFrame, materiality_thresholds: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = []
    summaries = []
    for (effect, resolution), group in annual.groupby(["effect", "resolution"], sort=False):
        group = group.sort_values("year_previous").reset_index(drop=True)
        for threshold in materiality_thresholds:
            material = group["net_materiality_share"] >= threshold
            signs = np.sign(group["net_mt"]).where(material, 0).astype(int)
            reversals = (signs != signs.shift(1)) & (signs != 0) & (signs.shift(1).fillna(0) != 0)
            nonzero = int((signs != 0).sum())
            dominant = int(signs.loc[signs != 0].value_counts().max()) if nonzero else 0
            summaries.append({
                "effect": effect,
                "resolution": resolution,
                "materiality_threshold": threshold,
                "material_transitions": nonzero,
                "adjacent_sign_reversals": int(reversals.sum()),
                "dominant_sign_share": dominant / nonzero if nonzero else np.nan,
            })
            for index in group.index[reversals]:
                previous = group.iloc[index - 1]
                current = group.iloc[index]
                events.append({
                    "effect": effect,
                    "resolution": resolution,
                    "materiality_threshold": threshold,
                    "previous_transition": f"{int(previous.year_previous)}_{int(previous.year_current)}",
                    "current_transition": f"{int(current.year_previous)}_{int(current.year_current)}",
                    "previous_net_mt": previous.net_mt,
                    "current_net_mt": current.net_mt,
                    "previous_materiality_share": previous.net_materiality_share,
                    "current_materiality_share": current.net_materiality_share,
                })
    return pd.DataFrame(summaries), pd.DataFrame(events)


def build_materiality_summary(
    annual: pd.DataFrame, materiality_thresholds: tuple[float, ...]
) -> pd.DataFrame:
    rows = []
    for (effect, resolution), group in annual.groupby(["effect", "resolution"], sort=False):
        for threshold in materiality_thresholds:
            rows.append({
                "effect": effect,
                "resolution": resolution,
                "materiality_threshold": threshold,
                "transitions_at_or_above": int((group["net_materiality_share"] >= threshold).sum()),
                "n_transitions": len(group),
                "mean_net_materiality_share": group["net_materiality_share"].mean(),
                "median_net_materiality_share": group["net_materiality_share"].median(),
                "mean_cancellation_ratio": group["cancellation_ratio"].mean(),
                "minimum_cancellation_ratio": group["cancellation_ratio"].min(),
                "maximum_cancellation_ratio": group["cancellation_ratio"].max(),
            })
    return pd.DataFrame(rows)


def build_composition(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    composition_rows = []
    shift_rows = []
    for dimension, category_column in PARTITIONS.items():
        subset = frame.loc[frame["dimension_level"] == dimension].copy()
        keys = ["effect", "resolution", "year_previous", "year_current"]
        subset["negative_magnitude_mt"] = -subset["negative_mt"]
        for value, output in [
            ("gross_mt", "gross_share"),
            ("positive_mt", "positive_share"),
            ("negative_magnitude_mt", "negative_magnitude_share"),
        ]:
            denominator = subset.groupby(keys)[value].transform("sum")
            subset[output] = np.where(denominator > 0, subset[value] / denominator, np.nan)
        for row in subset.itertuples(index=False):
            composition_rows.append({
                "effect": row.effect,
                "resolution": row.resolution,
                "year_previous": row.year_previous,
                "year_current": row.year_current,
                "dimension": dimension,
                "category": str(getattr(row, category_column)),
                "net_mt": row.net_mt,
                "gross_mt": row.gross_mt,
                "gross_share": row.gross_share,
                "positive_share": row.positive_share,
                "negative_magnitude_share": row.negative_magnitude_share,
                "cancellation_ratio": row.cancellation_ratio,
            })
        for (effect, resolution), group in subset.groupby(["effect", "resolution"], sort=False):
            pivoted = {
                share: group.pivot(index="year_previous", columns=category_column, values=share).sort_index()
                for share in ("gross_share", "positive_share", "negative_magnitude_share")
            }
            years = pivoted["gross_share"].index.to_list()
            for position in range(1, len(years)):
                record = {
                    "effect": effect,
                    "resolution": resolution,
                    "dimension": dimension,
                    "previous_start_year": years[position - 1],
                    "current_start_year": years[position],
                    "year_current": years[position] + 1,
                }
                for share, pivot in pivoted.items():
                    record[share.replace("share", "tvd")] = (
                        0.5 * (pivot.iloc[position] - pivot.iloc[position - 1]).abs().sum()
                    )
                shift_rows.append(record)
    composition = pd.DataFrame(composition_rows)
    shifts = pd.DataFrame(shift_rows)
    return composition, shifts


def build_anomalies(
    annual: pd.DataFrame,
    shifts: pd.DataFrame,
    mad_thresholds: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = []
    sensitivity = []
    series = []
    for (effect, resolution), group in annual.groupby(["effect", "resolution"], sort=False):
        group = group.sort_values("year_previous").copy()
        for field, label, scale in [
            ("net_mt", "annual_net_change_mt", 1.0),
            ("gross_mt", "annual_gross_change_mt", 1.0),
            ("cancellation_ratio", "annual_cancellation_change_pp", 100.0),
        ]:
            changes = group[field].diff() * scale
            series.append(pd.DataFrame({
                "effect": effect,
                "resolution": resolution,
                "dimension": "overall",
                "metric": label,
                "year_current": group["year_current"],
                "value": changes,
            }).dropna())
    for (effect, resolution, dimension), group in shifts.groupby(
        ["effect", "resolution", "dimension"], sort=False
    ):
        for field in ("gross_tvd", "positive_tvd", "negative_magnitude_tvd"):
            series.append(pd.DataFrame({
                "effect": effect,
                "resolution": resolution,
                "dimension": dimension,
                "metric": field,
                "year_current": group["year_current"],
                "value": group[field],
            }))
    all_series = pd.concat(series, ignore_index=True)
    for key, group in all_series.groupby(
        ["effect", "resolution", "dimension", "metric"], sort=False
    ):
        scores, median, mad, zero_mad = _robust_scores(group["value"])
        scored = group.copy()
        scored["robust_mad_score"] = scores
        scored["series_median"] = median
        scored["series_mad"] = mad
        scored["zero_mad"] = zero_mad
        selected = scored.loc[scored["robust_mad_score"].abs() >= 3.5]
        candidates.extend(selected.to_dict(orient="records"))
        for threshold in mad_thresholds:
            sensitivity.append({
                **dict(zip(["effect", "resolution", "dimension", "metric"], key, strict=True)),
                "mad_score_threshold": threshold,
                "candidate_count": int((scored["robust_mad_score"].abs() >= threshold).sum()),
                "n_observations": len(scored),
                "series_median": median,
                "series_mad": mad,
                "zero_mad": zero_mad,
            })
    return pd.DataFrame(candidates), pd.DataFrame(sensitivity)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--materiality-thresholds", nargs="+", type=float,
                        default=(0.0, 0.01, 0.05, 0.10))
    parser.add_argument("--mad-thresholds", nargs="+", type=float,
                        default=(2.5, 3.0, 3.5, 4.0))
    args = parser.parse_args(argv)
    materiality_thresholds = tuple(args.materiality_thresholds)
    mad_thresholds = tuple(args.mad_thresholds)
    if any(value < 0 or value >= 1 for value in materiality_thresholds):
        parser.error("materiality thresholds must be in [0, 1)")
    if any(value <= 0 for value in mad_thresholds):
        parser.error("MAD thresholds must be positive")

    frame = load_and_validate(args.annual_metrics.resolve())
    annual = build_annual_overall(frame)
    reversal_summary, reversal_events = build_reversals(annual, materiality_thresholds)
    materiality = build_materiality_summary(annual, materiality_thresholds)
    composition, shifts = build_composition(frame)
    anomalies, sensitivity = build_anomalies(annual, shifts, mad_thresholds)
    outputs = {
        "annual_overall_metrics.csv": annual,
        "sign_reversal_summary.csv": reversal_summary,
        "sign_reversal_events.csv": reversal_events,
        "materiality_summary.csv": materiality,
        "composition_metrics.csv": composition,
        "composition_shifts.csv": shifts,
        "anomaly_candidates.csv": anomalies,
        "anomaly_sensitivity.csv": sensitivity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, output in outputs.items():
        _write_csv(args.output_dir / filename, output)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": args.annual_metrics.as_posix(),
        "effects": list(EFFECTS),
        "resolutions": list(RESOLUTIONS),
        "transitions": [f"{y}_{y + 1}" for y in sorted(frame["year_previous"].unique())],
        "input_units": "Mt CO2 (source conversion: kg / 1e9)",
        "materiality_thresholds": list(materiality_thresholds),
        "mad_score_thresholds": list(mad_thresholds),
        "anomaly_language": "descriptive robust anomaly candidates; not formal structural breaks",
        "formulas": {
            "gross_mt": "sum(abs(Value)) / 1e9 at the persisted compact-cell grain",
            "cancellation_ratio": "1 - abs(net_mt) / gross_mt",
            "net_materiality_share": "abs(net_mt) / gross_mt",
            "composition_tvd": "0.5 * sum_categories(abs(share_t - share_t_minus_1))",
            "robust_mad_score": "0.67448975 * (value - median(value)) / MAD(value)",
            "zero_mad_policy": "score=0 and zero_mad=true; never infinity",
        },
        "outputs": {name: len(output) for name, output in outputs.items()},
    }
    _atomic_json(args.output_dir / "analysis_manifest.json", manifest)
    print(f"Wrote temporal diagnostics to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
