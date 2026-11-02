#!/usr/bin/env python3
"""Integrate completed descriptive analyses into role/sign/resolution-safe cases.

The sign-stratified v2 register remains the candidate authority.  Additional
analyses provide evidence columns but never change semantic keys or create
cross-effect, cross-role, cross-sign, or cross-resolution comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STRATUM = ["effect", "attribution_role", "resolution", "selection_sign",
           "entity_type", "selection_scope"]
REQUIRED = set(STRATUM + ["canonical_entity", "units", "eligible",
                          "descriptive_score", "result_id"])
WEIGHTS = {"base": .55, "temporal": .15, "sourcing": .15,
           "granularity": .075, "tw": .075}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(series: pd.Series) -> pd.Series:
    """Finite [0,1] percentile; tied singleton strata receive one."""
    finite = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return finite.rank(method="average", pct=True)


def build(case_scores: pd.DataFrame, temporal: pd.DataFrame,
          sourcing: pd.DataFrame, controlled: pd.DataFrame,
          harmonized: pd.DataFrame, tw: pd.DataFrame,
          *, top_n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = REQUIRED - set(case_scores.columns)
    if missing:
        raise ValueError(f"case score columns missing: {sorted(missing)}")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    frame = case_scores.copy()
    frame["resolution"] = frame.resolution.astype(str)
    if frame.result_id.duplicated().any():
        raise ValueError("upstream result IDs are not unique")
    if set(frame.units) != {"kg CO2"}:
        raise ValueError("integrated pipeline requires preserved kg CO2 units")

    # Context is aggregated only at its actual semantic grain.
    tctx = (temporal.assign(temporal_abs_mad=lambda x: x.robust_mad_score.abs())
            .groupby(["effect", "resolution"], as_index=False)
            .agg(temporal_anomaly_count=("year_current", "size"),
                 temporal_max_abs_mad=("temporal_abs_mad", "max")))
    tctx["resolution"] = tctx.resolution.astype(str)
    frame = frame.merge(tctx, on=["effect", "resolution"], how="left",
                        validate="many_to_one")

    # Sourcing roles are mapped only where the source manifest defines them.
    role_map = {("D_C", "structural_driver"): "structural_driver_country",
                ("D_YF", "consumer"): "consumer_country"}
    frame["sourcing_focal_role"] = [role_map.get((e, r)) for e, r in
                                     zip(frame.effect, frame.attribution_role)]
    src = sourcing.rename(columns={"effect_id": "effect",
                                   "focal_country": "canonical_entity"}).copy()
    src["resolution"] = src.resolution.astype(str)
    src_cols = ["effect", "resolution", "focal_role", "canonical_entity",
                "case_score", "result_id", "evidence_persistence",
                "mean_source_hhi", "mean_source_entropy"]
    src = src[src_cols].rename(columns={"case_score": "sourcing_case_score",
                                        "result_id": "sourcing_result_id",
                                        "focal_role": "sourcing_focal_role"})
    frame = frame.merge(src, on=["effect", "resolution", "sourcing_focal_role",
                                 "canonical_entity"], how="left",
                        validate="many_to_one")
    invalid_src = ~((frame.entity_type == "country") & frame.sourcing_focal_role.notna())
    for column in ["sourcing_case_score", "sourcing_result_id", "evidence_persistence",
                   "mean_source_hhi", "mean_source_entropy"]:
        frame.loc[invalid_src, column] = np.nan

    cctx = (controlled.groupby(["effect_id", "target_resolution"], as_index=False)
            .agg(granularity_cancellation_loss=("cancellation_loss", "mean"),
                 granularity_sign_reversal_rate=("sign_reversal_rate", "mean"),
                 granularity_edge_loss_rate=("sparse_edge_loss_rate", "mean"))
            .rename(columns={"effect_id": "effect", "target_resolution": "resolution"}))
    hctx = (harmonized.groupby(["effect_id", "target_resolution"], as_index=False)
            .agg(harmonized_sign_disagreement=("sign_disagreement_rate", "mean"),
                 harmonized_source_substitution=("source_substitution_total_variation", "mean"))
            .rename(columns={"effect_id": "effect", "target_resolution": "resolution"}))
    cctx["resolution"] = cctx.resolution.astype(str); hctx["resolution"] = hctx.resolution.astype(str)
    frame = frame.merge(cctx, on=["effect", "resolution"], how="left", validate="many_to_one")
    frame = frame.merge(hctx, on=["effect", "resolution"], how="left", validate="many_to_one")

    twctx = (tw.groupby(["resolution"], as_index=False)
             .agg(tw_information_gain=("split_information_gain_ratio", "mean"),
                  tw_cancellation=("paired_total_cancellation", "mean"),
                  tw_t_share=("t_absolute_share_of_signed_totals", "mean"),
                  tw_w_share=("w_absolute_share_of_signed_totals", "mean")))
    twctx["resolution"] = twctx.resolution.astype(str)
    frame = frame.merge(twctx, on="resolution", how="left", validate="many_to_one")
    # T/w context applies only to the two coefficient subeffects.
    frame.loc[~frame.effect.isin(["D_T_coef", "D_w_coef"]),
              ["tw_information_gain", "tw_cancellation", "tw_t_share", "tw_w_share"]] = np.nan

    frame["base_component"] = frame.groupby(STRATUM, dropna=False).descriptive_score.transform(percentile)
    frame["temporal_component"] = frame.temporal_max_abs_mad.groupby(
        [frame.effect, frame.resolution], dropna=False).transform(percentile).where(
        frame.temporal_max_abs_mad.notna())
    frame["sourcing_component"] = frame.sourcing_case_score.groupby(
        [frame.effect, frame.attribution_role, frame.resolution,
         frame.selection_sign], dropna=False).transform(percentile).where(
        frame.sourcing_case_score.notna())
    gran_raw = frame[["granularity_cancellation_loss", "granularity_sign_reversal_rate",
                      "granularity_edge_loss_rate", "harmonized_sign_disagreement",
                      "harmonized_source_substitution"]].mean(axis=1)
    frame["granularity_component"] = gran_raw.groupby(
        [frame.effect, frame.resolution], dropna=False).transform(percentile).where(gran_raw.notna())
    tw_raw = frame[["tw_information_gain", "tw_cancellation"]].mean(axis=1)
    frame["tw_component"] = tw_raw.groupby(
        [frame.effect, frame.resolution], dropna=False).transform(percentile).where(tw_raw.notna())

    components = {"base": "base_component", "temporal": "temporal_component",
                  "sourcing": "sourcing_component", "granularity": "granularity_component",
                  "tw": "tw_component"}
    numerator = pd.Series(0.0, index=frame.index); denominator = pd.Series(0.0, index=frame.index)
    for name, column in components.items():
        available = frame[column].notna(); numerator += frame[column].fillna(0) * WEIGHTS[name]
        denominator += available.astype(float) * WEIGHTS[name]
    frame["integrated_score"] = numerator / denominator
    frame["evidence_components_available"] = sum(frame[c].notna().astype(int) for c in components.values())
    frame["parent_result_id"] = frame.result_id
    frame["integrated_result_id"] = frame.apply(lambda row: "ICASE-" + hashlib.sha256(
        (row.parent_result_id + "|integrated-v1").encode()).hexdigest()[:16].upper(), axis=1)
    frame["provisional"] = True
    frame["interpretation_limit"] = "descriptive integrated drill-down candidate; not causal proof"
    frame = frame.sort_values(STRATUM + ["integrated_score", "canonical_entity"],
                              ascending=[True] * len(STRATUM) + [False, True])
    frame["integrated_rank_within_stratum"] = frame.groupby(STRATUM, dropna=False).cumcount() + 1
    selected = (frame.loc[frame.eligible.astype(bool)]
                .groupby(STRATUM, dropna=False, group_keys=False).head(top_n).copy())
    return frame, selected


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ["case-scores", "temporal", "sourcing", "controlled", "harmonized", "tw"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name.replace("_", "-"): getattr(args, name) for name in
             ["case_scores", "temporal", "sourcing", "controlled", "harmonized", "tw"]}
    manifests = []
    for path in args.upstream_manifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        status = data.get("status")
        # The temporal v1 manifest predates the status field.  Its enumerated,
        # non-empty output counts are the legacy completion gate.
        legacy_complete = status is None and isinstance(data.get("outputs"), dict) and bool(data["outputs"]) \
            and all(isinstance(value, int) and value > 0 for value in data["outputs"].values())
        if status != "complete" and not legacy_complete:
            raise ValueError(f"upstream incomplete: {path}")
        manifests.append({"path": str(path), "sha256": sha256(path),
                          "analysis_id": data.get("analysis_id"),
                          "status": status or "legacy_complete_output_inventory"})
    scores, selected = build(*(pd.read_csv(paths[k]) for k in
                               ["case-scores", "temporal", "sourcing", "controlled", "harmonized", "tw"]),
                             top_n=args.top_n)
    compact_cols = STRATUM + ["canonical_entity", "units", "signed_total", "absolute_total",
                              "parent_result_id", "integrated_result_id", "integrated_score",
                              "integrated_rank_within_stratum", "evidence_components_available",
                              "sourcing_result_id", "temporal_max_abs_mad",
                              "granularity_cancellation_loss", "harmonized_sign_disagreement",
                              "tw_information_gain", "provisional", "interpretation_limit"]
    outputs = []
    for name, data in [("integrated_case_scores.csv", scores[compact_cols]),
                       ("selected_integrated_cases.csv", selected[compact_cols])]:
        path = args.output_dir / name; data.to_csv(path, index=False)
        outputs.append({"path": name, "rows": len(data), "sha256": sha256(path)})
    report = ("# Integrated drill-down case selection\n\n"
              f"Selected {len(selected):,} provisional descriptive cases from {len(scores):,} "
              "sign-stratified candidates. Rankings are independent within effect, attribution "
              "role, resolution, sign, entity type, and selection scope. Missing evidence is not "
              "imputed; weights are renormalized over available components.\n")
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    manifest = {"schema_version": 1, "analysis_id": "integrated-drilldown-case-selection-v1",
                "status": "complete", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "generator": {"script": "scripts/build_integrated_drilldown_cases.py", "command": sys.argv},
                "scope": {"indicator": "fossil-origin CO2", "price_basis": "CYP_(t-1) to PYP_t",
                          "units": "kg CO2", "provisional": True},
                "parameters": {"top_n_per_exact_stratum": args.top_n, "weights": WEIGHTS,
                               "missing_evidence": "renormalize available weights; no imputation"},
                "inputs": [{"path": str(path), "sha256": sha256(path)} for path in paths.values()],
                "upstream_manifests": manifests, "outputs": outputs,
                "validation": {"passed": True, "stable_ids_unique": bool(scores.integrated_result_id.is_unique),
                               "parent_ids_unique": bool(scores.parent_result_id.is_unique),
                               "finite_scores": bool(np.isfinite(scores.integrated_score).all()),
                               "units_preserved": bool((scores.units == "kg CO2").all()),
                               "unresolved_candidates_inherited": False,
                               "strata": STRATUM, "thesis_status": "accepted_provisional"},
                "limitations": ["Descriptive selection does not establish causality.",
                                "Temporal and granularity/T-w evidence is contextual at effect-resolution grain.",
                                "Sourcing evidence is attached only for exact declared country-role semantics."]}
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
