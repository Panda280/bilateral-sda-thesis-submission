#!/usr/bin/env python3
"""Build publication exhibits from validated Kyoto-period group products.

The script is deliberately presentation-only: it does not select empirical
claims, invent unsupported bilateral cells, or create reporting windows beyond
P1, P2, and P3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PERIODS = ["P1", "P2", "P3"]
PERIOD_LABELS = {
    "P1": "P1: 2001–2004",
    "P2": "P2: 2005–2007",
    "P3": "P3: 2008–2012",
}
GROUPS = ["Red", "Non-Red", "Non-Participant"]
GROUP_LABELS = {
    "Red": "Reduction",
    "Non-Red": "Non-Reduction",
    "Non-Participant": "Non-Participation",
}
GROUP_COLORS = {
    "Red": "#4477AA",
    "Non-Red": "#EE6677",
    "Non-Participant": "#228833",
}
EFFECT_ROLE = {
    "D_v": "emitter",
    "D_C": "structural_driver",
    "D_T_coef": "structural_driver",
    "D_w_coef": "structural_driver",
    "D_YF": "consumer",
    "D_YB": "consumer",
    "D_y": "consumer",
}
EFFECTS = list(EFFECT_ROLE)
EFFECT_LABELS = {
    "D_v": r"$D_v$",
    "D_C": r"$D_C$",
    "D_T_coef": r"$D_{T,coef}$",
    "D_w_coef": r"$D_{w,coef}$",
    "D_YF": r"$D_{Y_F}$",
    "D_YB": r"$D_{Y_B}$",
    "D_y": r"$D_y$",
}
RELATIONSHIPS = ["Domestic", "Cross-border"]
CONFIG_LABELS = {
    "exio_2013": "EXIOBASE sources · 2013-compatible grid",
    "exio_2016": "EXIOBASE sources · 2016-compatible grid",
    "exio_R3": "EXIOBASE sources · R3 grid",
}
MT = 1e9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def validate_inputs(effects: pd.DataFrame, interactions: pd.DataFrame) -> dict:
    effect_required = {
        "period", "configuration", "resolution", "role", "group", "effect",
        "net", "positive", "negative", "gross", "cancellation",
        "transition_count", "coverage_status", "units", "price_basis",
    }
    interaction_required = {
        "period", "configuration", "resolution", "effect", "role",
        "source_group", "destination_group", "relationship", "net",
        "positive", "negative", "gross", "cancellation", "transition_count",
        "coverage_status", "units", "price_basis",
    }
    _require_columns(effects, effect_required, "group effects")
    _require_columns(interactions, interaction_required, "interactions")
    if set(effects.period) != set(PERIODS) or set(interactions.period) != set(PERIODS):
        raise ValueError("inputs must contain exactly P1, P2, and P3")
    if set(effects.group) != set(GROUPS):
        raise ValueError("group effects do not contain the fixed group registry")
    if set(interactions.source_group) != set(GROUPS) or set(interactions.destination_group) != set(GROUPS):
        raise ValueError("interactions do not contain the fixed 3-by-3 group registry")
    if set(interactions.effect) != {"D_C", "D_YF"}:
        raise ValueError("bilateral displays are restricted to registered D_C and D_YF products")
    if set(interactions.role) != {"structural_driver", "consumer"}:
        raise ValueError("unexpected bilateral attribution role")
    expected_counts = {"P1": 4, "P2": 3, "P3": 5}
    for period, count in expected_counts.items():
        if not (effects.loc[effects.period == period, "transition_count"] == count).all():
            raise ValueError(f"group-effect {period} coverage is not {count}")
        if not (interactions.loc[interactions.period == period, "transition_count"] == count).all():
            raise ValueError(f"interaction {period} coverage is not {count}")
    for frame, name in ((effects, "group effects"), (interactions, "interactions")):
        if set(frame.coverage_status) != {"complete"}:
            raise ValueError(f"{name} contains non-complete rows")
        if set(frame.units) != {"kg CO2"}:
            raise ValueError(f"{name} units are not kg CO2")
        if set(frame.price_basis) != {"CYP_(t-1) -> PYP_t"}:
            raise ValueError(f"{name} price basis mismatch")
        if not np.allclose(frame.net, frame.positive + frame.negative, rtol=0, atol=0.02):
            raise ValueError(f"{name}: net != positive + negative")
        if not np.allclose(frame.gross, frame.positive - frame.negative, rtol=0, atol=0.02):
            raise ValueError(f"{name}: gross != positive - negative")
    domestic_cross = interactions[
        (interactions.relationship == "Domestic")
        & (interactions.source_group != interactions.destination_group)
    ]
    if not domestic_cross.empty:
        raise ValueError("domestic cross-group observations are not allowed")
    configs = sorted(set(effects.configuration) & set(interactions.configuration))
    if set(effects.configuration) != set(interactions.configuration):
        raise ValueError("effect and interaction configuration inventories differ")
    for config in configs:
        for effect in ("D_C", "D_YF"):
            for period in PERIODS:
                q = interactions[(interactions.configuration == config) & (interactions.effect == effect) & (interactions.period == period)]
                totals = q.groupby(["source_group", "destination_group"], observed=True).size()
                if set(totals.index) != {(s, d) for s in GROUPS for d in GROUPS}:
                    raise ValueError(f"incomplete matrix: {config}/{effect}/{period}")
                expected_rows = pd.Series({(s, d): (2 if s == d else 1) for s in GROUPS for d in GROUPS})
                if not totals.sort_index().equals(expected_rows.sort_index()):
                    raise ValueError(f"relationship inventory mismatch: {config}/{effect}/{period}")
    return {"configurations": configs, "expected_transition_counts": expected_counts}


def select_overview(effects: pd.DataFrame, config: str) -> pd.DataFrame:
    parts = []
    for effect, role in EFFECT_ROLE.items():
        q = effects[
            (effects.configuration == config)
            & (effects.effect == effect)
            & (effects.role == role)
        ].copy()
        parts.append(q)
    out = pd.concat(parts, ignore_index=True)
    expected = len(PERIODS) * len(GROUPS) * len(EFFECTS)
    if len(out) != expected:
        raise ValueError(f"overview selection for {config}: expected {expected} rows, found {len(out)}")
    return out


def _symmetric_limit(values: pd.Series) -> float:
    high = float(np.nanmax(np.abs(values))) if len(values) else 1.0
    return max(1.0, high * 1.12)


def _positive_limit(values: pd.Series) -> float:
    high = float(np.nanmax(values)) if len(values) else 1.0
    return max(1.0, high * 1.12)


def plot_group_overview(frame: pd.DataFrame, config: str, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    net_limit = _symmetric_limit(frame.net / MT)
    gross_limit = _positive_limit(frame.gross / MT)
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), sharey=True, constrained_layout=True)
    offsets = {"Red": -0.22, "Non-Red": 0.0, "Non-Participant": 0.22}
    y = np.arange(len(EFFECTS))
    for col, period in enumerate(PERIODS):
        q = frame[frame.period == period]
        net_ax, gross_ax = axes[0, col], axes[1, col]
        for group in GROUPS:
            z = q[q.group == group].set_index("effect").reindex(EFFECTS)
            yy = y + offsets[group]
            net_ax.scatter(z.net / MT, yy, s=32, color=GROUP_COLORS[group], zorder=3)
            gross_ax.scatter(z.gross / MT, yy, s=38, facecolors="none", edgecolors=GROUP_COLORS[group], linewidths=1.4)
            # The line makes the magnitude of cancellation legible without
            # encoding it as a second, incomparable stacked contribution.
            gross_ax.hlines(yy, 0, z.gross / MT, color=GROUP_COLORS[group], alpha=0.35, lw=1)
        net_ax.axvline(0, color="#333333", lw=0.7)
        net_ax.set_xlim(-net_limit, net_limit)
        gross_ax.set_xlim(0, gross_limit)
        net_ax.set_title(PERIOD_LABELS[period])
        net_ax.set_xlabel("Net (Mt CO$_2$)")
        gross_ax.set_xlabel("Gross (Mt CO$_2$)")
        net_ax.grid(axis="x", color="#dddddd", lw=0.5)
        gross_ax.grid(axis="x", color="#dddddd", lw=0.5)
    axes[0, 0].set_yticks(y, [EFFECT_LABELS[e] for e in EFFECTS])
    axes[1, 0].set_yticks(y, [EFFECT_LABELS[e] for e in EFFECTS])
    axes[0, 0].invert_yaxis()
    fig.suptitle(f"Group-attributed effects by Kyoto period\n{CONFIG_LABELS.get(config, config)}")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUP_COLORS[g], markeredgecolor=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in GROUPS]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def aggregate_relationship_view(interactions: pd.DataFrame, config: str, effect: str) -> pd.DataFrame:
    q = interactions[(interactions.configuration == config) & (interactions.effect == effect)].copy()
    out = q.groupby(["period", "relationship", "destination_group"], as_index=False, observed=True).agg(
        net=("net", "sum"), positive=("positive", "sum"), negative=("negative", "sum"), gross=("gross", "sum")
    )
    out["cancellation"] = np.where(out.gross > 0, 1 - np.abs(out.net) / out.gross, np.nan)
    return out


def plot_relationship_view(frame: pd.DataFrame, config: str, effect: str, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    net_limit = _symmetric_limit(frame.net / MT)
    gross_limit = _positive_limit(frame.gross / MT)
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 6.8), sharex=False, sharey=True, constrained_layout=True)
    y = np.arange(len(GROUPS))
    for row, relationship in enumerate(RELATIONSHIPS):
        for col, period in enumerate(PERIODS):
            ax = axes[row, col]
            q = frame[(frame.period == period) & (frame.relationship == relationship)].set_index("destination_group").reindex(GROUPS)
            for yi, group in enumerate(GROUPS):
                ax.plot([0, q.loc[group, "gross"] / MT], [yi, yi], color="#bbbbbb", lw=1.2, zorder=1)
                ax.scatter(q.loc[group, "gross"] / MT, yi, s=48, facecolors="white", edgecolors=GROUP_COLORS[group], linewidths=1.5, zorder=2)
                ax.scatter(q.loc[group, "net"] / MT, yi, s=38, color=GROUP_COLORS[group], zorder=3)
            ax.axvline(0, color="#333333", lw=0.7)
            ax.set_xlim(-net_limit, max(net_limit, gross_limit))
            ax.grid(axis="x", color="#dddddd", lw=0.5)
            if row == 0:
                ax.set_title(PERIOD_LABELS[period])
            if col == 0:
                ax.set_ylabel(f"{relationship}\nDestination group")
            ax.set_xlabel("Mt CO$_2$")
    axes[0, 0].set_yticks(y, [GROUP_LABELS[g] for g in GROUPS])
    axes[0, 0].invert_yaxis()
    fig.suptitle(f"Domestic and cross-border {EFFECT_LABELS[effect]} by destination group\n{CONFIG_LABELS.get(config, config)}")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="#555555", label="Net"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", label="Gross"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def aggregate_matrices(interactions: pd.DataFrame, config: str, effect: str) -> pd.DataFrame:
    q = interactions[(interactions.configuration == config) & (interactions.effect == effect)].copy()
    out = q.groupby(["period", "source_group", "destination_group"], as_index=False, observed=True).agg(
        net=("net", "sum"), positive=("positive", "sum"), negative=("negative", "sum"), gross=("gross", "sum")
    )
    out["cancellation"] = np.where(out.gross > 0, 1 - np.abs(out.net) / out.gross, np.nan)
    return out


def plot_matrices(frame: pd.DataFrame, config: str, effect: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    net_limit = _symmetric_limit(frame.net / MT)
    gross_limit = _positive_limit(frame.gross / MT)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4), constrained_layout=True)
    net_image = gross_image = None
    for col, period in enumerate(PERIODS):
        q = frame[frame.period == period].set_index(["source_group", "destination_group"])
        net = np.array([[q.loc[(s, d), "net"] / MT for d in GROUPS] for s in GROUPS])
        gross = np.array([[q.loc[(s, d), "gross"] / MT for d in GROUPS] for s in GROUPS])
        cancellation = np.array([[q.loc[(s, d), "cancellation"] * 100 for d in GROUPS] for s in GROUPS])
        net_ax, gross_ax = axes[0, col], axes[1, col]
        net_image = net_ax.imshow(net, cmap="RdBu_r", vmin=-net_limit, vmax=net_limit)
        gross_image = gross_ax.imshow(gross, cmap="YlOrBr", vmin=0, vmax=gross_limit)
        for i in range(3):
            for j in range(3):
                net_ax.text(j, i, f"{net[i, j]:+.0f}", ha="center", va="center", fontsize=8)
                gross_ax.text(j, i, f"{gross[i, j]:.0f}\n({cancellation[i, j]:.0f}% c.)", ha="center", va="center", fontsize=7.5)
        net_ax.set_title(PERIOD_LABELS[period])
        for ax in (net_ax, gross_ax):
            ax.set_xticks(range(3), [GROUP_LABELS[g] for g in GROUPS], rotation=28, ha="right")
            ax.set_yticks(range(3), [GROUP_LABELS[g] for g in GROUPS] if col == 0 else [])
        if col == 0:
            net_ax.set_ylabel("Net\nSource group")
            gross_ax.set_ylabel("Gross (cancellation %)\nSource group")
        gross_ax.set_xlabel("Destination group")
    fig.colorbar(net_image, ax=axes[0, :], location="right", shrink=0.72, label="Net (Mt CO$_2$)")
    fig.colorbar(gross_image, ax=axes[1, :], location="right", shrink=0.72, label="Gross (Mt CO$_2$)")
    fig.suptitle(f"Source-group to destination-group {EFFECT_LABELS[effect]} interactions\n{CONFIG_LABELS.get(config, config)}")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def caption_data(overviews: dict[str, pd.DataFrame], interactions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for config, frame in overviews.items():
        for period in PERIODS:
            for effect in EFFECTS:
                q = frame[(frame.period == period) & (frame.effect == effect)]
                dominant = q.loc[q.gross.idxmax()]
                rows.append({
                    "figure_family": "group_effect_overview", "configuration": config, "period": period,
                    "effect": effect, "role": EFFECT_ROLE[effect], "relationship": "not_applicable",
                    "source_group": "group_marginal", "destination_group": dominant.group,
                    "net_mt_co2": dominant.net / MT, "gross_mt_co2": dominant.gross / MT,
                    "cancellation_percent": dominant.cancellation * 100,
                    "selection_note": "largest group gross for the effect-period",
                })
        for effect in ("D_C", "D_YF"):
            rel = aggregate_relationship_view(interactions, config, effect)
            mat = aggregate_matrices(interactions, config, effect)
            for period in PERIODS:
                for relationship in RELATIONSHIPS:
                    q = rel[(rel.period == period) & (rel.relationship == relationship)]
                    dominant = q.loc[q.gross.idxmax()]
                    rows.append({
                        "figure_family": "domestic_foreign", "configuration": config, "period": period,
                        "effect": effect, "role": "structural_driver" if effect == "D_C" else "consumer",
                        "relationship": relationship, "source_group": "summed",
                        "destination_group": dominant.destination_group,
                        "net_mt_co2": dominant.net / MT, "gross_mt_co2": dominant.gross / MT,
                        "cancellation_percent": dominant.cancellation * 100,
                        "selection_note": "largest destination-group gross within relationship",
                    })
                q = mat[mat.period == period]
                dominant = q.loc[q.gross.idxmax()]
                rows.append({
                    "figure_family": "interaction_matrix", "configuration": config, "period": period,
                    "effect": effect, "role": "structural_driver" if effect == "D_C" else "consumer",
                    "relationship": "domestic_plus_cross_border", "source_group": dominant.source_group,
                    "destination_group": dominant.destination_group, "net_mt_co2": dominant.net / MT,
                    "gross_mt_co2": dominant.gross / MT, "cancellation_percent": dominant.cancellation * 100,
                    "selection_note": "largest group-pair gross for the effect-period",
                })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publication-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.publication_dir.mkdir(parents=True, exist_ok=True)

    effects_path = args.products_dir / "kyoto_period_group_effects.csv"
    interactions_path = args.products_dir / "kyoto_period_group_interactions.csv"
    source_manifest_path = args.products_dir / "analysis_manifest.json"
    effects = pd.read_csv(effects_path, dtype={"resolution": str})
    interactions = pd.read_csv(interactions_path, dtype={"resolution": str})
    checks = validate_inputs(effects, interactions)

    outputs: list[dict] = []
    publication_outputs: list[dict] = []
    overviews: dict[str, pd.DataFrame] = {}
    for config in checks["configurations"]:
        overview = select_overview(effects, config)
        overviews[config] = overview
        paths = [args.output_dir / f"group_effect_overview_{config}.pdf"]
        plot_group_overview(overview, config, paths[0])
        for effect in ("D_C", "D_YF"):
            relationship = aggregate_relationship_view(interactions, config, effect)
            matrix = aggregate_matrices(interactions, config, effect)
            relationship_path = args.output_dir / f"domestic_foreign_{effect}_{config}.pdf"
            matrix_path = args.output_dir / f"group_interaction_matrix_{effect}_{config}.pdf"
            plot_relationship_view(relationship, config, effect, relationship_path)
            plot_matrices(matrix, config, effect, matrix_path)
            paths.extend([relationship_path, matrix_path])
        for path in paths:
            outputs.append({"path": path.name, "sha256": sha256(path)})
            destination = args.publication_dir / path.name
            shutil.copy2(path, destination)
            publication_outputs.append({"path": str(destination), "sha256": sha256(destination)})

    caption_path = args.output_dir / "caption_data.csv"
    captions = caption_data(overviews, interactions)
    captions.to_csv(caption_path, index=False)
    outputs.append({"path": caption_path.name, "rows": len(captions), "sha256": sha256(caption_path)})

    report_path = args.output_dir / "VALIDATION_REPORT.md"
    report_path.write_text(
        "# Kyoto-period group exhibit validation\n\n"
        "Status: generated; all display gates passed.\n\n"
        "- Displays use only P1 (4 transitions), P2 (3), and P3 (5).\n"
        "- The fixed group order is Reduction, Non-Reduction, Non-Participation.\n"
        "- Group-effect overviews use one declared role per hierarchy-safe leaf.\n"
        "- Bilateral figures use only registered `D_C`/structural-driver and `D_YF`/consumer products.\n"
        "- Domestic and cross-border are separate facets; same-group cross-border is not called domestic.\n"
        "- Every interaction matrix contains all nine source-group/destination-group cells.\n"
        "- Net and gross are displayed together; matrix gross panels also state cancellation.\n"
        "- All three periods use identical layouts and within-figure scales.\n"
        "- Figures are descriptive accounting displays and make no causal policy claim.\n",
        encoding="utf-8",
    )
    outputs.append({"path": report_path.name, "sha256": sha256(report_path)})

    manifest = {
        "schema_version": 1,
        "analysis_id": "kyoto-period-group-exhibits-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"script": "implementation/bilateral_sda_attribution/scripts/build_kyoto_period_group_exhibits.py", "command": sys.argv},
        "source_product_manifest_sha256": sha256(source_manifest_path),
        "inputs": [
            {"path": str(effects_path), "sha256": sha256(effects_path)},
            {"path": str(interactions_path), "sha256": sha256(interactions_path)},
        ],
        "scope": {
            "periods": PERIODS, "groups": GROUPS, "configurations": checks["configurations"],
            "bilateral_effect_roles": {"D_C": "structural_driver", "D_YF": "consumer"},
            "units_display": "Mt CO2", "source_units": "kg CO2", "price_basis": "CYP_(t-1) -> PYP_t",
        },
        "validation": {
            "passed": True, "expected_transition_counts": checks["expected_transition_counts"],
            "identities": "passed", "complete_3x3_matrices": True,
            "domestic_cross_group_observations": 0, "unsupported_bilateral_effects_inferred": False,
        },
        "outputs": outputs,
        "publication_outputs": publication_outputs,
        "limitations": [
            "Descriptive accounting attributions, not causal policy estimates.",
            "Analytical groups are static classifications, not legal compliance categories.",
            "Domestic means same country; same-group cross-border remains cross-border.",
        ],
    }
    manifest_path = args.output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
