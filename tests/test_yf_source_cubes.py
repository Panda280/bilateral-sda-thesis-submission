import json

import numpy as np
import pyarrow.parquet as pq

from sda import build_yf_source_country_cube, build_yf_source_group_sector_cube
from sda.attribution.final_demand import attribute_final_demand_composition
from sda.output import export_yf_source_country_cube, export_yf_source_group_sector_cube


def _inputs():
    n, c, s = 4, 2, 2
    component = np.array([[1.0, -0.2], [0.3, 0.1], [-0.4, 0.8], [0.2, -0.1]])
    return component, dict(v0=np.array([0.8, 0.9, 1.0, 1.1]),
        delta_v=np.array([0.01, -0.01, 0.02, 0.0]), L0=np.eye(n) + 0.02,
        L1=np.eye(n) + 0.03, sectors_per_country=s,
        source_countries=("B", "B", "A", "A"), consumer_countries=("B", "A"),
        emitter_labels=(("B", "s0"), ("B", "s1"), ("A", "s0"), ("A", "s1")))


def test_yf_source_country_cube_reconciles_and_keeps_consumer_role():
    component, kwargs = _inputs(); cube = build_yf_source_country_cube(component, **kwargs)
    parent = attribute_final_demand_composition(component, v0=kwargs["v0"], delta_v=kwargs["delta_v"],
        L0=kwargs["L0"], L1=kwargs["L1"], sectors_per_country=kwargs["sectors_per_country"])
    assert cube.baseline.shape == (4, 2, 2)
    assert cube.consumer_country.role == "consumer_country"
    expected_parent = parent.reshape(4, 2, 2).sum(axis=2)
    np.testing.assert_allclose(cube.baseline.sum(axis=-1), expected_parent)
    assert cube.reconciliation.passed


def test_yf_source_country_slices_match_explicit_endpoint_reference():
    component, kwargs = _inputs()
    cube = build_yf_source_country_cube(component, **kwargs)
    delta_l = kwargs["L1"] - kwargs["L0"]
    kernel = (
        kwargs["v0"][:, None] * kwargs["L0"]
        + 0.5 * kwargs["v0"][:, None] * delta_l
        + 0.5 * kwargs["delta_v"][:, None] * kwargs["L0"]
        + (1.0 / 3.0) * kwargs["delta_v"][:, None] * delta_l
    )
    source_rows = np.asarray(kwargs["source_countries"], dtype=object)
    countries = tuple(cube.source_country.labels)
    emitters = len(kwargs["emitter_labels"])
    consumers = len(kwargs["consumer_countries"])
    sectors = kwargs["sectors_per_country"]

    for source_index, source_country in enumerate(countries):
        masked = component * (source_rows == source_country)[:, None]
        endpoint = np.einsum("ei,ic->eic", kernel, masked)
        expected = endpoint.reshape(emitters, len(countries), sectors, consumers)
        expected = expected.sum(axis=1).transpose(0, 2, 1)
        np.testing.assert_allclose(cube.baseline[..., source_index], expected.sum(axis=2))


def test_yf_source_group_sector_cube_reconciles_parent_and_groups():
    component, kwargs = _inputs()
    cube = build_yf_source_group_sector_cube(component, **kwargs,
        source_sectors=("s0", "s1", "s0", "s1"),
        source_group_by_country={"A": "Red", "B": "Non-Red"})
    parent = attribute_final_demand_composition(component, v0=kwargs["v0"], delta_v=kwargs["delta_v"],
        L0=kwargs["L0"], L1=kwargs["L1"], sectors_per_country=kwargs["sectors_per_country"])
    assert cube.baseline.shape == (4, 2, 4, 2)
    expected_parent = parent.reshape(4, 2, 2).sum(axis=2)
    np.testing.assert_allclose(cube.baseline.sum(axis=(-1, -2)), expected_parent)
    assert all(report.passed for report in cube.group_reconciliation.values())


def test_yf_group_sector_cube_uses_dc_comparable_collapsed_shape():
    component, kwargs = _inputs()
    cube = build_yf_source_group_sector_cube(component, **kwargs,
        source_sectors=("s0", "s1", "s0", "s1"),
        source_group_by_country={"A": "Red", "B": "Non-Red"})

    assert cube.baseline.shape == (4, 2, 4, 2)
    assert cube.baseline.nbytes == 4 * 2 * 4 * 2 * np.dtype(np.float64).itemsize


def test_yf_cube_exports_are_streamed_and_semantically_named(tmp_path):
    component, kwargs = _inputs()
    country = build_yf_source_country_cube(component, **kwargs)
    grouped = build_yf_source_group_sector_cube(component, **kwargs,
        source_sectors=("s0", "s1", "s0", "s1"),
        source_group_by_country={"A": "Red", "B": "Non-Red"})
    export_yf_source_country_cube(country, tmp_path / "country", run_id="r", units="kg", chunk_rows=3)
    export_yf_source_group_sector_cube(grouped, tmp_path / "grouped", run_id="r", units="kg", chunk_rows=5)
    country_table = pq.read_table(tmp_path / "country" / "D_YF.source_country_cube.parquet")
    grouped_table = pq.read_table(tmp_path / "grouped" / "D_YF.source_group_sector_cube.parquet")
    assert country_table.column_names == ["Emitter_Country", "Emitter_Sector", "Consumer_Country", "Source_Country", "Value"]
    assert grouped_table.column_names == ["Emitter_Country", "Emitter_Sector", "Consumer_Country", "Source_Group", "Source_Sector", "Value"]
    assert "Driver_Country" not in country_table.column_names + grouped_table.column_names
    grouped_rows = grouped_table.to_pylist()
    grouped_lookup = {
        (row["Emitter_Country"], row["Emitter_Sector"], row["Consumer_Country"],
         row["Source_Group"], row["Source_Sector"]): row["Value"]
        for row in grouped_rows
    }
    for index in np.ndindex(grouped.baseline.shape):
        emitter, consumer, source_group, source_sector = index
        key = (*grouped.emitter.labels[emitter], grouped.consumer_country.labels[consumer],
               grouped.source_group.labels[source_group],
               grouped.source_sector.labels[source_sector])
        np.testing.assert_allclose(grouped_lookup[key], grouped.baseline[index])
    manifest = json.loads((tmp_path / "country" / "yf_source_country_cube_report.json").read_text())
    assert manifest["effect"] == "D_YF"
    assert manifest["attribution_scope"] == "final-demand sourcing"
