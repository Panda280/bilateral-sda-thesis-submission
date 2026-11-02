import numpy as np

from sda import build_source_group_sector_offshoring_cube
from sda.attribution import SOURCE_ORIGIN_GROUPS, build_group_sector_source_masks, build_source_origin_masks
from test_source_country_cube import _inputs


GROUPS = {"A": "Red", "B": "Non-Red"}
SECTORS = ("s0", "s1", "s0", "s1")


def test_masks_partition_and_reconstruct_existing_source_groups():
    inputs = _inputs()
    masks = build_group_sector_source_masks(
        inputs["source_countries"], SECTORS, inputs["destination_countries"], GROUPS
    )
    canonical = build_source_origin_masks(
        inputs["source_countries"], inputs["destination_countries"], GROUPS
    )
    assert tuple(masks) == tuple(
        (group, sector) for group in SOURCE_ORIGIN_GROUPS for sector in ("s0", "s1")
    )
    np.testing.assert_array_equal(sum(masks.values()), np.ones((4, 4), dtype=int))
    for group in SOURCE_ORIGIN_GROUPS:
        np.testing.assert_array_equal(
            sum(masks[(group, sector)] for sector in ("s0", "s1")), canonical[group]
        )


def test_group_sector_cube_passes_parent_and_group_reconciliations():
    inputs = _inputs()
    cube = build_source_group_sector_offshoring_cube(
        **inputs,
        source_sectors=SECTORS,
        source_group_by_country=GROUPS,
    )
    assert cube.baseline.shape == (4, 2, 4, 2)
    assert cube.source_group.labels == SOURCE_ORIGIN_GROUPS
    assert cube.source_sector.labels == ("s0", "s1")
    assert cube.reconciliation.passed
    assert set(cube.group_reconciliation) == set(SOURCE_ORIGIN_GROUPS)
    assert all(report.passed for report in cube.group_reconciliation.values())


def test_country_cube_accelerated_cpu_path_preserves_results():
    from sda import build_source_country_offshoring_cube

    inputs = _inputs()
    sequential = build_source_country_offshoring_cube(**inputs)
    batched = build_source_country_offshoring_cube(**inputs, batch_size=2)
    np.testing.assert_allclose(batched.baseline, sequential.baseline)
