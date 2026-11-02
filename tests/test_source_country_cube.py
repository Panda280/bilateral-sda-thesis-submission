import numpy as np

from sda import build_source_country_offshoring_cube


def _inputs():
    n = 4
    C0 = np.full((n, n), 0.2)
    C1 = C0 + np.array([
        [0.02, -0.01, 0.00, 0.01],
        [0.00, 0.01, -0.01, 0.02],
        [-0.01, 0.00, 0.02, 0.01],
        [0.01, -0.02, 0.00, -0.01],
    ])
    A_x0 = np.full((n, n), 0.1)
    A_x1 = A_x0 + np.arange(n * n).reshape(n, n) / 1000
    return dict(
        C0=C0, C1=C1, A_x0=A_x0, A_x1=A_x1,
        v0=np.array([0.8, 0.9, 1.0, 1.1]),
        delta_v=np.array([0.01, -0.01, 0.02, 0.00]),
        L0=np.eye(n) + 0.02, L1=np.eye(n) + 0.03,
        y0=np.array([2.0, 3.0, 4.0, 5.0]),
        delta_y=np.array([0.1, -0.1, 0.2, 0.0]),
        source_countries=("B", "B", "A", "A"),
        destination_countries=("B", "B", "A", "A"),
        emitter_labels=(("B", "s0"), ("B", "s1"), ("A", "s0"), ("A", "s1")),
        structural_driver_labels=(
            ("B", "s0"), ("B", "s1"), ("A", "s0"), ("A", "s1")
        ),
    )


def test_builds_reconciled_cube_with_deterministic_axes():
    cube = build_source_country_offshoring_cube(**_inputs())

    assert cube.baseline.shape == (4, 2, 2)
    assert cube.driver_country.labels == ("B", "A")
    assert cube.source_country.labels == ("B", "A")
    assert cube.reconciliation.passed
    np.testing.assert_allclose(
        cube.baseline.sum(axis=2),
        cube.baseline[:, :, 0] + cube.baseline[:, :, 1],
    )
    assert cube.metadata["effect"] == "D_C"


def test_explicit_source_order_only_reorders_source_axis():
    inputs = _inputs()
    default = build_source_country_offshoring_cube(**inputs)
    reordered = build_source_country_offshoring_cube(
        **inputs, country_order=("A", "B"), metadata={"run_id": "tiny"}
    )

    assert reordered.source_country.labels == ("A", "B")
    np.testing.assert_allclose(reordered.baseline, default.baseline[:, :, ::-1])
    assert reordered.driver_country.labels == default.driver_country.labels
    assert reordered.metadata["run_id"] == "tiny"
