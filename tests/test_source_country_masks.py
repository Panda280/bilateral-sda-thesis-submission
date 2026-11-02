import numpy as np
import pytest

from sda.attribution import build_per_country_source_masks


def test_country_masks_cover_multiple_sectors_and_fold_domestic_cells():
    sources = ("B", "B", "A", "A", "C", "C")
    destinations = ("A", "B", "C")

    masks = build_per_country_source_masks(sources, destinations)

    assert tuple(masks) == ("B", "A", "C")
    np.testing.assert_array_equal(sum(masks.values()), np.ones((6, 3), dtype=int))
    assert masks["B"][0, 1]  # domestic destination remains in B's mask
    assert masks["B"][1, 0]  # foreign destination is in the same source mask
    assert masks["A"][2, 0]
    assert masks["A"][2, 1]
    assert not masks["A"][1, 1]
    assert masks["C"][5, 2]


def test_explicit_country_order_is_deterministic_and_must_be_exact():
    sources = ("B", "B", "A", "C")
    destinations = ("A", "B", "C")

    masks = build_per_country_source_masks(
        sources, destinations, country_order=("C", "A", "B")
    )
    assert tuple(masks) == ("C", "A", "B")

    with pytest.raises(ValueError, match="unknown=.*D"):
        build_per_country_source_masks(
            sources, destinations, country_order=("A", "B", "C", "D")
        )
    with pytest.raises(ValueError, match="missing=.*C"):
        build_per_country_source_masks(
            sources, destinations, country_order=("A", "B")
        )
    with pytest.raises(ValueError, match="unique"):
        build_per_country_source_masks(
            sources, destinations, country_order=("A", "B", "C", "C")
        )


@pytest.mark.parametrize(
    ("sources", "destinations", "match"),
    [
        ((), ("A",), "source.*non-empty"),
        (("A",), (), "destination.*non-empty"),
        (("",), ("A",), "source.*non-empty strings"),
        ((" A ",), ("A",), "source.*non-empty strings"),
        (("A",), (" ",), "destination.*non-empty strings"),
        (("A",), ("UNKNOWN",), "unknown destination"),
    ],
)
def test_empty_or_unknown_labels_are_rejected(sources, destinations, match):
    with pytest.raises(ValueError, match=match):
        build_per_country_source_masks(sources, destinations)
