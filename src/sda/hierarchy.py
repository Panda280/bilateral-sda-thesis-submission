"""Single source of truth for canonical effect names and hierarchy."""

from types import MappingProxyType

TOP_LEVEL_EFFECTS = ("D_v", "D_C", "D_A_x", "D_YF", "D_YB", "D_y")
NESTED_EFFECTS = ("D_T_coef", "D_w_coef")
CANONICAL_EFFECTS = TOP_LEVEL_EFFECTS + NESTED_EFFECTS

EFFECT_HIERARCHY = MappingProxyType(
    {
        "Global emissions change": MappingProxyType(
            {
                "Emission intensity": MappingProxyType({"D_v": None}),
                "Leontief/production structure": MappingProxyType(
                    {
                        "D_C": None,
                        "D_A_x": MappingProxyType(
                            {"D_T_coef": None, "D_w_coef": None}
                        ),
                    }
                ),
                "Final demand": MappingProxyType(
                    {"D_YF": None, "D_YB": None, "D_y": None}
                ),
            }
        )
    }
)

LEGACY_EFFECT_ALIASES = MappingProxyType(
    {"Dint": "D_v", "D_C2": "D_C", "D_Ax2": "D_A_x", "DYy": "D_y"}
)
