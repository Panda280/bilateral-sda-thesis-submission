"""Canonical public API for bilateral SDA attribution."""

from .hierarchy import CANONICAL_EFFECTS, EFFECT_HIERARCHY
from .model import EffectDimensions, EffectResult, LabelSet, SDAResult, ValidationReport
from .offshoring import (
    SourceCountryCube,
    SourceGroupSectorCube,
    build_source_country_offshoring_cube,
    build_source_group_sector_offshoring_cube,
)
from .decomposition import run_canonical_sda, run_canonical_sda_state
from .final_demand_sourcing import (
    YFSourceCountryCube, YFSourceGroupSectorCube,
    build_yf_source_country_cube, build_yf_source_group_sector_cube,
)
from .sourcing_interactions import (
    SourceGroupSectorInteraction, YFGroupProductInteraction,
    build_source_group_sector_interaction, build_yf_group_product_interaction,
)
from .state import MappingStateAdapter, SDAState, SDAStateAdapter, SourceGroupInputs, StateLabels
from .integration import EXIODeflatorStateAdapter
from .planning import ExecutionPlan, plan_sample_execution
from .sampled import SampledSDAStatistics, run_chunked_samples

__all__ = [
    "CANONICAL_EFFECTS",
    "EFFECT_HIERARCHY",
    "EffectDimensions",
    "EffectResult",
    "LabelSet",
    "SDAResult",
    "SourceCountryCube",
    "SourceGroupSectorCube",
    "YFSourceCountryCube",
    "YFSourceGroupSectorCube",
    "SourceGroupSectorInteraction",
    "YFGroupProductInteraction",
    "ValidationReport",
    "run_canonical_sda",
    "run_canonical_sda_state",
    "build_source_country_offshoring_cube",
    "build_source_group_sector_offshoring_cube",
    "build_yf_source_country_cube",
    "build_yf_source_group_sector_cube",
    "build_source_group_sector_interaction",
    "build_yf_group_product_interaction",
    "MappingStateAdapter",
    "SDAState",
    "SDAStateAdapter",
    "SourceGroupInputs",
    "StateLabels",
    "EXIODeflatorStateAdapter",
    "ExecutionPlan",
    "plan_sample_execution",
    "SampledSDAStatistics",
    "run_chunked_samples",
]
