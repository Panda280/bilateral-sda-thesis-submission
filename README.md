# Bilateral SDA Attribution

Dimension-aware structural decomposition analysis for multiregional input–output
systems. This thesis snapshot separates code and user documentation from internal
planning and development records.

## Thesis scope

The manuscript examines temporal aggregation, geographical relations and sectoral
aggregation within EXIOBASE, using IPCC AR4 GWP100 and transitions ending in
2001–2012. WIOD-compatible classifications provide comparison resolutions.

## Installation and checks

Python 3.11 or later is required.

```console
python -m pip install -e ".[test,parquet]"
python -m pytest
python -m sda --help
```

The default tests use synthetic CPU arrays. Large MRIO runs require separately
obtained data, the corresponding price-basis inputs and configured execution paths.
Analysis scripts may require additional libraries and campaign files; the minimal
package installation is not a complete empirical campaign environment.

## Effect hierarchy

```text
ΔG = D_v + D_C + D_A_x + D_YF + D_YB + D_y
D_A_x = D_T_coef + D_w_coef
```

The second identity is nested: the parent and its children are not independent
additive effects. Effects describe accounting attribution, not causal estimates.

## Repository contents

- `src/sda/`: decomposition, attribution, aggregation, reporting and validation.
- `scripts/`: campaign and thesis analysis utilities.
- `tests/`: synthetic and regression checks.
- `config/` and `profiles/`: supplied mappings and example settings.
- `notebooks/`: exploration notebooks with stored execution outputs removed.
- [`concordance/`](concordance/): country, industry and product mappings, with source hashes and mapping boundaries.
