# Country, industry and product concordances

Unchanged copies of the supplied thesis concordances. `manifest.json` records
SHA-256 hashes and CSV row counts so the published files can be checked against
their local sources. These are mappings, not raw MRIO matrices.

## Country and analytical-group mappings

- `countries_concordance.csv`: EXIOBASE, WIOD-2016 and WIOD-2013 country-code links.
- `kyoto_country_groups.csv`: supplied analytical-group assignments.
- `normalized_2026-08-31/country_concordance_by_resolution.csv` and
  `analytical_group_registry.csv`: reviewed country and analytical-group registries.

## Products, industries and sector aggregation

- `EXIOBASE3_product_to_WIOD_concordance.xlsx`: 200 EXIOBASE product rows, including
  EXIOBASE industry codes/names and WIOD-2016/WIOD-2013 target codes. The dated
  `EXIOBASE3_product_to_WIOD_concordance_2026-08-31.xlsx` is also retained as a
  distinct source snapshot. They are not assumed to be interchangeable.
- `EXIOBASE3_R3_aggregation_98group_2026-08-31.xlsx`: the thesis-specific 98-sector
  R3 comparison grid, including product/industry links and the group registry.
- `EXIOBASE3_R3_aggregation.xlsx`: the separate 96-group source variant. It is
  retained for traceability and must not substitute for the thesis's 98-group grid.
- `wiod2013_woid2016_exio.csv`: supplied industry-code crosswalk.
- `deflator_exio_concordance.csv`: deflator-product to EXIOBASE-product links;
  this is not the price-index dataset.
- `normalized_2026-08-31/sector_concordance_by_resolution.csv`: reviewed native
  sector registries. `granularity_sector_mapping_review.csv` and
  `fine_to_coarse_country_sector_review.csv` retain the reviewed cross-grid links.

## Price deflators

- `price_deflators_ultimate.xlsx`: supplied price-deflator workbook, copied
  unchanged. Use `deflator_exio_concordance.csv` for the associated product links.
  Its inclusion does not establish which deflator observations a historical run
  consumed; check the run configuration and price-basis metadata separately.

## Mapping boundaries

The normalized source validation manifest is retained unchanged. It identifies
four unresolved one-to-many relationships: R3 groups 54, 86 and 96 to WIOD-2016,
and R3 group 96 to WIOD-2013. Their targets remain blank; this package does not
invent allocation weights. Reviewed overrides for products C_RATV and C_MEIN
are recorded in that manifest. A supplied source workbook is not automatically
identical to its reviewed derivative.

The codebase's historical default R3 filename points to the 96-group variant;
the thesis uses the explicitly named 98-group comparison workbook. Select the
mapping matching the run and record its hash; these files alone do not establish
which workbook a historical run consumed. Country/sector resolution, units and
CYP/PYP price basis must be checked separately for each application.
