#!/usr/bin/env bash
# NAT-05: re-run the native-WIOD CO2 campaign with the EXIOBASE direct-intensity
# sanitization policy (sector-median replacement at 100x + absolute cap at 1e9),
# applied to all three native configurations. See
# docs/analysis/native_wiod_campaign_validation_20260903/NAT05_RERUN_PLAN.md
# for the full step sequence, including the smoke-gate regeneration this
# launcher depends on.  Decision D-2026-029.
#
# Runs under `nice -n 10` to yield CPU to other users on the shared host.
set -euo pipefail

repo=/local/s2053268/ie-thesis/IE_Thesis_EXIO_WIOT
campaign=/local/s2053268/sda-campaign-2000-2014-v2
stamp=$(date +%Y%m%d)
output="$campaign/native-wiod-co2-2001-2009-v2-$stamp"
smoke="$campaign/native-impact-smoke-v2-$stamp"   # produced by step 1 of the plan

# EXIOBASE-standard intensity policy (config.py defaults for IMPACT_DATABASE=EXIOBASE)
export SDA_V_SECTOR_OUTLIER_MULTIPLE=100
export SDA_V_INTENSITY_MAX=1000000000

mkdir -p "$output"
export PYTHONPATH="$repo/SDA"
exec nice -n 10 "$repo/.venv/bin/python" "$repo/scripts/run_native_wiod_campaign.py" \
  --repo "$repo" \
  --input-root "$campaign/build" \
  --output-root "$output" \
  --campaign-id "native-wiod-co2-2001-2009-v2" \
  --v-sector-outlier-multiple 100 \
  --v-intensity-max 1000000000 \
  --hierarchy-gate "$smoke/hierarchy_validation.json" \
  --impact-gate "$smoke/effective_impact_jrc2019_to_2016.json" \
  --impact-gate "$smoke/effective_impact_jrc2019_to_2013.json" \
  --impact-gate "$smoke/effective_impact_wiod2013_native.json" \
  --impact-gate "$smoke/effective_impact_jrc2019_flows2013.json" \
  --impact-gate "$smoke/effective_impact_wiod2013_flows2016.json" \
  --year-first 2001 --year-last 2009 --samples 8 --gpu 1 \
  >>"$output/launcher.log" 2>&1
