#!/usr/bin/env bash
set -euo pipefail

repo=/local/s2053268/ie-thesis/IE_Thesis_EXIO_WIOT
campaign=/local/s2053268/sda-campaign-2000-2014-v2
output="$campaign/native-wiod-co2-2001-2009-20260902"
smoke="$campaign/native-impact-smoke-20260902"

mkdir -p "$output"
export PYTHONPATH="$repo/SDA"
exec "$repo/.venv/bin/python" "$repo/scripts/run_native_wiod_campaign.py" \
  --repo "$repo" \
  --input-root "$campaign/build" \
  --output-root "$output" \
  --hierarchy-gate "$smoke/native-safe/hierarchy_validation.json" \
  --impact-gate "$smoke/effective_impact_jrc2019_to_2016_native_safe.json" \
  --impact-gate "$smoke/effective_impact_jrc2019_to_2013_native_safe.json" \
  --impact-gate "$smoke/effective_impact_wiod2013_native_safe.json" \
  --year-first 2001 --year-last 2009 --samples 8 --gpu 1 \
  >>"$output/launcher.log" 2>&1
