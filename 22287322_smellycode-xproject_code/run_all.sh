#!/usr/bin/env bash
# Runs the full pipeline. Usage: bash run_all.sh <path to SmellyCode++ CSV> [work dir]
# Your outputs go to the work dir (default: work/); results/ holds the outputs of the reported run for comparison.
set -euo pipefail
CSV="${1:?give the path to the SmellyCode++ CSV}"
W="${2:-work}"
mkdir -p "$W"
# ---- primary protocol (pre-specified) ----
python src/step1_clean_group.py "$CSV" "$W" | tee "$W/step1.log"       # ~2 min
python src/cleaning_records.py "$CSV" "$W" | tee "$W/cleaning_records.log"   # ~1 min; per-sample record of step 1 (Table I, upper panel)
python src/step2_split.py "$W"              | tee "$W/step2.log"       # seconds
python src/clusters.py "$W"                 | tee "$W/clusters.log"    # seconds
python src/step3_tune.py "$W"               | tee "$W/step3.log"       # ~30 min on 2 CPU cores
python src/step4_test.py "$W"               | tee "$W/step4.log"       # ~8 min
python src/step5_contrast.py "$W" "$CSV"    | tee "$W/step5.log"       # ~5 min
# ---- post hoc, exploratory: granularity-matched analysis, checks of the rule, the bootstrap, the populations and the parser-unclassified samples, cleaning sensitivity ----
python src/granularity.py "$W"              | tee "$W/granularity.log" # seconds
python src/posthoc_tune.py "$W"             | tee "$W/posthoc_tune.log"   # ~15 min
python src/posthoc_test.py "$W"             | tee "$W/posthoc_test.log"   # ~6 min
python src/validate_granularity.py "$W"     | tee "$W/granularity_validation.log"   # ~12 min on 2 CPU cores
python src/bootstrap_details.py "$W" "$CSV" | tee "$W/bootstrap_details.log"      # ~4 min
python src/population_check.py "$W"        | tee "$W/population_check.log"       # seconds; primary models on the matched populations
python src/unparsed_check.py --prepare "$W" "$W/unverified"                  # seconds; step H: work dir without the 873 parser-unclassified samples
python src/posthoc_tune.py "$W/unverified"  | tee "$W/unverified/posthoc_tune.log"   # ~15 min; steps B and C repeated without them
python src/posthoc_test.py "$W/unverified"  | tee "$W/unverified/posthoc_test.log"   # ~6 min
python src/unparsed_check.py "$W" "$W/unverified" | tee "$W/unparsed_check.log"     # ~1 min; their influence on the post hoc results
python src/cleaning_sensitivity.py "$CSV" "$W" | tee "$W/cleaning_sensitivity.log"   # ~2 min
python src/step6_facts.py "$CSV" "$W"       | tee "$W/facts.log"       # ~1 min
python src/fig3.py "$W/results.json" "$W/posthoc_results.json" "$W/fig3.png"
echo "done: compare the files in $W with those in results/ (python src/compare_outputs.py $W results)"
