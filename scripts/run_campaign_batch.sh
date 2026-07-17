#!/usr/bin/env bash
# Rigor-firming campaign — remaining compute batch.
#   Phase 2b: --min-stratum weak matched-random controls for the 6 Regime-B tasks
#             -> Mantel-Haenszel OR (the (N,E)-clean, denominator-symmetric headline
#             that replaces the competence-contaminated probability-of-superiority).
#   Baselines: bump the k=5 first-pass CPPN cells to k=50 (removes the "first pass" label).
# Sequential, 40 workers, continue-on-error, per-job logs. Run AFTER vision finishes.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
W=40
LOG=experiments/campaign_batch.log
echo "=== campaign batch start $(date) ===" | tee -a "$LOG"

REGIME_B="cartpole_masked acrobot_shaped pendulum_sparse pendulum lunarlander mountaincar_shaped"
echo "--- Phase 2b: weak-matched controls + Mantel-Haenszel OR ---" | tee -a "$LOG"
for t in $REGIME_B; do
  echo "[$(date)] Phase2b weak control: $t" | tee -a "$LOG"
  out="experiments/$t/matched_random/allpool_weak"
  $PY scripts/run_matched_random_control.py --task "$t" --min-stratum weak --num-random 5 \
     --max-workers $W --output-dir "$out" \
     >> "experiments/$t/allpool_weak.log" 2>&1 \
     || { echo "  !! $t control FAILED" | tee -a "$LOG"; continue; }
  $PY scripts/analyze_mantel_haenszel.py --pool-dir "experiments/$t/pool" \
     --control-jsonl "$out/results.jsonl" --out "$out/mantel_haenszel.json" \
     >> "experiments/$t/allpool_weak.log" 2>&1 \
     || echo "  !! $t MH FAILED" | tee -a "$LOG"
done

BASE_TASKS="acrobot acrobot_shaped cartpole cartpole_masked lunarlander pendulum pendulum_sparse mountaincar_shaped"
echo "--- Baselines: k=5 -> k=50 ---" | tee -a "$LOG"
for t in $BASE_TASKS; do
  echo "[$(date)] baseline k50: $t" | tee -a "$LOG"
  $PY scripts/run_baseline_prior.py --task "$t" --encoding cppn_hyperneat --min-stratum low_mid \
     --num-random 50 --max-workers $W --output-dir "experiments/baselines/cppn_hyperneat/$t/k50" \
     >> "experiments/baselines/cppn_hyperneat/$t/k50.log" 2>&1 \
     || echo "  !! $t baseline FAILED" | tee -a "$LOG"
done
echo "=== campaign batch done $(date) ===" | tee -a "$LOG"
