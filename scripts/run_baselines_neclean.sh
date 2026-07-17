#!/usr/bin/env bash
# (N,E)-clean CPPN baselines: weak-matched CPPN controls (--min-stratum weak) + MH OR
# (MorphoNAS pool vs CPPN), the Movement-2 analogue of the (N,E)-clean reach-map. Cells =
# the structured-control / memory rungs + the de-confounder pairs (so the MN-over-CPPN
# reward-gating can be re-expressed as a RoR of ORs too). Waits for the D batch
# (masked-Acrobot) to free the cores, then runs sequentially at 40 workers.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python3; W=40; LOG=experiments/baselines_neclean.log
echo "=== waiting for the D batch (masked-Acrobot) to finish ===" | tee -a "$LOG"
until grep -q "controls done" experiments/neclean_controls.log 2>/dev/null; do sleep 30; done
echo "=== CPPN (N,E)-clean baselines start $(date) ===" | tee -a "$LOG"
for t in cartpole cartpole_masked mountaincar mountaincar_shaped pendulum_sparse pendulum acrobot acrobot_shaped; do
  echo "[$(date)] cppn weak: $t" | tee -a "$LOG"
  out="experiments/baselines/cppn_hyperneat/$t/allpool_weak"
  $PY scripts/run_baseline_prior.py --task "$t" --encoding cppn_hyperneat --min-stratum weak \
     --num-random 5 --max-workers $W --output-dir "$out" \
     >> "experiments/baselines/cppn_hyperneat/$t/allpool_weak.log" 2>&1 \
     || { echo "  !! $t cppn FAILED" | tee -a "$LOG"; continue; }
  $PY scripts/analyze_mantel_haenszel.py --pool-dir "experiments/$t/pool" \
     --control-jsonl "$out/results.jsonl" --out "$out/mantel_haenszel.json" \
     >> "experiments/baselines/cppn_hyperneat/$t/allpool_weak.log" 2>&1 \
     || echo "  !! $t MH FAILED" | tee -a "$LOG"
done
echo "=== CPPN (N,E)-clean baselines done $(date) ===" | tee -a "$LOG"
