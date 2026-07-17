#!/usr/bin/env bash
# D: the missing (N,E)-clean controls. Weak-matched (--min-stratum weak) control +
# Mantel-Haenszel OR for the ladder rungs that lack one, so every rung has a
# (N,E)-clean number for the metric decision (A). Fast rungs first; masked-Acrobot
# (full-horizon) last.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python3; W=40; LOG=experiments/neclean_controls.log
echo "=== (N,E)-clean controls start $(date) ===" | tee -a "$LOG"
for t in cartpole mountaincar acrobot_masked; do
  echo "[$(date)] weak control: $t" | tee -a "$LOG"
  out="experiments/$t/matched_random/allpool_weak"
  $PY scripts/run_matched_random_control.py --task "$t" --min-stratum weak --num-random 5 \
     --max-workers $W --output-dir "$out" >> "experiments/$t/allpool_weak.log" 2>&1 \
     || { echo "  !! $t control FAILED" | tee -a "$LOG"; continue; }
  $PY scripts/analyze_mantel_haenszel.py --pool-dir "experiments/$t/pool" \
     --control-jsonl "$out/results.jsonl" --out "$out/mantel_haenszel.json" \
     >> "experiments/$t/allpool_weak.log" 2>&1 || echo "  !! $t MH FAILED" | tee -a "$LOG"
done
echo "=== (N,E)-clean controls done $(date) ===" | tee -a "$LOG"
