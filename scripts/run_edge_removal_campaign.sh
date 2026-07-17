#!/usr/bin/env bash
# Within-step recurrence test with a capacity control.
#   baseline            : grown net as-is (reproduces committed rate)
#   edge_removal        : DAG-ify (kill the recurrent core)         -> ~20% edges gone
#   random_edge_removal : drop the SAME NUMBER of edges at random   -> capacity control
# If edge_removal collapses competence but random_edge_removal does not, the
# recurrent core is load-bearing (not merely edge capacity). CartPole family first
# (fast, decisive preview), then the strong Acrobot rung.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # repo root (scripts/ is one level down)
PY=.venv/bin/python
W=42
LOG=experiments/recurrence_edge_campaign.log
: > "$LOG"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run() {  # task, mode
  say ">>> $1 $2"
  if $PY scripts/run_recurrence_ablation.py --task "$1" --recurrence-mode "$2" --max-workers "$W" >>"$LOG" 2>&1; then
    say "    ok: $1 $2"
  else
    say "    FAIL: $1 $2 (rc=$?)"
  fi
}

say "=== edge_removal campaign START (workers=$W) ==="
# CartPole family (fast): full three-way on control + masked
run cartpole        baseline
run cartpole        edge_removal
run cartpole        random_edge_removal
run cartpole_masked edge_removal
run cartpole_masked random_edge_removal
# Acrobot (strong control rung, ~30 min/mode): the decisive three-way
run acrobot         baseline
run acrobot         edge_removal
run acrobot         random_edge_removal
say "=== edge_removal campaign DONE ==="
