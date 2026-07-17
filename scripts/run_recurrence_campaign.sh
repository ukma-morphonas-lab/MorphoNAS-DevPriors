#!/usr/bin/env bash
# Recurrence campaign for the developmental-priors study (item 3):
#   Run B - functional ablation (does masked competence rely on cross-step memory?)
#   Run A - structural measure (is grown wiring more recurrent than matched random?)
# Launched under tmux; one line per run to experiments/recurrence_campaign.log,
# full run output appended there, per-run summaries in each output dir.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1   # repo root (scripts/ is one level down)
PY=.venv/bin/python
W=42
LOG=experiments/recurrence_campaign.log
: > "$LOG"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== recurrence campaign START (workers=$W) ==="

# ── Run B: functional recurrence ablation on the memory (masked-POMDP) rungs ──
# baseline must reproduce the committed masked competence rate (10.32% / 5.68%);
# state_reset removes cross-step memory. Collapse under state_reset == recurrence-borne.
for task in cartpole_masked acrobot_masked; do
  for mode in baseline state_reset; do
    say ">>> RUN B  task=$task mode=$mode"
    if $PY scripts/run_recurrence_ablation.py --task "$task" --recurrence-mode "$mode" \
         --max-workers "$W" >>"$LOG" 2>&1; then
      say "    ok: RUN B $task $mode"
    else
      say "    FAIL: RUN B $task $mode (rc=$?)"
    fi
  done
done

# ── Run A: structural recurrence across the gradient (control/memory/nav/densifiable) ──
for task in acrobot cartpole cartpole_masked acrobot_masked \
            mountaincar pendulum_sparse lunarlander_sparse frozenlake_shaped; do
  say ">>> RUN A  task=$task"
  if $PY scripts/analyze_recurrence_structure.py --task "$task" \
       --max-workers "$W" >>"$LOG" 2>&1; then
    say "    ok: RUN A $task"
  else
    say "    FAIL: RUN A $task (rc=$?)"
  fi
done

say "=== recurrence campaign DONE ==="
