#!/usr/bin/env bash
# Push-button validation of the reach-map ladder scripts. Run locally before building
# the AMI, and again ON the bring-up machine before snapshotting, so the AMI is
# known-good. Tiny slices only (~3-4 min total). Asserts the load-bearing
# invariant: the generalized control reproduces the locked Acrobot script
# byte-for-byte, and shard+merge equals the unsharded union.
#
# Usage:  bash scripts/smoke_ladder.sh
# Needs:  .venv, experiments/acrobot/pool, experiments/acrobot/matched_random/full
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python3
W="${WORKERS:-8}"   # set WORKERS=2 on small boxes (e.g. t3.medium bring-up)
SM=experiments/_smoke_ladder
rm -rf "$SM"; mkdir -p "$SM"
trap 'rm -rf "$SM"' EXIT

echo "== [1/6] control: new vs locked (byte-equivalence) + shard/merge union =="
$PY scripts/run_matched_random_control.py --task acrobot --max-sources 30 --num-random 2 \
    --output-dir "$SM/new" --max-workers "$W" >/dev/null
$PY scripts/run_acrobot_matched_random.py --max-sources 30 --num-random 2 \
    --output-dir "$SM/orig" --max-workers "$W" >/dev/null
$PY scripts/run_matched_random_control.py --task acrobot --max-sources 30 --num-random 2 \
    --num-shards 2 --shard-index 0 --output-dir "$SM/shards/shard0" --max-workers "$W" >/dev/null
$PY scripts/run_matched_random_control.py --task acrobot --max-sources 30 --num-random 2 \
    --num-shards 2 --shard-index 1 --output-dir "$SM/shards/shard1" --max-workers "$W" >/dev/null
$PY scripts/merge_shards.py --mode control --shard-glob "$SM/shards/shard*" \
    --pool-dir experiments/acrobot/pool --out-dir "$SM/merged" >/dev/null
$PY - "$SM" <<'PY'
import json, sys
SM = sys.argv[1]
def load(p): return {(r["source_id"], r["random_seed"]): r for r in (json.loads(l) for l in open(p))}
new, orig = load(f"{SM}/new/results.jsonl"), load(f"{SM}/orig/results.jsonl")
keys = set(new) & set(orig)
err = max(abs(new[k].get("baseline_reward", 0) - orig[k].get("baseline_reward", 0))
          for k in keys if new[k].get("valid"))
merged = {(r["source_id"], r["random_seed"]) for r in (json.loads(l) for l in open(f"{SM}/merged/results.jsonl"))}
assert err == 0, f"reward drift vs locked script: {err}"
assert merged == set(new), "shard merge != unsharded union"
print(f"   PASS  max|reward diff vs locked|={err}  shard-union-equal=True  ({len(keys)} graphs)")
PY

echo "== [2/6] pool gen: mountaincar + lunarlander (tiny; calibration read) =="
for t in mountaincar lunarlander; do
  $PY scripts/run_pool.py --task "$t" --target-valid 12 --output-dir "$SM/pool_$t" --max-workers "$W" >/dev/null
  $PY - "$SM/pool_$t/pool_metadata.json" "$t" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); rs = m.get("reward_summary", {})
print(f"   {sys.argv[2]:12s} valid={m['valid_count']} comp_frac={m['competent_fraction']*100:.1f}% "
      f"gate={'PASS' if m['gate_pass'] else 'FAIL'} reward[min={rs.get('min')},med={rs.get('median')},max={rs.get('max')}]")
PY
done

echo "== [3/6] weight extraction (perfect subset) =="
$PY scripts/analyze_mn_weight_distribution.py --task acrobot --min-stratum perfect \
    --output-dir "$SM/wdist" >/dev/null
NPY="$SM/wdist/mn_edge_weights.npy"

echo "== [4/6] weight-ablation cells (max-sources 4) + decision rule =="
$PY scripts/run_weight_ablation.py --task acrobot --topology mn_regrow --weight-mode mn_keep \
    --num-random 2 --max-sources 4 --output-dir "$SM/ab/mn_regrow__mn_keep" --max-workers "$W" >/dev/null
$PY scripts/run_weight_ablation.py --task acrobot --topology mn_regrow --weight-mode uniform \
    --num-random 2 --max-sources 4 --output-dir "$SM/ab/mn_regrow__uniform" --max-workers "$W" >/dev/null
$PY scripts/run_weight_ablation.py --task acrobot --topology random --weight-mode uniform \
    --num-random 2 --max-sources 4 --output-dir "$SM/ab/random__uniform" --max-workers "$W" >/dev/null
$PY scripts/run_weight_ablation.py --task acrobot --topology random --weight-mode mn_empirical \
    --num-random 2 --max-sources 4 --empirical-weights "$NPY" \
    --output-dir "$SM/ab/random__mn_empirical" --max-workers "$W" >/dev/null
$PY scripts/analyze_weight_ablation.py --ablation-dir "$SM/ab" --out "$SM/ab/verdict.json" \
    | grep -E "VERDICT|baseline|DECISIVE|floor" | sed 's/^/   /'

echo "== [5/6] cross-task RoR on existing k=5 full run =="
$PY scripts/analyze_ratios.py acrobot=experiments/acrobot/matched_random/full/summary_stats.json \
    --out "$SM/ror.json" | grep -E "GROWS|persists|killed" | sed 's/^/   /'

echo "== [6/6] Mantel-Haenszel code path (on existing low_mid full; smoke only) =="
$PY scripts/analyze_mantel_haenszel.py --pool-dir experiments/acrobot/pool \
    --control-jsonl experiments/acrobot/matched_random/full/results.jsonl \
    --out "$SM/mh.json" | grep -E "MH odds|artifact" | sed 's/^/   /'

echo "ALL SMOKE CHECKS PASSED"
