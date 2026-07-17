#!/usr/bin/env python3
"""
CPPN / HyperNEAT vision arm: the indirect-encoding baseline for the 8x8-digits
vision rung (the non-gym, image->propagate->argmax eval). It REUSES the existing
MN arm (experiments/vision_digits/prior/mn.jsonl) and the exact digit eval from
run_vision_prior.py, swapping the matched-random generator (generate_random_rnn)
for sample_cppn_hyperneat. (N,E)-matched per valid MN net; the same weak-
connectivity drop as the random arm (an MN net with e < n-1 cannot be matched by
a weakly-connected graph, so it returns None and is excluded -- in both arms).

Output: experiments/baselines/cppn_hyperneat/vision_digits/{cppn.jsonl, summary.json}
with R(MN/CPPN) at the same accuracy bar the vision rung used.
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.append(os.path.abspath("code"))
sys.path.append(os.path.abspath("scripts"))

import run_vision_prior as rvp  # noqa: E402  (reuse the exact digit eval)
from MorphoNAS_DevPriors.baseline_encoders import sample_cppn_hyperneat  # noqa: E402
from MorphoNAS_DevPriors.ratio_stats import fisher_within_task, ratio_ci_mover  # noqa: E402

_W = {}


def _init_worker(test_size, test_seed, cppn_hidden):
    _W["X"], _W["y"] = rvp._load_digits(test_size, test_seed)
    _W["cppn_hidden"] = int(cppn_hidden)


def _cppn_eval(args):
    src_id, n, e, gen_seed = args
    G = sample_cppn_hyperneat(n, e, rvp.INPUT_DIM, rvp.OUTPUT_DIM,
                              np.random.default_rng(gen_seed), cppn_hidden=_W["cppn_hidden"])
    if G is None:
        return {"src_id": src_id, "valid": False}
    acc, preds = rvp._accuracy(rvp._make_prop(G), _W["X"], _W["y"])
    frac_mode = float(np.bincount(preds, minlength=rvp.OUTPUT_DIM).max() / len(preds))
    return {"src_id": src_id, "valid": True, "acc": acc, "n": int(n), "e": int(e),
            "frac_mode": frac_mode}


def main():
    p = argparse.ArgumentParser(description="CPPN/HyperNEAT vision baseline (8x8 digits)")
    p.add_argument("--mn-jsonl", default="experiments/vision_digits/prior/mn.jsonl")
    p.add_argument("--cppn-hidden", type=int, default=4)
    p.add_argument("--num-random", type=int, default=10, help="CPPN graphs per valid MN net")
    p.add_argument("--test-size", type=int, default=500)
    p.add_argument("--test-seed", type=int, default=20260601)
    p.add_argument("--bar", type=float, default=0.15)
    p.add_argument("--control-base-seed", type=int, default=912_200_000)
    p.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    p.add_argument("--output-dir", default="experiments/baselines/cppn_hyperneat/vision_digits")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # MN arm (reused): valid nets + competence at the bar.
    mnv = [json.loads(l) for l in open(args.mn_jsonl) if json.loads(l).get("valid")]
    mn_acc = np.array([r["acc"] for r in mnv])
    mn_comp = int((mn_acc >= args.bar).sum())
    print(f"MN (reused): valid={len(mnv)} competent(>={args.bar})={mn_comp} "
          f"({mn_comp/len(mnv)*100:.2f}%) acc_mean={mn_acc.mean():.3f} max={mn_acc.max():.3f}")

    # CPPN arm: k graphs per valid MN net at its (n,e).
    tasks = []
    for r in mnv:
        for k in range(args.num_random):
            tasks.append((r["id"], int(r["n"]), int(r["e"]), args.control_base_seed + r["id"] * 100 + k))
    cppn = []
    with Pool(args.max_workers, initializer=_init_worker,
              initargs=(args.test_size, args.test_seed, args.cppn_hidden)) as pool:
        for r in pool.imap_unordered(_cppn_eval, tasks, chunksize=8):
            cppn.append(r)
    with open(os.path.join(args.output_dir, "cppn.jsonl"), "w") as f:
        for r in cppn:
            f.write(json.dumps(r) + "\n")

    cv = [r for r in cppn if r.get("valid")]
    c_acc = np.array([r["acc"] for r in cv]) if cv else np.array([0.0])
    c_comp = int((c_acc >= args.bar).sum())
    print(f"CPPN: valid={len(cv)}/{len(tasks)} competent={c_comp} ({c_comp/max(len(cv),1)*100:.3f}%) "
          f"acc_mean={c_acc.mean():.3f} max={c_acc.max():.3f} "
          f"mean frac_mode={np.mean([r['frac_mode'] for r in cv]) if cv else float('nan'):.2f}")

    ci = ratio_ci_mover(mn_comp, len(mnv), c_comp, len(cv))
    fp = fisher_within_task(mn_comp, len(mnv), c_comp, len(cv))
    summary = {
        "experiment": "baseline_prior", "arm": "cppn_hyperneat", "task": "vision_digits",
        "dataset": "sklearn_digits_8x8", "bar": args.bar, "chance": 1.0 / rvp.OUTPUT_DIM,
        "encoding_config": {"encoding": "cppn_hyperneat", "cppn_hidden": args.cppn_hidden,
                            "ne_matched": True, "weight_range": [0.01, 1.0]},
        # mirror the gym-runner schema so analyze_ratios.py can read it too
        "morphonas": {"n_total": len(mnv), "nonweak_count": mn_comp,
                      "nonweak_rate": mn_comp / len(mnv), "solved_count": 0, "solved_rate": 0.0},
        "n_random_valid": len(cv), "random_nonweak_count": c_comp, "random_solved_count": 0,
        "mn": {"n": len(mnv), "competent": mn_comp, "acc_mean": float(mn_acc.mean())},
        "cppn": {"n": len(cv), "competent": c_comp, "acc_mean": float(c_acc.mean()),
                 "acc_max": float(c_acc.max()),
                 "mean_frac_mode": float(np.mean([r["frac_mode"] for r in cv])) if cv else None},
        "ratio_nonweak": ci, "fisher": fp,
    }
    with open(os.path.join(args.output_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    hi = f"{ci['hi']:.2f}" if ci["hi"] else "inf"
    print(f"R(MN/CPPN, acc>={args.bar}) = {ci['point']:.2f}x [{ci['lo']:.2f}, {hi}] "
          f"Fisher p={fp['p_one_sided_greater']:.2e} -> {args.output_dir}/summary_stats.json")


if __name__ == "__main__":
    main()
