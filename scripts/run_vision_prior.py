#!/usr/bin/env python3
"""
Vision rung (feedforward feature composition): the un-evolved-prior matched-random
control on a static image-classification task, on THIS repo's canonical CPU engine
(no GPU bridge needed -- same ruleset as every other rung, just a non-gym eval).

Classification is a one-shot forward map (image -> propagate -> argmax over the
output nodes -> predicted class), with the network reset per image (images are
independent; no temporal structure). This is the family where a recurrence-supplying
prior has the least obvious purchase -- the predicted-null / sharpening counterpoint
to control and memory.

Dataset: sklearn 8x8 digits (64 features, 10 balanced classes; chance ~10%), pixels
normalized to [0,1]. For each random-genome MN network we grow it, score test
accuracy, then generate k topology-matched random directed graphs (same N,E,
weights ~U[0.01,1], weakly connected) and score them on the SAME test set. R =
MN-competent-rate / matched-random-competent-rate at a competence bar above chance.

Output: <output-dir>/{mn.jsonl, random.jsonl, summary.json}.
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
from MorphoNAS.genome import Genome  # noqa: E402
from MorphoNAS.grid import Grid  # noqa: E402
from MorphoNAS.neural_propagation import NeuralPropagator  # noqa: E402
from MorphoNAS_DevPriors.random_graph import generate_random_rnn  # noqa: E402
from MorphoNAS_DevPriors.ratio_stats import fisher_within_task, ratio_ci_mover  # noqa: E402

INPUT_DIM, OUTPUT_DIM = 64, 10
_W = {}


def _load_digits(test_size, seed):
    from sklearn.datasets import load_digits
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)  # [0,1]
    y = d.target.astype(int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    te = idx[:test_size]
    return X[te], y[te]


def _make_prop(G):
    return NeuralPropagator(G=G, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                            activation_function=NeuralPropagator.tanh_activation,
                            extra_thinking_time=2, additive_update=False)


def _accuracy(prop, Xte, yte):
    preds = np.empty(len(Xte), dtype=int)
    for i in range(len(Xte)):
        prop.reset()
        prop.propagate(Xte[i])
        preds[i] = int(prop.get_output().argmax())
    return float((preds == yte).mean()), preds


def _init_worker(test_size, test_seed, min_edges):
    _W["X"], _W["y"] = _load_digits(test_size, test_seed)
    _W["min_edges"] = int(min_edges)
    _W["min_neurons"] = INPUT_DIM + OUTPUT_DIM


def _grow_eval(args):
    idx, seed, gp = args
    g = Genome.random(rng=np.random.default_rng(seed), size_x=gp[0], size_y=gp[1],
                      max_growth_steps=gp[2], num_morphogens=gp[3])
    grid = Grid(g); grid.run_simulation(verbose=False)
    G = grid.get_graph()
    n, e = G.number_of_nodes(), G.number_of_edges()
    if n < _W["min_neurons"] or e < _W["min_edges"]:
        return {"id": idx, "seed": seed, "valid": False, "n": n, "e": e}
    acc, preds = _accuracy(_make_prop(G), _W["X"], _W["y"])
    # collapse = predicts (almost) one class -> input-insensitive
    frac_mode = float(np.bincount(preds, minlength=OUTPUT_DIM).max() / len(preds))
    return {"id": idx, "seed": seed, "valid": True, "acc": acc, "n": int(n),
            "e": int(e), "frac_mode": frac_mode}


def _rand_eval(args):
    src_id, n, e, gen_seed = args
    G = generate_random_rnn(n, e, np.random.default_rng(gen_seed),
                            weight_range=(0.01, 1.0), max_retries=100)
    if G is None:
        return {"src_id": src_id, "valid": False}
    acc, preds = _accuracy(_make_prop(G), _W["X"], _W["y"])
    frac_mode = float(np.bincount(preds, minlength=OUTPUT_DIM).max() / len(preds))
    return {"src_id": src_id, "valid": True, "acc": acc, "n": int(n), "e": int(e),
            "frac_mode": frac_mode}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-nets", type=int, default=500)
    p.add_argument("--num-random", type=int, default=5)
    p.add_argument("--test-size", type=int, default=500)
    p.add_argument("--test-seed", type=int, default=20260601)
    p.add_argument("--pool-seed-start", type=int, default=12_000_001)
    p.add_argument("--control-base-seed", type=int, default=12_200_000)
    p.add_argument("--bar", type=float, default=0.20, help="competence bar: accuracy >= bar (chance ~0.10)")
    p.add_argument("--min-edges", type=int, default=5)
    p.add_argument("--size-x", type=int, default=20)
    p.add_argument("--size-y", type=int, default=20)
    p.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    p.add_argument("--output-dir", type=str, default="experiments/vision_digits/prior")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    gp = (args.size_x, args.size_y, 200, 3)

    # majority-class baseline on the test set (the trivial constant-predictor accuracy)
    Xte, yte = _load_digits(args.test_size, args.test_seed)
    maj = float(np.bincount(yte).max() / len(yte))
    print(f"digits test: n={len(yte)} chance~{1/OUTPUT_DIM:.3f} majority-class={maj:.3f} | bar={args.bar}")

    # ---- MN pool ----
    mn_tasks = [(i + 1, args.pool_seed_start + i, gp) for i in range(args.num_nets)]
    mn = []
    with Pool(args.max_workers, initializer=_init_worker,
              initargs=(args.test_size, args.test_seed, args.min_edges)) as pool:
        for r in pool.imap_unordered(_grow_eval, mn_tasks, chunksize=4):
            mn.append(r)
    with open(os.path.join(args.output_dir, "mn.jsonl"), "w") as f:
        for r in mn:
            f.write(json.dumps(r) + "\n")
    mnv = [r for r in mn if r.get("valid")]
    mn_acc = np.array([r["acc"] for r in mnv])
    mn_comp = int((mn_acc >= args.bar).sum())
    print(f"MN: valid={len(mnv)}/{len(mn)} acc mean={mn_acc.mean():.3f} max={mn_acc.max():.3f} "
          f"p90={np.percentile(mn_acc,90):.3f} | >=bar: {mn_comp} ({mn_comp/len(mnv)*100:.2f}%) "
          f"| mean frac_mode={np.mean([r['frac_mode'] for r in mnv]):.2f}")

    # ---- matched-random control ----
    rand_tasks = []
    for r in mnv:
        for k in range(args.num_random):
            rand_tasks.append((r["id"], r["n"], r["e"], args.control_base_seed + r["id"] * 100 + k))
    rnd = []
    with Pool(args.max_workers, initializer=_init_worker,
              initargs=(args.test_size, args.test_seed, args.min_edges)) as pool:
        for r in pool.imap_unordered(_rand_eval, rand_tasks, chunksize=8):
            rnd.append(r)
    with open(os.path.join(args.output_dir, "random.jsonl"), "w") as f:
        for r in rnd:
            f.write(json.dumps(r) + "\n")
    rv = [r for r in rnd if r.get("valid")]
    r_acc = np.array([r["acc"] for r in rv])
    r_comp = int((r_acc >= args.bar).sum())
    print(f"random: valid={len(rv)}/{len(rand_tasks)} acc mean={r_acc.mean():.3f} max={r_acc.max():.3f} "
          f"| >=bar: {r_comp} ({r_comp/len(rv)*100:.3f}%) "
          f"| mean frac_mode={np.mean([r['frac_mode'] for r in rv]):.2f}")

    ci = ratio_ci_mover(mn_comp, len(mnv), r_comp, len(rv))
    fp = fisher_within_task(mn_comp, len(mnv), r_comp, len(rv))
    summary = {
        "dataset": "sklearn_digits_8x8", "input_dim": INPUT_DIM, "output_dim": OUTPUT_DIM,
        "test_size": len(yte), "chance": 1.0 / OUTPUT_DIM, "majority_class": maj, "bar": args.bar,
        "mn": {"n": len(mnv), "acc_mean": float(mn_acc.mean()), "acc_max": float(mn_acc.max()),
               "competent": mn_comp, "rate": mn_comp / len(mnv)},
        "random": {"n": len(rv), "acc_mean": float(r_acc.mean()), "acc_max": float(r_acc.max()),
                   "competent": r_comp, "rate": r_comp / len(rv)},
        "R": ci, "fisher": fp,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    hi = f"{ci['hi']:.2f}" if ci['hi'] else "inf"
    print(f"R(acc>=bar) = {ci['point']:.2f}x [{ci['lo']:.2f}, {hi}]  Fisher p={fp['p_one_sided_greater']:.2e}")
    print(f"-> {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
