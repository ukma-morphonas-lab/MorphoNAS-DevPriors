#!/usr/bin/env python3
"""
Bar-policy v2 prototype (Regime-B): compare the current p90-of-MN non-weak bar
against behaviorally-anchored bars, from STORED rewards (no re-evaluation).

For each task:
  * Tier 1 (bar-free): probability of superiority A = P(MN reward > random reward)
    paired by (N,E) via source_id (Cliff's delta = 2A-1). Immune to bar choice.
  * Tier 2 (competence ratio): R = MN-rate / random-rate at a behaviorally-anchored
    bar (null-floor+margin or bimodal trough), with a MOVER+Wilson CI, next to the
    current p90 R for the same data.
  * dose-response: R across a small bar grid (shows the cut is non-load-bearing).

Read-only analysis. Usage: .venv/bin/python3 scripts/analyze_bar_policy.py
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "code")
from MorphoNAS_DevPriors.ratio_stats import ratio_ci_mover  # noqa: E402
from MorphoNAS_DevPriors.task_registry import get_task  # noqa: E402


def load_mn_pool(task: str):
    """Return dict id->reward and arrays of (reward,N,E). Keyed by every plausible
    id (network_id, seed) so the random arm's source_id resolves whatever it used."""
    rewards, ne = [], []
    by_id = {}
    files = sorted(glob.glob(f"experiments/{task}/pool/networks/*.json"))
    for fp in files:
        with open(fp) as f:
            j = json.load(f)
        r = j.get("baseline_reward")
        if r is None:
            continue
        rewards.append(r)
        ns = j.get("network_stats") or {}
        ne.append((ns.get("neurons"), ns.get("connections")))
        for key in (j.get("network_id"), j.get("seed"),
                    os.path.splitext(os.path.basename(fp))[0]):
            if key is not None:
                by_id[key] = r
                by_id[str(key)] = r
    return by_id, np.asarray(rewards, dtype=float), ne, len(files)


def load_random(task: str, sub: str):
    rows = []
    fp = f"experiments/{task}/matched_random/{sub}/results.jsonl"
    if not os.path.exists(fp):
        return rows, fp
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("valid", True):
                continue
            br = r.get("baseline_reward")
            if br is None:
                rws = r.get("rewards")
                if not rws:
                    continue
                br = float(np.mean(rws))
            rows.append((float(br), r.get("num_neurons"),
                         r.get("num_connections"), r.get("source_id"),
                         r.get("source_stratum")))
    return rows, fp


def r_at_bar(mn_rewards: np.ndarray, rand_rewards: np.ndarray, bar: float) -> dict:
    mn_x = int((mn_rewards >= bar).sum())
    rd_x = int((rand_rewards >= bar).sum())
    ci = ratio_ci_mover(mn_x, len(mn_rewards), rd_x, len(rand_rewards))
    return ci


def prob_superiority(mn_by_id: dict, rand_rows: list) -> tuple:
    """A = P(MN_source > random_graph) over matched (source, graph) pairs."""
    wins = 0.0
    n = 0
    miss = 0
    for rwd, _N, _E, sid, _ss in rand_rows:
        m = mn_by_id.get(sid, mn_by_id.get(str(sid)))
        if m is None:
            miss += 1
            continue
        n += 1
        if m > rwd:
            wins += 1.0
        elif m == rwd:
            wins += 0.5
    return (wins / n if n else float("nan")), n, miss


def do_nothing_floor(task: str, episodes: int = 20) -> dict:
    """Constant-action and random-action episode-length floor for a control task."""
    spec = get_task(task)
    env = spec.make_env()
    out = {}
    try:
        n_actions = env.action_space.n
    except Exception:
        n_actions = 2
    for label, pol in [("const0", lambda o: 0),
                       ("const1", lambda o: min(1, n_actions - 1)),
                       ("uniform", None)]:
        lengths = []
        rng = np.random.default_rng(12345)
        for ep in range(episodes):
            obs, _ = env.reset(seed=10_000 + ep)
            done = False
            steps = 0
            while not done:
                a = int(rng.integers(n_actions)) if pol is None else pol(obs)
                obs, _r, term, trunc, _ = env.step(a)
                steps += 1
                done = term or trunc
            lengths.append(steps)
        out[label] = float(np.mean(lengths))
    env.close()
    return out


def nonweak_threshold(task: str) -> float:
    spec = get_task(task)
    # the 'weak' stratum high bound = the non-weak cut
    for name, low, high in spec.stratum_bounds:
        if name == "weak":
            return high
    return float("nan")


def summarize(task: str, sub: str, new_bars: list, label_new: str):
    print("=" * 78)
    print(f"TASK {task}   (random sub-run: {sub})")
    print("=" * 78)
    mn_by_id, mn_rw, mn_ne, n_files = load_mn_pool(task)
    rand_rows, rfp = load_random(task, sub)
    if not rand_rows:
        print(f"  no random results at {rfp}; skipping")
        return None
    rand_rw = np.asarray([r[0] for r in rand_rows], dtype=float)
    p90 = nonweak_threshold(task)

    def pct(a):
        q = np.percentile(a, [0, 10, 50, 90, 99, 100])
        return "min {:.1f} p10 {:.1f} med {:.1f} p90 {:.1f} p99 {:.1f} max {:.1f}".format(*q)

    print(f"  MN pool n={n_files} (reward {pct(mn_rw)})")
    print(f"  random n={len(rand_rw)} (reward {pct(rand_rw)})")
    ss = {}
    for _r, _n, _e, _s, st in rand_rows:
        ss[st] = ss.get(st, 0) + 1
    print(f"  random source_stratum: {ss}")

    a, npair, miss = prob_superiority(mn_by_id, rand_rows)
    print(f"\n  Tier-1  probability of superiority A = P(MN>rand) = {a:.4f}  "
          f"(Cliff's delta {2*a-1:+.3f}; {npair} paired, {miss} unmatched)")
    if miss > 0.5 * (npair + miss):
        print("    !! >50% random rows unmatched to a pool id — check source_id keying")

    print(f"\n  Tier-2  competence ratio R = MN-rate / random-rate")
    print(f"  {'bar':>10} {'MN rate':>12} {'rand rate':>14} {'R':>9}  [95% CI]   note")
    rows = [(p90, "CURRENT p90")] + [(b, label_new if i == 0 else "") for i, b in enumerate(new_bars)]
    for bar, note in rows:
        ci = r_at_bar(mn_rw, rand_rw, bar)
        hi = ci["hi"]
        hi_s = "inf" if hi is None else f"{hi:.1f}"
        pt = ci["point"]
        pt_s = "inf" if pt == float("inf") else f"{pt:.2f}"
        print(f"  {bar:>10.1f} {ci['mn_rate']*100:>10.2f}% ({ci['mn_count']}) "
              f"{ci['rand_rate']*100:>9.3f}% ({ci['rand_count']}) {pt_s:>8}x"
              f"  [{ci['lo']:.1f}, {hi_s}]  {note}")
    p90_ci = r_at_bar(mn_rw, rand_rw, p90)
    new_bar = new_bars[0] if new_bars else None
    new_ci = r_at_bar(mn_rw, rand_rw, new_bar) if new_bar is not None else None
    return {"task": task, "A": a, "npair": npair, "p90": p90,
            "p90_R": p90_ci["point"], "p90_lo": p90_ci["lo"],
            "new_bar": new_bar,
            "new_R": (new_ci["point"] if new_ci else None),
            "new_lo": (new_ci["lo"] if new_ci else None)}


def main():
    # masked CartPole: trivial-baseline anchored. The conservative null is the BEST
    # non-learning policy: random-action (22.4) beats constant-action (9.4) here.
    print("\n##### trivial-policy floors for cartpole_masked #####")
    floor = do_nothing_floor("cartpole_masked")
    f_const = min(floor["const0"], floor["const1"])
    f_rand = floor["uniform"]
    anchor = round(f_rand)
    print(f"  floors: constant-action {f_const:.1f}, random-action {f_rand:.1f}"
          f"  -> anchor (beats-random) = {anchor}")
    mc_bars = [float(anchor), 28.0, round(1.5 * f_rand), 40.0, 44.0]

    # Regime-B config: Tier-1 A for all; Tier-2 competence bar where one is meaningful.
    cfgs = [
        dict(task="cartpole_masked", sub="full_k50", role="memory advantage",
             bars=mc_bars, label=f"BEATS-RANDOM (>={anchor})"),
        dict(task="acrobot_shaped", sub="full", role="non-densifiable control",
             bars=[-100.0, 0.0, 50.0], label="TROUGH (>=0, bimodal gap)"),
        dict(task="pendulum_sparse", sub="full_k50", role="dose-response instrument",
             bars=[13.0, 17.0, 20.0, 24.0], label="upright-count grid"),
        dict(task="pendulum", sub="full", role="densified inversion companion",
             bars=[], label="(Tier-1 only)"),
        dict(task="lunarlander", sub="full", role="parity rung",
             bars=[], label="(Tier-1 only)"),
        dict(task="mountaincar_shaped", sub="full", role="densified inversion",
             bars=[], label="(Tier-1 only)"),
    ]
    results = []
    for c in cfgs:
        r = summarize(c["task"], c["sub"], c["bars"], c["label"])
        if r:
            r["role"] = c["role"]
            results.append(r)

    print("\n" + "=" * 78)
    print("REGIME-B GRADIENT — bar-free Tier-1 A recovers advantage / parity / inversion")
    print("=" * 78)
    print(f"  {'task':>20} {'role':>30} {'A=P(MN>rand)':>13}  reading")
    for r in results:
        a = r["A"]
        rd = "ADVANTAGE" if a > 0.6 else ("PARITY" if a >= 0.45 else "INVERSION")
        print(f"  {r['task']:>20} {r['role']:>30} {a:>13.4f}  {rd}  ({r['npair']} pairs)")
    print("\n  Tier-2 competence ratio (p90 -> behavioral bar), where one applies:")
    for r in results:
        if r["new_R"] is not None:
            f = lambda x: "inf" if x == float("inf") else f"{x:.1f}x"
            print(f"  {r['task']:>20}: p90 {f(r['p90_R'])}  ->  @{r['new_bar']:g} {f(r['new_R'])}")


if __name__ == "__main__":
    main()
