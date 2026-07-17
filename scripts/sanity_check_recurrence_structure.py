#!/usr/bin/env python3
"""
Sanity-check analyze_recurrence_structure.py before it reshapes the central claim.

1) Anchor tests: SCC/spectral on hand-built graphs with known answers.
2) Real nets: for a sample of grown acrobot nets, report actual (N,E) vs the
   stored network_stats (the (N,E)-match guard), weakly-connected fragmentation,
   isolated nodes, and largest-SCC fraction measured BOTH over all N and over the
   largest weakly-connected component -- then the matched-random graph generated at
   the grown net's ACTUAL (N,E). This exposes an (N,E) mismatch or a fragmentation
   artifact if either is driving the grown-vs-random SCC gap.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import networkx as nx
import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS_DevPriors.random_graph import generate_random_rnn


def stats(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    scc = [len(c) for c in nx.strongly_connected_components(G)]
    largest = max(scc) if scc else 0
    wccs = list(nx.weakly_connected_components(G))
    wcc_sizes = [len(c) for c in wccs]
    lwcc = max(wcc_sizes) if wcc_sizes else 0
    scc_in_wcc = 0
    if lwcc > 0:
        sub = G.subgraph(max(wccs, key=len))
        ss = [len(c) for c in nx.strongly_connected_components(sub)]
        scc_in_wcc = max(ss) if ss else 0
    Ab = nx.to_numpy_array(G, weight=None)
    srb = float(np.max(np.abs(np.linalg.eigvals(Ab)))) if Ab.size else 0.0
    return dict(n=n, e=e, largest_scc=largest, frac_overN=largest / n if n else 0.0,
                n_wcc=len(wcc_sizes), largest_wcc=lwcc,
                frac_in_wcc=scc_in_wcc / lwcc if lwcc else 0.0,
                n_isolated=len(list(nx.isolates(G))), spectral_bin=srb,
                has_cycle=largest > 1)


print("=== ANCHOR TESTS ===")
anchors = [
    ("ring5 (expect frac=1.00)", nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])),
    ("chain5 DAG (expect largest_scc=1, frac=0.20)", nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])),
    ("cycle3+tail (expect largest_scc=3, frac=0.60)", nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)])),
]
for name, g in anchors:
    s = stats(g)
    print(f"  {name}: largest_scc={s['largest_scc']} frac_overN={s['frac_overN']:.2f} "
          f"has_cycle={s['has_cycle']} spectral_bin={s['spectral_bin']:.2f}")

task = "acrobot"
nd = f"experiments/{task}/pool/networks"
files = sorted(f for f in os.listdir(nd) if f.endswith(".json"))
valid = []
for f in files:
    d = json.load(open(os.path.join(nd, f)))
    if d.get("valid"):
        valid.append(d)
comp = [d for d in valid if d.get("stratum") != "weak"][:3]
weak = sorted((d for d in valid if d.get("stratum") == "weak"),
              key=lambda d: d["network_stats"].get("neurons", 0))
sample = comp + [weak[0], weak[len(weak) // 2], weak[-1]]

print(f"\n=== REAL NETS ({task}): grown vs matched-random ===")
mismatch = 0
for d in sample:
    ns = d["network_stats"]
    sN, sE = int(ns["neurons"]), int(ns["connections"])
    genome = Genome.from_dict(d["genome"])
    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    sg = stats(grid.get_graph())
    tag = "" if (sg["n"], sg["e"]) == (sN, sE) else "  <<< (N,E) MISMATCH vs network_stats!"
    if tag:
        mismatch += 1
    rng = np.random.default_rng(999000 + int(d["network_id"]))
    Gr = generate_random_rnn(sg["n"], sg["e"], rng)
    sr = stats(Gr) if Gr is not None else None
    print(f"\n net {d['network_id']} stratum={d['stratum']}")
    print(f"   network_stats (N,E)=({sN},{sE}) | regrown actual (N,E)=({sg['n']},{sg['e']}){tag}")
    print(f"   GROWN : isolated={sg['n_isolated']} #WCC={sg['n_wcc']} largestWCC={sg['largest_wcc']} "
          f"({sg['largest_wcc']/sg['n']:.2f} of N) | largestSCC={sg['largest_scc']} "
          f"frac_overN={sg['frac_overN']:.3f} frac_in_largestWCC={sg['frac_in_wcc']:.3f} specBin={sg['spectral_bin']:.2f}")
    if sr:
        print(f"   RANDOM@({sg['n']},{sg['e']}): #WCC={sr['n_wcc']} largestSCC={sr['largest_scc']} "
              f"frac_overN={sr['frac_overN']:.3f} specBin={sr['spectral_bin']:.2f}")
    else:
        print(f"   RANDOM@({sg['n']},{sg['e']}): could not build a weakly-connected graph")

print(f"\n(N,E) mismatches in sample: {mismatch}/{len(sample)}")
