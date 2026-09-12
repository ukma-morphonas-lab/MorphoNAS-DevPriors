# MorphoNAS-DevPriors

Reproducibility companion for the study **"Grown, not trained: what a developmental map makes likely before selection"** (Medvid & Glybovets, National University of Kyiv-Mohyla Academy).

An *un-evolved* (random-genome) MorphoNAS network is a competent controller far more often than random directed wiring matched on the same neuron and edge counts. On Acrobot about **one developed network in ten** clears the floor-escape bar, versus about one in 140 matched-random graphs – a size-stratified odds ratio of **17.6 [14.1, 22.6]**. The advantage is **outcome-dependent**: rescoring the very same trajectories under a progress reward instead of a structured target flips its sign (MountainCar 6.9× → 0.27×, Pendulum 3.5× → 0.27×), while height-shaped Acrobot stays enriched (19.5×). This is a **functional prior** over behavior, measured before any learning or selection. Which topological feature carries it is not yet isolated: the null matches only size and density, leaving higher-order structure unmatched.

The headline numbers are the canonical paired-reward campaign in [`reproduction/`](reproduction/) – one developed graph and trajectory scored under both reward definitions, 20,000-replicate source-cluster bootstrap intervals. This repository also holds the broader **exploratory** evidence layer the forthcoming full paper develops: the cross-family map, velocity-masked (partially observed) variants, navigation and vision frontiers, recurrence and weight/topology probes, a threshold-robustness sweep, and CPPN/HyperNEAT baselines. Those axes are exploratory and their mechanisms are not yet isolated; the claim above rests only on the matched-random control.

This repository holds the runnable engine, the experiment data, and the analysis and figure scripts that produce every reported number. **The findings, magnitudes, confidence intervals, and figures live in the paper;** this repo is the evidence layer underneath it.

## What is here

| Path | What it holds |
| --- | --- |
| `code/MorphoNAS/` | vendored growth engine (canonical ruleset; do not edit – see provenance below) |
| `code/MorphoNAS_DevPriors/` | shared libraries (`task_registry`, `ratio_stats`, MH stats, graph samplers, wrappers) |
| `scripts/` | runners (`run_*.py`) + analysis (`analyze_*.py`) + figures (`plot_*.py`) + the repro gate |
| `experiments/` | pools, matched-random controls, de-confounder cells, figures – manifest: `experiments/README.md` |

The experiment manifest [`experiments/README.md`](experiments/README.md) maps every run directory to what it is, its result file, and the analysis and figure script that consume it. The script map is [`scripts/README.md`](scripts/README.md).

## Scope

This release is the full evidence layer of the study. The **headline** result – the matched-random control and the reward-axis de-confounder, both from the paired-reward campaign in [`reproduction/`](reproduction/) – is what the LBA claims. The remaining axes are **exploratory** (full-paper material): the cross-family map (control, partially observed, navigation, vision), the observation-axis de-confounder, the selectivity dose-response, the recurrence/topology axes (RWG reference, weight ablation), a **feedback-arc-set recurrence ablation** and a **structural recurrence measure** (descriptively, grown wiring has a smaller largest strongly-connected component than matched-random at fixed size; interpretation is deferred to the full paper), a **competence-threshold robustness sweep**, and the structured **CPPN / HyperNEAT indirect-encoding baselines**. The evolved-competitive comparison remains part of the forthcoming full paper and is not included here.

## Setup

Python 3.11–3.13. Dependencies are pinned in `pyproject.toml` / `uv.lock`.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .          # or: uv sync
```

Key dependencies: numpy (<2.0), torch, gymnasium, box2d-py, networkx, scipy, scikit-learn. Box2D (LunarLander) needs a C++ toolchain and SWIG at build time (`swig`, `gcc-c++`).

## Reproducibility gate

The six evidence cells reproduce **end-to-end** in a pinned CPU environment: build `.venv-reproduction`, verify the numerical canary (`reproduction/canary.expected.json`), then run the canonical paired-reward campaign, which re-grows every committed genome and scores byte-identical trajectories under both outcome definitions. Full workflow and locked wheels are in [`reproduction/README.md`](reproduction/README.md); the deterministic result is `reproduction/paired_reward_campaign/paired_reward_campaign_results.json`.

A lighter aggregate-regression check re-grows the Acrobot genome pool on the vendored engine and re-evaluates a sample, then checks the recomputed competence rates against the recorded result:

```bash
.venv/bin/python3 scripts/verify_pool_reproduce.py --sample-weak 50 --seed 12345
```

On Linux x86_64 this reproduces `experiments/acrobot/matched_random/pool_reproduce_check.linux-x86_64.json` **bit-for-bit**. The script writes a fresh `pool_reproduce_check.json` next to the reference; diff the two to confirm (an empty diff is the pass). Check A (the strata counts and the 10.26% non-weak / 1.74% solved rates over all 5000 stored networks) is exact and seed-independent. Check B re-grows all 513 competent sources plus a weak sample and records how many keep their stratum (488/513 non-weak, 82/87 solved); the engine's regrowth differs only at the ULP level (largest single-source change about 404 in reward, mean about 17), and that variability is itself fixed in the reference, so on x86_64 the diff is empty. Runtime is roughly 2–4 min on 16 cores. This pool-growth check scores the stored phenotypes directly (513/5000 competent, 10.26%); the headline paired-reward campaign re-scores each network's byte-identical trajectory under both reward definitions and reports 517/5000 (10.34%), the small difference being the expected ULP-level effect of the two scoring paths.

A faster wiring smoke test (about 3–4 min, tiny slices) asserts that the generalized control reproduces the locked Acrobot driver byte-for-byte and that shard-plus-merge equals the unsharded union:

```bash
WORKERS=2 bash scripts/smoke_ladder.sh
```

> **Platform note.** The committed pool phenotypes and the reproduce reference are the Linux x86_64 record. On a different architecture (e.g. arm64) the engine is logic-identical but floating-point results differ at the ULP level, so rates reproduce in distribution but not bit-for-bit. This is by design; compare against the `.linux-x86_64.json` reference on x86_64.

## Reproduce a result

1. **Find the claim** in the paper – every reported magnitude names its task family and metric.
2. **Find the data**: the headline odds ratios and intervals are in `reproduction/paired_reward_campaign/paired_reward_campaign_results.json` (the paired campaign that generates the paper's Table 1). Per-task matched-random detail is under `experiments/<task>/…`: `mantel_haenszel.json` for the (neurons, edges)-clean odds ratio, `summary_stats.json` for the prevalence ratio and its CI, `results.jsonl` for per-graph rows.
3. **Re-run the analysis** with the matching `scripts/analyze_*.py`, or regenerate a figure with `scripts/plot_*.py`; the manifest pairs each run with its scripts. The headline figures (F1 gradient, F2 de-confounder) read the committed `mantel_haenszel.json` aggregates and reproduce with no regrowth; the diagnostic figures (tail, dose-response, memory rung) recompute from the per-network phenotypes in `experiments/<task>/pool/networks/`.
4. **Regenerate the pool** with `scripts/run_pool.py --task <t>`; pools regrow bit-identically from the per-task seeds in `code/MorphoNAS_DevPriors/task_registry.py`, as the reproducibility gate proves.

## Data

The matched-random measurements (`results.jsonl`, `summary_stats.json`, `mantel_haenszel.json`) and the grown MorphoNAS pool phenotypes (`experiments/<task>/pool/networks/*.json`) are committed so that the gate and every figure run on a fresh clone with nothing to regenerate. The phenotypes are a deterministic function of the committed genome seeds (the gate verifies they regrow bit-for-bit), so a leaner mirror can drop them and regenerate on demand; this repository keeps them for a self-contained, audit-ready artifact.

## Engine provenance

`code/MorphoNAS/` is the canonical MorphoNAS developmental ruleset ([github.com/sergemedvid/MorphoNAS](https://github.com/sergemedvid/MorphoNAS), arXiv:2507.13785), copied verbatim and made importable, plus two approved fixes from the MorphoNAS-PL line carried as explicit base commits:

- a **per-episode network reset** in the gym evaluation (`fitness_functions.GymFitnessFunction`), which the seed-paper evaluation did not perform; and
- the **`W[post, pre]` propagation-direction orientation** in `neural_propagation._get_weight_matrix` (forward signal flow).

Evaluation uses **no plasticity** (propagators built with no edge hook). Growth (`grid.py`, `genome.py`) is logic-identical to the canonical ruleset. Please do not edit the vendored engine; if the engine or evaluation ever changes, re-run the reproducibility gate and record the result.

## Citation

If you use this code or data, please cite the paper (see [`CITATION.cff`](CITATION.cff)) and the MorphoNAS engine (arXiv:2507.13785).

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)); the vendored engine under `code/MorphoNAS/` is covered by the same licence, its authors being the copyright holders named there. The experiment data under `experiments/` and the committed campaign rows under `reproduction/` are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([`LICENSE-DATA`](LICENSE-DATA)).
