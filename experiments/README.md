# experiments/ – the data manifest

The reverse index for the study: which directory holds which result, produced or consumed by which script. This file maps every run directory to **what it is, its result file, and the analysis and figure script** that uses it. The findings and numbers are in the paper.

> **Headline metric:** the paper's headline odds ratios and intervals are from the canonical **paired-reward campaign** in [`reproduction/paired_reward_campaign/`](../reproduction/paired_reward_campaign/) – one developed graph and trajectory scored under both reward definitions, 20,000-replicate source-cluster bootstrap. The per-cell `mantel_haenszel.json` files below (`allpool_weak` / `full_allpool`) are the **earlier unpaired (N,E)-clean analysis**, retained for transparency and superseded by the paired campaign (their odds ratios differ slightly, e.g. Acrobot 16.1 there vs 17.6 in the paired headline). The prevalence ratio (`summary_stats.json`, the matched-to-competent R) is a further per-cell comparison, not the headline.

## Run-directory naming legend

Per-task layout is `<task>/{pool, matched_random/<cell>}`. The `<cell>` names:

| cell name | what it is |
|---|---|
| `pool/` | the full **5000-net genome pool**; `networks/` = grown phenotypes (regenerable from seeds), `pool_metadata.json` = reward summary + strata + competent fraction |
| `pool_pilot/` | a ~500-net pilot pool that gates a task and calibrates strata (kept only for `frozenlake`, whose frontier null has no full pool) |
| `matched_random/full/` | **competent-matched control, k=5** – random graphs (N,E)-matched to each non-weak MorphoNAS source (`summary_stats.json` = prevalence ratio R + MOVER/Wilson CI) |
| `matched_random/full_k50`, `_k500`, `_k20` | same control with more random graphs per source (`--num-random` bump) to firm or finite-ize a fragile ratio; a strict superset of the lower-k run |
| `matched_random/allpool_weak/`, `full_allpool/` | **(N,E)-clean weak-matched control** – random matched to all sources, binned by exact (neurons, edges), giving the **Mantel–Haenszel OR** (`mantel_haenszel.json`). `full_allpool` is the older name (acrobot, lunarlander_sparse); `allpool_weak` is the same thing for the rest |
| `matched_random/full_fixedseed`, `full_k50_fixedseed` | fixed generator seeds (not source-keyed) – a reproducibility-robustness variant |
| `matched_random/verify/` | a tiny correctness run |

**Result files in any cell:** `results.jsonl` (one row per random graph: rewards, neurons, edges, `source_id`, stratum), `summary_stats.json` (prevalence ratio R + CI), `mantel_haenszel.json` ((N,E)-clean MH OR + marginal-to-all ratio), and (LBA cells) `mh_cluster_bootstrap.json` (source-cluster bootstrap 95% CI for that OR).

**`pool/networks/` JSONs are regenerable** – grown phenotypes; the per-task seeds in `code/MorphoNAS_DevPriors/task_registry.py` regrow them bit-for-bit (`scripts/verify_pool_reproduce.py` proves it). They are committed here for a self-contained artifact and are the bulk of the repo by file count.

## The reach-map

| family | directories | what it is | analysis → figure |
|---|---|---|---|
| control ladder | `cartpole/`, `acrobot/`, `mountaincar/`, `pendulum/`, `lunarlander/` | 5-task pools + matched-random controls | `analyze_ratios.py`, `analyze_mantel_haenszel.py` → `plot_gradient_neclean.py` (F1) |
| reward de-confounder | `mountaincar_shaped/`, `acrobot_shaped/`, `pendulum_sparse/`, `lunarlander_sparse/` | dense/structure-free vs sparse/structure-gated reward, paired seeds, byte-identical trajectories | `analyze_deconfounder_neclean.py` → `plot_deconfounder_neclean.py` (F2 top) |
| observation / memory | `cartpole_masked/`, `acrobot_masked/` (with `cartpole/`, `acrobot/` as genome-paired unmasked companions) | velocity / angular-velocity masked POMDP rungs | `analyze_deconfounder_neclean.py`, `analyze_bar_sweep.py` → `plot_memory_rung.py`, `plot_deconfounder_neclean.py` (F2 bottom) |
| selectivity dose-response + recurrence | `reach_map/rwg/` | the RWG recurrence reference (fixed architecture, random weights) and the bar-sweep | `analyze_bar_sweep.py` → `plot_dose_response.py` (F4) |
| weight ablation | `acrobot/weight_ablation/` | the topology-vs-weights factorial (advantage survives uniform-random weights) | `analyze_weight_ablation.py` |
| navigation | `frozenlake/`, `frozenlake_shaped/` | the goal-reach frontier null and the progress inversion | `analyze_ratios.py` |
| vision | `vision_digits/prior_full/` | 8×8-digit feedforward classification prior (`summary.json`) | `run_vision_prior.py` |
| transparency | (re-analysis of the above) | naive vs (N,E)-clean, the size/density confound quantified | `analyze_bar_policy.py` → `plot_naive_vs_neclean.py` (F3) |

## Figures

**`figures_neclean/`** – the (neurons, edges)-clean figure catalog:
- `F1_gradient.png` – the family magnitude gradient (MH OR per rung, CIs) – `plot_gradient_neclean.py`
- `F2_deconfounder.png` – the de-confounder, reward + observation axes – `plot_deconfounder_neclean.py`
- `F3_naive_vs_neclean.png` – the size/density confound quantified (naive vs (N,E)-clean) – `plot_naive_vs_neclean.py`
- `F4_dose_response.png` – the selectivity dose-response (MH OR vs bar) – `plot_dose_response.py`
- `D1_tail.png` – the tail-phenomenon diagnostic (why a threshold metric) – `plot_tail_diagnostic.py`

## Reproducibility

`acrobot/matched_random/pool_reproduce_check.linux-x86_64.json` is the immutable acceptance baseline; `scripts/verify_pool_reproduce.py` regenerates `pool_reproduce_check.json` and it must match bit-for-bit on Linux x86_64. See the top-level README for the gate and the platform note.

## Additions beyond the reach-map

Beyond the reach-map above, this release adds:

- `<task>/recurrence_ablation/` – the feedback-arc-set recurrence ablation (`baseline`; `edge_removal` = DAG-ify; `random_edge_removal` = equal-capacity control; `state_reset`), with the paired McNemar summary in `comparison.json`. Scripts: `run_recurrence_ablation.py`, `analyze_recurrence_ablation.py`.
- `<task>/recurrence_structure/` – the structural recurrence measure (largest-SCC fraction, cycle rate, binary spectral radius; grown vs (neurons, edges)-matched random) in `summary_stats.json` + `per_source.jsonl`. Scripts: `analyze_recurrence_structure.py` (`--from-per-source` re-aggregates without regrowth), `sanity_check_recurrence_structure.py`, `reaggregate_recurrence_by_competence.py`.
- `_threshold_sweep/summary.json` – the competence-threshold robustness sweep: every published odds ratio is reproduced, and sign and ordering are stable across bars from the top 2% to 15% of grown networks. Script: `threshold_sweep.py`.
- `baselines/cppn_hyperneat/` – the structured indirect-encoding baselines (MorphoNAS vs a random CPPN/HyperNEAT encoder, (neurons, edges)-clean). Scripts: `run_baseline_prior.py`, `analyze_baselines.py`, `plot_baseline_*.py`.
