# scripts/ – runners, analysis, figures, checks

Every script is a thin driver over the shared libraries in `code/MorphoNAS_DevPriors/` and the vendored engine in `code/MorphoNAS/`. The pattern is **`run_*` (generate data) → `analyze_*` (compute the ratio / odds ratio) → `plot_*` (figure)**. Run with the project venv: `.venv/bin/python3 scripts/<name>.py [...]`.

## Runners

| script | role |
|---|---|
| `run_pool.py` | grow a genome pool (`--task <t> --target-valid {500,5000}`); phenotypes regrow bit-for-bit from the per-task seeds |
| `run_matched_random_control.py` | the matched-random control (`--num-random {5,50,500}`), random graphs (N,E)-matched to each MorphoNAS source |
| `run_acrobot_matched_random.py` | the original Acrobot-specific control driver (kept for provenance; the generalized runner reproduces it byte-for-byte) |
| `run_weight_ablation.py` | the topology-vs-weights factorial (`--topology --weight-mode`) |
| `run_rwg_axis.py` | the RWG recurrence reference (fixed architecture, random weights) |
| `run_vision_prior.py` | the 8×8-digit vision rung (MorphoNAS + matched-random) |

## Analysis

| script | role |
|---|---|
| `analyze_ratios.py` | competence ratio R + MOVER/Wilson CIs |
| `analyze_mantel_haenszel.py` | the (neurons, edges)-adjusted Mantel–Haenszel odds ratio |
| `analyze_source_cluster_bootstrap.py` | source-cluster bootstrap 95% CI for the MH odds ratio (resamples grown sources with their nested nulls; leaves the RBG intervals nearly unchanged) |
| `analyze_deconfounder_neclean.py` | the within-task de-confounder (reward and observation axes), (N,E)-clean |
| `analyze_bar_sweep.py` | the selectivity dose-response across nested competence bars |
| `analyze_bar_policy.py` | behavioral-bar competence + the naive-vs-(N,E)-clean transparency comparison |
| `analyze_weight_ablation.py` | weight-ablation recovery (topology vs weights) |

## Figures

| script | output |
|---|---|
| `plot_gradient_neclean.py` | F1 – the family magnitude gradient (MH OR per rung) |
| `plot_deconfounder_neclean.py` | F2 – the de-confounder, reward + observation axes |
| `plot_naive_vs_neclean.py` | F3 – the size/density confound quantified |
| `plot_dose_response.py` | F4 – the selectivity dose-response |
| `plot_tail_diagnostic.py` | D1 – the tail-phenomenon diagnostic |
| `plot_memory_rung.py` | the masked-vs-unmasked memory-axis figure |

## Checks and utilities

| script | role |
|---|---|
| `verify_pool_reproduce.py` | **the acceptance gate** – regrow + re-evaluate the Acrobot pool, check bit-for-bit against the recorded x86_64 baseline |
| `smoke_ladder.sh` | fast wiring smoke test: the generalized control reproduces the locked Acrobot driver, and shard+merge equals the unsharded union |
| `smoke_pendulum.py` | a minimal single-task smoke check (engine + evaluation wiring) |
| `merge_shards.py` | merge sharded control or pool runs into one result (for runs split across parallel workers) |

## Additions beyond the reach-map

| script | role |
|---|---|
| `run_recurrence_ablation.py` / `analyze_recurrence_ablation.py` | feedback-arc-set recurrence ablation vs an equal-capacity random-edge control |
| `analyze_recurrence_structure.py` / `sanity_check_recurrence_structure.py` | structural recurrence measure (SCC, cycles, spectral radius); `--from-per-source` re-aggregates |
| `reaggregate_recurrence_by_competence.py` | the competent-tail cut of the structural measure |
| `threshold_sweep.py` | competence-threshold robustness sweep (reproduces every published odds ratio) |
| `run_baseline_prior.py` / `analyze_baselines.py` / `plot_baseline_*.py` | structured CPPN/HyperNEAT indirect-encoding baselines |
| `run_paired_reward.py` / `analyze_paired_reward.py` | the reward-axis de-confounder (paired re-scoring of byte-identical trajectories) |
