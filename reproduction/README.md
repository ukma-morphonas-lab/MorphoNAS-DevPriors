# Canonical reproduction

This directory defines a narrower and stricter environment than the general
development lock. It reproduces the six evidence-bearing cells of the
developmental-priors study from committed genome inputs and code:

- MountainCar target / progress;
- Pendulum upright / dense;
- Acrobot native / height-shaped.

Within each pair, one graph and one policy execution produce both scores. The
runner advances the environments in lockstep and fails on any observation,
termination, or trajectory difference. Inference resamples source clusters, so
each grown source stays attached to its five requested matched-random controls.

## Reproducibility boundary

The exact scientific inputs are the committed per-network JSON files listed in
`manifests/*.jsonl`. Every manifest records the anchor-file, genome, and rollout-
seed hashes. Seeds are retained as provenance, but are not treated as the sole
input: a small subset of historical Acrobot genomes differs from a fresh
`Genome.random(seed)` result by one floating-point ULP across NumPy lineages.

The canonical runtime is:

- Linux x86-64;
- CPython 3.12.3;
- the exact, hashed binary wheels in
  `requirements-linux-x86_64-py312.txt`;
- CPU-only, single-threaded PyTorch with deterministic algorithms and MKL-DNN
  disabled;
- fixed hash, BLAS, and thread environment variables from
  `environment.json`;
- an exact three-pair graph/trajectory canary before any campaign.

The older `scripts/verify_pool_reproduce.py` remains a useful historical
aggregate-regression check, but it is not a byte-exact graph gate: its committed
Acrobot fixture reports 60 changed `(N,E)` shapes among 563 re-grown cases and
36 stratum changes. The current campaign therefore grows every committed genome
once in the frozen environment and matches controls to that freshly grown
shape; historical shapes are retained only as an audit field.

## 1. Build the environment

Environment creation can take several minutes, so on the experiment host run it
inside tmux:

```bash
tmux new-session -d -s devpriors-reproduction-env \
  'cd /path/to/MorphoNAS-DevPriors && bash scripts/create_reproduction_environment.sh 2>&1 | tee /tmp/devpriors-reproduction-env.log'
tmux attach -t devpriors-reproduction-env
```

The creation script refuses to overwrite `.venv-reproduction` and installs with
`pip --require-hashes`. Verify later with:

```bash
env PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  .venv-reproduction/bin/python scripts/check_reproduction_environment.py
```

## 2. Verify the numerical canary

Normal use is fail-closed:

```bash
env PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 MKL_CBWR=COMPATIBLE \
  .venv-reproduction/bin/python scripts/verify_reproduction_canary.py
```

Maintainers use `--record` exactly once when establishing a new locked
environment. It refuses to overwrite an existing expected fixture. The fixture
must then be reviewed and committed before the full campaign. The Acrobot
canary deliberately uses `network_00056`, the lineage sentinel that changed
shape under the previously drifted server environment.

## 3. Run the campaign in tmux

Use an output path outside the Git worktree; canonical execution requires the
worktree to be clean.

```bash
.venv-reproduction/bin/python scripts/start_paired_reward_campaign_tmux.py \
  --session devpriors-paired-reward-2026 \
  --output-root /absolute/path/devpriors-paired-reward-2026
```

The default campaign is 5,000 sources per pair, five matched-random requests per
source, 20 rollout seeds, 32 workers, and 20,000 source-cluster bootstrap
replicates. It is resumable without overwriting completed rows:

```bash
.venv-reproduction/bin/python scripts/start_paired_reward_campaign_tmux.py \
  --session devpriors-paired-reward-2026-resume \
  --output-root /absolute/path/devpriors-paired-reward-2026 --resume
```

The final deterministic evidence file is
`paired_reward_campaign_results.json`. Each pair directory also contains canonical
`sources.jsonl`, `controls.jsonl`, `paired_reward_results.json`, runtime metadata, and
private/public execution fingerprints. Native Acrobot's pre-fixed 2%, 3%, 5%,
8%, and 12% grown-tail sensitivity grid is derived from the same raw rewards.

## Manifest verification

The committed manifests can be regenerated and compared without writing:

```bash
.venv/bin/python scripts/build_anchor_manifests.py --check
```

Any mismatch is input drift and must be investigated, not accepted silently.
