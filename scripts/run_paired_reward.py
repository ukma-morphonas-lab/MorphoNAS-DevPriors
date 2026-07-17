#!/usr/bin/env python3
"""Canonical paired reward-definition experiment for the ALIFE LBA.

Each stored anchor genome is grown once. Each matched-random graph is generated
once. A pair of environments is then advanced in lockstep with the same action;
the runner asserts exact observation, termination, and trajectory equality at
every step while collecting the two reward definitions. This makes pairing an
executed invariant rather than an assumption inferred from separate campaigns.

Historical results are never modified. The runner writes a new directory with
``sources.jsonl``, ``controls.jsonl``, progress, metadata, and private/public
execution-environment records. Runs are resumable by source and control key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import struct
import subprocess
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from MorphoNAS_DevPriors.reproducibility import (  # noqa: E402
    canonical_graph_sha256,
    canonical_json_sha256,
    canonicalize_jsonl,
    configure_torch_cpu,
    environment_fingerprint,
    set_early_runtime_environment,
    sha256_file,
    verify_locked_environment,
)

# This must precede NumPy, PyTorch, Gymnasium, and engine imports.
set_early_runtime_environment()

import numpy as np  # noqa: E402

from MorphoNAS.genome import Genome  # noqa: E402
from MorphoNAS.grid import Grid  # noqa: E402
from MorphoNAS.weight_channel import WeightChannel, set_weight_channel  # noqa: E402
from MorphoNAS_DevPriors.logging_config import setup_logging  # noqa: E402
from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads  # noqa: E402
from MorphoNAS_DevPriors.progress import ProgressWriter  # noqa: E402
from MorphoNAS_DevPriors.random_graph import generate_random_rnn  # noqa: E402
from MorphoNAS_DevPriors.task_registry import get_task  # noqa: E402


PAIR_TASKS = {
    "mountaincar": ("mountaincar", "mountaincar_shaped"),
    "pendulum": ("pendulum_sparse", "pendulum"),
    "acrobot": ("acrobot", "acrobot_shaped"),
}

RUN_SCHEMA_VERSION = 2

# New seed bands, deliberately disjoint from the historical separately-run
# controls. The pair source index, not a historical source ID, keys each graph.
PAIR_CONTROL_BASE = {
    "mountaincar": 92_000_000,
    "pendulum": 102_000_000,
    "acrobot": 60_000_000,
}

_W: dict = {}
logger = logging.getLogger(__name__)


def _safe_bounds(spec) -> list[dict]:
    def safe(value):
        if value == float("inf"):
            return "inf"
        if value == -float("inf"):
            return "-inf"
        return float(value)

    return [
        {"name": name, "low": safe(low), "high": safe(high)}
        for name, low, high in spec.stratum_bounds
    ]


def _sha256(path: str | Path) -> str:
    return sha256_file(path)


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            capture_output=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _execution_environment(scientific: dict, include_private: bool) -> dict:
    git_status = _git_value("status", "--porcelain")
    fingerprint = {
        **scientific,
        "cpu_count": os.cpu_count(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_clean": git_status == "",
        "runner_sha256": _sha256(__file__),
    }
    if include_private:
        fingerprint["hostname"] = socket.gethostname()
        fingerprint["git_status_porcelain"] = git_status
        fingerprint["python_executable"] = str(Path(sys.executable).resolve())
    return fingerprint


def _write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _load_jsonl(path: Path, key_fields: tuple[str, ...]) -> dict[tuple, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = tuple(row[field] for field in key_fields)
            if key in rows:
                raise RuntimeError(f"duplicate key {key} in {path}:{line_number}")
            rows[key] = row
    return rows


def _assert_same_observation(left, right, context: str) -> np.ndarray:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
        max_delta = None
        if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            max_delta = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
        raise RuntimeError(
            f"trajectory invariant failed at {context}: "
            f"shape {a.shape}/{b.shape}, dtype {a.dtype}/{b.dtype}, max_delta={max_delta}"
        )
    return a


def _hash_array(digest, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(struct.pack("<I", contiguous.ndim))
    for size in contiguous.shape:
        digest.update(struct.pack("<Q", int(size)))
    digest.update(contiguous.tobytes())


def dual_rollouts(graph, spec_a, spec_b, env_a, env_b, eval_seeds: list[int]) -> dict:
    """Score one policy under two reward definitions with lockstep assertions."""
    if (spec_a.input_dim, spec_a.output_dim, spec_a.env_name) != (
        spec_b.input_dim,
        spec_b.output_dim,
        spec_b.env_name,
    ):
        raise ValueError(f"not a reward-only pair: {spec_a.name}, {spec_b.name}")

    propagator = spec_a.make_propagator(graph)
    rewards_a: list[float] = []
    rewards_b: list[float] = []
    lengths: list[int] = []
    digest = hashlib.sha256()

    for episode_index, seed in enumerate(eval_seeds):
        propagator.reset()
        obs_a, _ = env_a.reset(seed=int(seed))
        obs_b, _ = env_b.reset(seed=int(seed))
        obs = _assert_same_observation(obs_a, obs_b, f"reset episode={episode_index}")
        digest.update(struct.pack("<q", int(seed)))
        _hash_array(digest, obs)

        total_a = 0.0
        total_b = 0.0
        steps = 0
        done = False
        while not done:
            propagator.propagate(np.asarray(obs).flatten())
            action = int(propagator.get_output().argmax().item())
            digest.update(struct.pack("<q", action))

            next_a, reward_a, terminated_a, truncated_a, _ = env_a.step(action)
            next_b, reward_b, terminated_b, truncated_b, _ = env_b.step(action)
            if (bool(terminated_a), bool(truncated_a)) != (
                bool(terminated_b),
                bool(truncated_b),
            ):
                raise RuntimeError(
                    f"termination invariant failed episode={episode_index} step={steps}: "
                    f"{terminated_a}/{truncated_a} vs {terminated_b}/{truncated_b}"
                )
            obs = _assert_same_observation(
                next_a, next_b, f"episode={episode_index} step={steps}"
            )
            _hash_array(digest, obs)
            total_a += float(reward_a)
            total_b += float(reward_b)
            steps += 1
            done = bool(terminated_a or truncated_a)

        rewards_a.append(total_a)
        rewards_b.append(total_b)
        lengths.append(steps)

    def summary(spec, rewards):
        mean = float(np.mean(rewards)) if rewards else 0.0
        stratum = spec.get_stratum(mean)
        return {
            "mean_reward": mean,
            "rewards": [float(value) for value in rewards],
            "stratum": stratum,
            "competent": stratum != "weak",
        }

    return {
        "scores": {
            spec_a.name: summary(spec_a, rewards_a),
            spec_b.name: summary(spec_b, rewards_b),
        },
        "lengths": lengths,
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "trajectory_sha256": digest.hexdigest(),
        "trajectory_equality_verified": True,
    }


def _init_worker(pair: str, rollouts: int, weight_lo: float, weight_hi: float, max_retries: int) -> None:
    configure_worker_threads()
    configure_torch_cpu()
    set_weight_channel(WeightChannel())
    task_a, task_b = PAIR_TASKS[pair]
    spec_a, spec_b = get_task(task_a), get_task(task_b)
    _W.update(
        {
            "pair": pair,
            "spec_a": spec_a,
            "spec_b": spec_b,
            "env_a": spec_a.make_env(),
            "env_b": spec_b.make_env(),
            "rollouts": int(rollouts),
            "weight_range": (float(weight_lo), float(weight_hi)),
            "max_retries": int(max_retries),
        }
    )


def _source_eval(task: dict) -> dict:
    actual_anchor_sha256 = _sha256(task["anchor_path"])
    if actual_anchor_sha256 != task["anchor_sha256"]:
        raise RuntimeError(
            f"anchor byte hash failed for {task['anchor_path']}: "
            f"{actual_anchor_sha256} != {task['anchor_sha256']}"
        )
    with open(task["anchor_path"]) as handle:
        stored = json.load(handle)
    if int(stored["seed"]) != int(task["genome_seed"]):
        raise RuntimeError(f"anchor seed failed for {task['anchor_path']}")
    if int(stored["network_id"]) != int(task["anchor_network_id"]):
        raise RuntimeError(f"anchor network ID failed for {task['anchor_path']}")
    genome_sha256 = canonical_json_sha256(stored["genome"])
    if genome_sha256 != task["genome_sha256"]:
        raise RuntimeError(
            f"anchor genome hash failed for {task['anchor_path']}: "
            f"{genome_sha256} != {task['genome_sha256']}"
        )
    eval_seeds = list(stored.get("rollout_data", {}).get("eval_seeds", []))
    eval_seeds_sha256 = canonical_json_sha256([int(seed) for seed in eval_seeds])
    if eval_seeds_sha256 != task["eval_seeds_sha256"]:
        raise RuntimeError(f"anchor rollout-seed hash failed for {task['anchor_path']}")
    genome = Genome.from_dict(stored["genome"])
    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    graph = grid.get_graph()
    n = int(graph.number_of_nodes())
    e = int(graph.number_of_edges())
    recorded_shape = stored.get("network_stats", {})
    regrown_shape = {"neurons": n, "connections": e}
    topology_shape_match = recorded_shape == regrown_shape
    spec_a, spec_b = _W["spec_a"], _W["spec_b"]
    base = {
        "kind": "grown",
        "pair": _W["pair"],
        "pair_source_id": int(task["pair_source_id"]),
        "genome_seed": int(stored["seed"]),
        "anchor_network_id": int(stored["network_id"]),
        "anchor_path": task["anchor_relpath"],
        "anchor_sha256": actual_anchor_sha256,
        "stored_network_stats": recorded_shape,
        "network_stats": regrown_shape,
        "topology_shape_match": topology_shape_match,
        "genome_sha256": genome_sha256,
        "graph_sha256": canonical_graph_sha256(graph),
    }
    if n < spec_a.min_neurons:
        return {**base, "valid": False, "error": "insufficient_neurons"}
    if len(eval_seeds) < _W["rollouts"]:
        return {
            **base,
            "valid": False,
            "error": f"only {len(eval_seeds)} stored eval seeds",
        }
    result = dual_rollouts(
        graph,
        spec_a,
        spec_b,
        _W["env_a"],
        _W["env_b"],
        eval_seeds[: _W["rollouts"]],
    )
    return {**base, "valid": True, **result}


def _control_eval(task: dict) -> dict:
    n = int(task["num_neurons"])
    e = int(task["num_connections"])
    random_seed = int(task["random_seed"])
    base = {
        "kind": "random",
        "pair": _W["pair"],
        "pair_source_id": int(task["pair_source_id"]),
        "k": int(task["k"]),
        "random_seed": random_seed,
        "network_stats": {"neurons": n, "connections": e},
    }
    rng = np.random.default_rng(random_seed)
    graph = generate_random_rnn(
        n,
        e,
        rng,
        weight_range=_W["weight_range"],
        max_retries=_W["max_retries"],
    )
    if graph is None:
        return {**base, "valid": False, "error": "not_weakly_connected"}
    base["graph_sha256"] = canonical_graph_sha256(graph)
    result = dual_rollouts(
        graph,
        _W["spec_a"],
        _W["spec_b"],
        _W["env_a"],
        _W["env_b"],
        list(task["eval_seeds"]),
    )
    return {**base, "valid": True, **result}


def _manifest_sources(
    manifest_path: Path,
    pair: str,
    max_sources: int,
    rollouts: int,
    selected_source_ids: set[int] | None,
) -> list[dict]:
    records: list[dict] = []
    with manifest_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("pair") != pair:
                raise RuntimeError(
                    f"{manifest_path}:{line_number} belongs to {row.get('pair')}, not {pair}"
                )
            expected_source_id = len(records) + 1
            if int(row["pair_source_id"]) != expected_source_id:
                raise RuntimeError(
                    f"{manifest_path}:{line_number} source ID {row['pair_source_id']} "
                    f"!= {expected_source_id}"
                )
            if int(row["rollout_seed_count"]) < rollouts:
                raise RuntimeError(
                    f"{manifest_path}:{line_number} contains only "
                    f"{row['rollout_seed_count']} rollout seeds"
                )
            anchor_path = REPO_ROOT / row["anchor_path"]
            if not anchor_path.is_file():
                raise RuntimeError(f"anchor does not exist: {anchor_path}")
            records.append(
                {
                    **row,
                    "anchor_path": str(anchor_path),
                    "anchor_relpath": row["anchor_path"],
                }
            )
    if selected_source_ids is not None:
        records = [
            record
            for record in records
            if int(record["pair_source_id"]) in selected_source_ids
        ]
        found = {int(record["pair_source_id"]) for record in records}
        if found != selected_source_ids:
            raise RuntimeError(
                f"source IDs absent from {manifest_path}: "
                f"{sorted(selected_source_ids - found)}"
            )
    if max_sources > 0:
        records = records[:max_sources]
    return records


def _run_phase(
    phase: str,
    worker,
    tasks: list[dict],
    output_path: Path,
    progress_path: Path,
    pool: Pool,
    progress_every: int,
) -> tuple[int, int]:
    progress = ProgressWriter(
        str(progress_path), total=len(tasks), phase=phase, meta={"output": str(output_path)}
    )
    valid = 0
    completed = 0
    with output_path.open("a") as output:
        for result in pool.imap_unordered(worker, tasks, chunksize=1):
            output.write(json.dumps(result, sort_keys=True) + "\n")
            completed += 1
            valid += int(bool(result.get("valid")))
            if completed % progress_every == 0 or completed == len(tasks):
                output.flush()
                os.fsync(output.fileno())
                progress.update(completed, valid=valid, invalid=completed - valid)
    progress.done(valid=valid, invalid=completed - valid)
    return completed, valid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, choices=sorted(PAIR_TASKS))
    parser.add_argument(
        "--anchor-manifest",
        default=None,
        help="default: reproduction/manifests/<pair>.jsonl",
    )
    parser.add_argument(
        "--environment-lock",
        default=str(REPO_ROOT / "reproduction/environment.json"),
    )
    parser.add_argument(
        "--allow-unlocked-environment",
        action="store_true",
        help="development/smoke tests only; marks output non-canonical",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-sources", type=int, default=5000)
    parser.add_argument(
        "--source-ids",
        default=None,
        help="comma-separated manifest source IDs (canary/diagnostics only)",
    )
    parser.add_argument("--num-random", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    parser.add_argument("--weight-lo", type=float, default=0.01)
    parser.add_argument("--weight-hi", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_sources < 0:
        raise SystemExit("--max-sources must be >= 0 (0 means no cap)")
    if not 1 <= args.num_random <= 100:
        raise SystemExit("--num-random must be in [1, 100] (the fixed seed stride is 100)")
    if args.rollouts <= 0:
        raise SystemExit("--rollouts must be > 0")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be > 0")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be > 0")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be > 0")
    if not args.weight_lo < args.weight_hi:
        raise SystemExit("--weight-lo must be less than --weight-hi")
    selected_source_ids = None
    if args.source_ids:
        try:
            selected_source_ids = {
                int(value) for value in args.source_ids.split(",") if value.strip()
            }
        except ValueError as error:
            raise SystemExit("--source-ids must be comma-separated integers") from error
        if not selected_source_ids or min(selected_source_ids) <= 0:
            raise SystemExit("--source-ids must contain positive integers")

    configure_worker_threads()
    if args.allow_unlocked_environment:
        configure_torch_cpu()
        scientific_environment = environment_fingerprint()
        scientific_environment.update(
            {
                "verified": False,
                "canonical": False,
                "warning": "unlocked development run; not publishable evidence",
            }
        )
    else:
        scientific_environment = verify_locked_environment(args.environment_lock)
        scientific_environment["canonical"] = True
        git_status = _git_value("status", "--porcelain")
        if git_status is None:
            raise SystemExit("canonical execution requires a Git worktree")
        if git_status:
            raise SystemExit(
                "canonical execution requires a clean Git worktree; current changes:\n"
                + git_status
            )
    task_a, task_b = PAIR_TASKS[args.pair]
    spec_a, spec_b = get_task(task_a), get_task(task_b)
    if (spec_a.input_dim, spec_a.output_dim, spec_a.env_name) != (
        spec_b.input_dim,
        spec_b.output_dim,
        spec_b.env_name,
    ):
        raise SystemExit(f"{task_a}/{task_b} is not a reward-only pair")

    anchor_manifest = Path(
        args.anchor_manifest
        or REPO_ROOT / "reproduction" / "manifests" / f"{args.pair}.jsonl"
    )
    if not anchor_manifest.is_absolute():
        anchor_manifest = (Path.cwd() / anchor_manifest).resolve()
    if not anchor_manifest.is_file():
        raise SystemExit(f"anchor manifest not found: {anchor_manifest}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources_path = out_dir / "sources.jsonl"
    controls_path = out_dir / "controls.jsonl"
    metadata_path = out_dir / "metadata.json"
    if not args.resume and any(out_dir.iterdir()):
        raise SystemExit(
            f"{out_dir} is not empty; pass --resume or choose a new directory"
        )
    if args.resume and not metadata_path.exists():
        raise SystemExit(f"cannot resume without {metadata_path}")

    source_tasks = _manifest_sources(
        anchor_manifest,
        args.pair,
        args.max_sources,
        args.rollouts,
        selected_source_ids,
    )
    if not source_tasks:
        raise SystemExit(f"no anchor networks found in {anchor_manifest}")
    existing_sources = _load_jsonl(sources_path, ("pair_source_id",))
    pending_sources = [
        task for task in source_tasks if (task["pair_source_id"],) not in existing_sources
    ]

    private_environment = _execution_environment(
        scientific_environment, include_private=True
    )
    public_environment = _execution_environment(
        scientific_environment, include_private=False
    )
    try:
        manifest_label = str(anchor_manifest.relative_to(REPO_ROOT))
    except ValueError:
        manifest_label = str(anchor_manifest)
    immutable_config = {
        "schema_version": RUN_SCHEMA_VERSION,
        "pair": args.pair,
        "tasks": [task_a, task_b],
        "anchor_manifest": manifest_label,
        "anchor_manifest_sha256": _sha256(anchor_manifest),
        "source_selection": "manifest order; exact committed genome JSON is the scientific input",
        "selected_source_ids": (
            sorted(selected_source_ids) if selected_source_ids is not None else None
        ),
        "requested_sources": len(source_tasks),
        "num_random_per_source": args.num_random,
        "rollouts": args.rollouts,
        "control_base_seed": PAIR_CONTROL_BASE[args.pair],
        "control_seed_formula": "base + pair_source_id * 100 + k",
        "weight_range": [args.weight_lo, args.weight_hi],
        "max_retries": args.max_retries,
        "engine_weight_channel": "saturating (v1.1 default)",
        "runner_sha256": _sha256(__file__),
        "code_git_commit": private_environment["git_commit"],
        "canonical_environment": not args.allow_unlocked_environment,
        "environment_lock_sha256": scientific_environment.get("lock_sha256"),
        "requirements_sha256": scientific_environment.get("requirements_sha256"),
    }
    previous_metadata = None
    if args.resume:
        previous_metadata = json.loads(metadata_path.read_text())
        if previous_metadata.get("status") == "complete":
            raise SystemExit(f"{out_dir} is already complete")
        if previous_metadata.get("immutable_config") != immutable_config:
            raise SystemExit(
                "resume configuration differs from metadata.json; choose a new output directory"
            )

    setup_logging(str(out_dir), "paired_reward.log")
    _write_json(out_dir / "execution_environment.private.json", private_environment)
    _write_json(out_dir / "execution_environment.public.json", public_environment)
    metadata = {
        "schema_version": RUN_SCHEMA_VERSION,
        "experiment": "paired_reward",
        "status": "running",
        "pair": args.pair,
        "tasks": [task_a, task_b],
        "anchor_manifest": manifest_label,
        "anchor_manifest_sha256": _sha256(anchor_manifest),
        "source_selection": "manifest order; exact committed genome JSON is the scientific input",
        "selected_source_ids": (
            sorted(selected_source_ids) if selected_source_ids is not None else None
        ),
        "canonical_environment": not args.allow_unlocked_environment,
        "requested_sources": len(source_tasks),
        "num_random_per_source": args.num_random,
        "rollouts": args.rollouts,
        "control_base_seed": PAIR_CONTROL_BASE[args.pair],
        "weight_range": [args.weight_lo, args.weight_hi],
        "max_retries": args.max_retries,
        "max_workers": args.max_workers,
        "engine_weight_channel": "saturating (v1.1 default)",
        "immutable_config": immutable_config,
        "task_specs": {
            task_a: {"bounds": _safe_bounds(spec_a), "note": spec_a.solved_note},
            task_b: {"bounds": _safe_bounds(spec_b), "note": spec_b.solved_note},
        },
        "pairing_contract": {
            "graph_constructed_once": True,
            "policy_propagated_once_per_step": True,
            "environment_observations_compared_exactly_each_step": True,
            "termination_compared_exactly_each_step": True,
            "reward_not_network_input": True,
        },
        "started_at": (
            previous_metadata["started_at"]
            if previous_metadata is not None
            else time.strftime("%Y-%m-%d %H:%M:%S %z")
        ),
        "runner_sha256": _sha256(__file__),
    }
    if previous_metadata is not None:
        metadata["resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    _write_json(metadata_path, metadata)

    logger.info(
        "paired reward run: pair=%s tasks=%s/%s sources=%d pending=%d workers=%d",
        args.pair,
        task_a,
        task_b,
        len(source_tasks),
        len(pending_sources),
        args.max_workers,
    )
    start = time.time()
    with Pool(
        args.max_workers,
        initializer=_init_worker,
        initargs=(args.pair, args.rollouts, args.weight_lo, args.weight_hi, args.max_retries),
    ) as pool:
        if pending_sources:
            _run_phase(
                "grown_sources",
                _source_eval,
                pending_sources,
                sources_path,
                out_dir / "sources.progress.json",
                pool,
                args.progress_every,
            )

        canonicalize_jsonl(sources_path, ("pair_source_id",))
        source_rows = _load_jsonl(sources_path, ("pair_source_id",))
        valid_sources = {
            key[0]: row for key, row in source_rows.items() if row.get("valid")
        }
        existing_controls = _load_jsonl(controls_path, ("pair_source_id", "k"))
        control_tasks = []
        base_seed = PAIR_CONTROL_BASE[args.pair]
        for source_id, row in sorted(valid_sources.items()):
            for k in range(args.num_random):
                if (source_id, k) in existing_controls:
                    continue
                control_tasks.append(
                    {
                        "pair_source_id": source_id,
                        "k": k,
                        "random_seed": base_seed + source_id * 100 + k,
                        "num_neurons": row["network_stats"]["neurons"],
                        "num_connections": row["network_stats"]["connections"],
                        "eval_seeds": row["eval_seeds"],
                    }
                )
        if control_tasks:
            _run_phase(
                "matched_random_controls",
                _control_eval,
                control_tasks,
                controls_path,
                out_dir / "controls.progress.json",
                pool,
                args.progress_every,
            )

    canonicalize_jsonl(sources_path, ("pair_source_id",))
    canonicalize_jsonl(controls_path, ("pair_source_id", "k"))
    final_sources = _load_jsonl(sources_path, ("pair_source_id",))
    final_controls = _load_jsonl(controls_path, ("pair_source_id", "k"))
    metadata.update(
        {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "elapsed_sec": round(time.time() - start, 3),
            "source_rows": len(final_sources),
            "valid_source_rows": sum(bool(row.get("valid")) for row in final_sources.values()),
            "historical_topology_shape_matches": sum(
                bool(row.get("topology_shape_match"))
                for row in final_sources.values()
            ),
            "historical_topology_shape_mismatches": sum(
                row.get("topology_shape_match") is False
                for row in final_sources.values()
            ),
            "control_rows": len(final_controls),
            "valid_control_rows": sum(bool(row.get("valid")) for row in final_controls.values()),
            "trajectory_assertion_failures": 0,
            "science_file_sha256": {
                "sources.jsonl": _sha256(sources_path),
                "controls.jsonl": _sha256(controls_path),
            },
        }
    )
    _write_json(out_dir / "metadata.json", metadata)
    logger.info("complete: %s", out_dir)


if __name__ == "__main__":
    main()
