#!/usr/bin/env python3
"""Run and analyze all three canonical reward pairs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from MorphoNAS_DevPriors.reproducibility import (  # noqa: E402
    set_early_runtime_environment,
    sha256_file,
    verify_locked_environment,
)

set_early_runtime_environment()

PAIRS = ("mountaincar", "pendulum", "acrobot")


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-sources", type=int, default=5000)
    parser.add_argument("--num-random", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_717)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--environment-lock",
        default=str(REPO_ROOT / "reproduction/environment.json"),
    )
    parser.add_argument(
        "--canary",
        default=str(REPO_ROOT / "reproduction/canary.expected.json"),
    )
    args = parser.parse_args()
    for name in ("max_sources", "num_random", "rollouts", "max_workers", "bootstrap"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    lock_path = Path(args.environment_lock).resolve()
    verify_locked_environment(lock_path)
    run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/verify_reproduction_canary.py"),
            "--lock",
            str(lock_path),
            "--expected",
            str(Path(args.canary).resolve()),
        ]
    )

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise SystemExit(
            f"{output_root} is not empty; choose a new path or pass --resume"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    pair_results = {}
    for pair in PAIRS:
        run_dir = output_root / pair
        metadata_path = run_dir / "metadata.json"
        complete = False
        if metadata_path.is_file():
            complete = json.loads(metadata_path.read_text()).get("status") == "complete"
        if not complete:
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/run_paired_reward.py"),
                "--pair",
                pair,
                "--output-dir",
                str(run_dir),
                "--max-sources",
                str(args.max_sources),
                "--num-random",
                str(args.num_random),
                "--rollouts",
                str(args.rollouts),
                "--max-workers",
                str(args.max_workers),
                "--environment-lock",
                str(lock_path),
            ]
            if args.resume and metadata_path.is_file():
                command.append("--resume")
            run(command)
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/analyze_paired_reward.py"),
                "--run-dir",
                str(run_dir),
                "--bootstrap",
                str(args.bootstrap),
                "--seed",
                str(args.bootstrap_seed),
                "--environment-lock",
                str(lock_path),
            ]
        )
        result_path = run_dir / "paired_reward_results.json"
        pair_results[pair] = {
            "path": f"{pair}/paired_reward_results.json",
            "sha256": sha256_file(result_path),
            "result": json.loads(result_path.read_text()),
        }

    campaign = {
        "schema_version": 1,
        "experiment": "canonical_paired_reward_campaign",
        "configuration": {
            "sources_per_pair": args.max_sources,
            "random_requests_per_source": args.num_random,
            "rollouts": args.rollouts,
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "pairs": list(PAIRS),
        },
        "environment_lock_sha256": sha256_file(lock_path),
        "canary_expected_sha256": sha256_file(Path(args.canary).resolve()),
        "pair_results": pair_results,
    }
    destination = output_root / "paired_reward_campaign_results.json"
    atomic_json(destination, campaign)
    print(f"canonical paired-reward campaign complete: {destination}")


if __name__ == "__main__":
    main()
