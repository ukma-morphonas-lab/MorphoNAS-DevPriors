#!/usr/bin/env python3
"""Run the fast numerical canary before a canonical reproduction campaign.

The expected fixture is intentionally absent until it is recorded once in the
fresh locked Linux environment.  Recording never overwrites an existing
fixture.  Normal verification re-executes three tiny paired runs and compares
their canonical science-file hashes and graph/trajectory hashes exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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
CANARY_SOURCE_ID = {
    "mountaincar": 1,
    "pendulum": 1,
    # network_00056: known to change (N,E) under the previously drifted server
    # environment, so it is deliberately the numerical-lineage sentinel.
    "acrobot": 15,
}


def first_jsonl(path: Path) -> dict:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"empty canary output: {path}")


def run_canary(lock_path: Path, work_dir: Path) -> dict:
    environment = verify_locked_environment(lock_path)
    result = {
        "schema_version": 1,
        "configuration": {
            "sources_per_pair": 1,
            "manifest_source_ids": CANARY_SOURCE_ID,
            "random_requests_per_source": 1,
            "rollouts": 3,
            "workers": 1,
        },
        "environment_lock_sha256": sha256_file(lock_path),
        "requirements_sha256": environment["requirements_sha256"],
        "runner_sha256": sha256_file(REPO_ROOT / "scripts/run_paired_reward.py"),
        "pairs": {},
    }
    for pair in PAIRS:
        output = work_dir / pair
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_paired_reward.py"),
            "--pair",
            pair,
            "--output-dir",
            str(output),
            "--max-sources",
            "1",
            "--source-ids",
            str(CANARY_SOURCE_ID[pair]),
            "--num-random",
            "1",
            "--rollouts",
            "3",
            "--max-workers",
            "1",
            "--progress-every",
            "1",
            "--environment-lock",
            str(lock_path),
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        source = first_jsonl(output / "sources.jsonl")
        control = first_jsonl(output / "controls.jsonl")
        result["pairs"][pair] = {
            "manifest_sha256": sha256_file(
                REPO_ROOT / "reproduction/manifests" / f"{pair}.jsonl"
            ),
            "sources_sha256": sha256_file(output / "sources.jsonl"),
            "controls_sha256": sha256_file(output / "controls.jsonl"),
            "source_graph_sha256": source.get("graph_sha256"),
            "source_trajectory_sha256": source.get("trajectory_sha256"),
            "control_valid": bool(control.get("valid")),
            "control_graph_sha256": control.get("graph_sha256"),
            "control_trajectory_sha256": control.get("trajectory_sha256"),
        }
    return result


def atomic_write(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", default=str(REPO_ROOT / "reproduction/environment.json")
    )
    parser.add_argument(
        "--expected",
        default=str(REPO_ROOT / "reproduction/canary.expected.json"),
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--work-parent", default=None)
    args = parser.parse_args()
    lock_path = Path(args.lock).resolve()
    expected_path = Path(args.expected).resolve()

    with tempfile.TemporaryDirectory(
        prefix="devpriors-paired-reward-canary-", dir=args.work_parent
    ) as directory:
        actual = run_canary(lock_path, Path(directory))

    if args.record:
        if expected_path.exists():
            raise SystemExit(f"refusing to overwrite {expected_path}")
        atomic_write(expected_path, actual)
        print(f"recorded canonical canary: {expected_path}")
        return
    if not expected_path.is_file():
        raise SystemExit(
            f"missing {expected_path}; record it once in the newly locked Linux environment"
        )
    expected = json.loads(expected_path.read_text())
    if actual != expected:
        print("expected:", json.dumps(expected, indent=2, sort_keys=True), file=sys.stderr)
        print("actual:", json.dumps(actual, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit("paired-reward numerical canary failed")
    print("canonical numerical canary verified for all three reward pairs")


if __name__ == "__main__":
    main()
