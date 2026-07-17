#!/usr/bin/env python3
"""Start the canonical paired-reward campaign in a named, logged tmux session."""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--session", default="devpriors-paired-reward")
    parser.add_argument(
        "--python", default=str(REPO_ROOT / ".venv-reproduction/bin/python")
    )
    parser.add_argument("--max-sources", type=int, default=5000)
    parser.add_argument("--num-random", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if shutil.which("tmux") is None:
        raise SystemExit("tmux is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.session):
        raise SystemExit("--session may contain only letters, digits, dot, underscore, dash")
    python = Path(args.python).resolve()
    if not python.is_file():
        raise SystemExit(f"Python executable not found: {python}")
    existing = subprocess.run(
        ["tmux", "has-session", "-t", args.session], capture_output=True
    )
    if existing.returncode == 0:
        raise SystemExit(f"tmux session already exists: {args.session}")

    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise SystemExit(
            f"{output_root} is not empty; choose a new path or pass --resume"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_root.parent / f"{output_root.name}.campaign.log"
    if log_path.exists() and not args.resume:
        raise SystemExit(f"refusing to overwrite existing log: {log_path}")

    campaign = [
        str(python),
        str(REPO_ROOT / "scripts/run_paired_reward_campaign.py"),
        "--output-root",
        str(output_root),
        "--max-sources",
        str(args.max_sources),
        "--num-random",
        str(args.num_random),
        "--rollouts",
        str(args.rollouts),
        "--max-workers",
        str(args.max_workers),
        "--bootstrap",
        str(args.bootstrap),
    ]
    if args.resume:
        campaign.append("--resume")
    env_command = ["env", *[f"{key}={value}" for key, value in RUNTIME_ENVIRONMENT.items()]]
    pipeline = (
        "set -o pipefail; cd "
        + shlex.quote(str(REPO_ROOT))
        + "; "
        + shlex.join(env_command + campaign)
        + " 2>&1 | tee "
        + ("-a " if args.resume else "")
        + shlex.quote(str(log_path))
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", args.session, "bash", "-lc", pipeline],
        check=True,
    )
    print(f"started tmux session {args.session}")
    print(f"log: {log_path}")
    print(f"attach: tmux attach -t {args.session}")
    print(f"tail: tail -f {shlex.quote(str(log_path))}")


if __name__ == "__main__":
    main()
