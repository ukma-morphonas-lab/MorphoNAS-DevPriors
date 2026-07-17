#!/usr/bin/env python3
"""Fail closed unless the canonical paired-reward environment is active."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from MorphoNAS_DevPriors.reproducibility import (  # noqa: E402
    set_early_runtime_environment,
    verify_locked_environment,
)

set_early_runtime_environment()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock", default=str(REPO_ROOT / "reproduction/environment.json")
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        fingerprint = verify_locked_environment(args.lock)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    if args.json:
        print(json.dumps(fingerprint, indent=2, sort_keys=True))
    else:
        packages = fingerprint["packages"]
        print(
            "canonical reproduction environment verified: "
            f"{fingerprint['system']} {fingerprint['machine']}, "
            f"CPython {fingerprint['python_version']}, "
            f"NumPy {packages['numpy']}, SciPy {packages['scipy']}, "
            f"PyTorch {packages['torch']} CPU"
        )


if __name__ == "__main__":
    main()
