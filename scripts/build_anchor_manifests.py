#!/usr/bin/env python3
"""Build or verify the exact input manifests for the developmental-priors rerun.

The stored genome JSON is the canonical input.  Seeds remain provenance, but
are insufficient as a byte-exact definition because historical NumPy versions
can differ by one ULP when mapping raw RNG bits to floating-point uniforms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_REPO_ROOT / "code"))

from MorphoNAS_DevPriors.reproducibility import (  # noqa: E402
    canonical_json_sha256,
    sha256_file,
)


PAIR_ANCHORS = {
    "mountaincar": "mountaincar",
    "pendulum": "pendulum_sparse",
    "acrobot": "acrobot",
}


def build_manifest(repo_root: Path, pair: str, max_sources: int) -> bytes:
    task = PAIR_ANCHORS[pair]
    network_dir = repo_root / "experiments" / task / "pool" / "networks"
    records = []
    for path in network_dir.glob("network_*.json"):
        stored = json.loads(path.read_text())
        if not stored.get("valid"):
            continue
        records.append((int(stored["seed"]), int(stored["network_id"]), path, stored))
    records.sort(key=lambda value: (value[0], value[1]))
    records = records[:max_sources]
    if len(records) != max_sources:
        raise RuntimeError(
            f"{network_dir} has {len(records)} valid anchors, expected {max_sources}"
        )

    lines = []
    for source_id, (seed, network_id, path, stored) in enumerate(records, start=1):
        eval_seeds = [
            int(value)
            for value in stored.get("rollout_data", {}).get("eval_seeds", [])
        ]
        if len(eval_seeds) < 20:
            raise RuntimeError(f"{path} contains only {len(eval_seeds)} rollout seeds")
        stats = stored.get("network_stats", {})
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        row = {
            "schema_version": 1,
            "pair": pair,
            "pair_source_id": source_id,
            "anchor_path": relative,
            "anchor_sha256": sha256_file(path),
            "anchor_network_id": network_id,
            "genome_seed": seed,
            "genome_sha256": canonical_json_sha256(stored["genome"]),
            "eval_seeds_sha256": canonical_json_sha256(eval_seeds),
            "rollout_seed_count": len(eval_seeds),
            "historical_network_stats": {
                "neurons": int(stats["neurons"]),
                "connections": int(stats["connections"]),
            },
        }
        lines.append(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        )
    return "".join(lines).encode("utf-8")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(SCRIPT_REPO_ROOT))
    parser.add_argument(
        "--output-dir", default=str(SCRIPT_REPO_ROOT / "reproduction/manifests")
    )
    parser.add_argument("--max-sources", type=int, default=5000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("pairs", nargs="*", choices=sorted(PAIR_ANCHORS))
    args = parser.parse_args()
    if args.max_sources <= 0:
        raise SystemExit("--max-sources must be positive")

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    pairs = args.pairs or list(PAIR_ANCHORS)
    for pair in pairs:
        content = build_manifest(repo_root, pair, args.max_sources)
        destination = output_dir / f"{pair}.jsonl"
        if args.check:
            if not destination.is_file():
                raise SystemExit(f"missing manifest: {destination}")
            if destination.read_bytes() != content:
                raise SystemExit(f"manifest drift: {destination}")
            print(f"verified {destination} ({args.max_sources} anchors)")
        else:
            if destination.exists():
                raise SystemExit(
                    f"refusing to overwrite {destination}; use --check or remove it explicitly"
                )
            atomic_write(destination, content)
            print(
                f"wrote {destination} ({args.max_sources} anchors, "
                f"sha256={sha256_file(destination)})"
            )


if __name__ == "__main__":
    main()
