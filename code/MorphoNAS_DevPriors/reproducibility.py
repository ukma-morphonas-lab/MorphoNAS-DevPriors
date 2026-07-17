"""Fail-closed runtime controls for the canonical paired-reward rerun.

The development environment remains convenient and broad.  The canonical campaign
uses a separate, exact Linux/CPython/CPU lock and verifies it before any result
row is written.  This module also owns the canonical graph and JSONL hashes so
the runner, canary, and audit scripts cannot silently define them differently.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import struct
from pathlib import Path


EARLY_RUNTIME_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "MKL_CBWR": "COMPATIBLE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def set_early_runtime_environment() -> None:
    """Set CPU/thread controls before NumPy or PyTorch is imported.

    Existing conflicting values are deliberately retained so the subsequent
    fail-closed environment check reports them instead of hiding them.
    ``PYTHONHASHSEED`` is not set here: it only takes effect at interpreter
    startup and therefore must be supplied by the launcher.
    """

    for name, value in EARLY_RUNTIME_ENVIRONMENT.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_graph_sha256(graph) -> str:
    """Hash the behavior-relevant directed topology and float64 weights.

    MorphoNAS graphs use integer node IDs.  Hashing their binary identities and
    IEEE-754 weight bits avoids dependence on NetworkX iteration order or text
    float formatting.
    """

    nodes = sorted(int(node) for node in graph.nodes())
    if len(nodes) != graph.number_of_nodes():
        raise RuntimeError("graph node identifiers are not unique integers")
    edges = sorted(
        (int(source), int(target), float(data["weight"]))
        for source, target, data in graph.edges(data=True)
    )
    digest = hashlib.sha256()
    digest.update(b"MorphoNAS-graph-v1\0")
    digest.update(struct.pack("<Q", len(nodes)))
    for node in nodes:
        digest.update(struct.pack("<q", node))
    digest.update(struct.pack("<Q", len(edges)))
    for source, target, weight in edges:
        digest.update(struct.pack("<qqd", source, target, weight))
    return digest.hexdigest()


def canonicalize_jsonl(path: str | Path, key_fields: tuple[str, ...]) -> None:
    """Atomically rewrite a JSONL file in canonical key and JSON order."""

    path = Path(path)
    if not path.exists():
        path.touch()
        return
    rows: dict[tuple, dict] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = tuple(row[field] for field in key_fields)
            if key in rows:
                raise RuntimeError(f"duplicate key {key} in {path}:{line_number}")
            rows[key] = row
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(rows):
            handle.write(
                json.dumps(
                    rows[key], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def configure_torch_cpu() -> dict:
    """Configure the deterministic single-threaded CPU execution contract."""

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this only before inter-op work starts.  A later check
        # still verifies the value, so swallowing here cannot weaken the gate.
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        raise RuntimeError("canonical reproduction is CPU-only, but CUDA is available")
    return {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
    }


def environment_fingerprint() -> dict:
    """Return the public scientific-runtime fingerprint (no hostname)."""

    import numpy
    import scipy
    import torch

    try:
        from threadpoolctl import threadpool_info

        pools = [
            {
                key: entry.get(key)
                for key in (
                    "user_api",
                    "internal_api",
                    "num_threads",
                    "prefix",
                    "version",
                    "threading_layer",
                    "architecture",
                )
                if entry.get(key) is not None
            }
            for entry in threadpool_info()
        ]
    except Exception as error:  # pragma: no cover - diagnostic fallback
        pools = [{"error": f"{type(error).__name__}: {error}"}]
    libc, libc_version = platform.libc_ver()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "libc": libc,
        "libc_version": libc_version,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "gymnasium",
                "matplotlib",
                "networkx",
                "numpy",
                "pygame",
                "scipy",
                "threadpoolctl",
                "torch",
            )
        },
        "numpy_runtime": numpy.__version__,
        "scipy_runtime": scipy.__version__,
        "torch_runtime": torch.__version__,
        "runtime_environment": {
            name: os.environ.get(name)
            for name in sorted(
                {**EARLY_RUNTIME_ENVIRONMENT, "PYTHONHASHSEED": "0"}
            )
        },
        "torch": {
            "cuda_available": bool(torch.cuda.is_available()),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "threads": int(torch.get_num_threads()),
            "interop_threads": int(torch.get_num_interop_threads()),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        },
        "threadpools": pools,
    }


def verify_locked_environment(lock_path: str | Path) -> dict:
    """Configure and verify the exact environment described by ``lock_path``."""

    lock_path = Path(lock_path).resolve()
    lock = json.loads(lock_path.read_text())
    failures: list[str] = []

    requirements = lock["requirements"]
    repo_root = lock_path.parent.parent
    requirements_path = repo_root / requirements["path"]
    locked_packages: dict[str, str] = {}
    if not requirements_path.is_file():
        failures.append(f"missing requirements file: {requirements_path}")
    else:
        actual_hash = sha256_file(requirements_path)
        if actual_hash != requirements["sha256"]:
            failures.append(
                f"requirements SHA-256 {actual_hash} != {requirements['sha256']}"
            )
        for line in requirements_path.read_text().splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^ \\]+)", line)
            if match:
                locked_packages[match.group(1)] = match.group(2)

    expected_platform = lock["platform"]
    libc, libc_version = platform.libc_ver()
    actual_platform = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "libc": libc,
        "libc_version": libc_version,
    }
    for key, expected in expected_platform.items():
        if actual_platform[key] != expected:
            failures.append(
                f"platform.{key}={actual_platform[key]!r}, expected {expected!r}"
            )

    for name, expected in lock["runtime_environment"].items():
        actual = os.environ.get(name)
        if actual != expected:
            failures.append(f"environment {name}={actual!r}, expected {expected!r}")

    for name, expected in lock["packages"].items():
        locked = locked_packages.get(name)
        if locked != expected:
            failures.append(
                f"environment lock expects {name}=={expected}, requirements pin {locked}"
            )

    actual_locked_packages: dict[str, str] = {}
    for name, expected in locked_packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"package {name} is not installed")
            continue
        actual_locked_packages[name] = actual
        if actual != expected:
            failures.append(f"package {name}=={actual}, expected {expected}")

    try:
        torch_state = configure_torch_cpu()
    except Exception as error:
        failures.append(f"PyTorch configuration failed: {type(error).__name__}: {error}")
        torch_state = {}
    for key in (
        "cuda_available",
        "deterministic_algorithms",
        "threads",
        "interop_threads",
        "mkldnn_enabled",
    ):
        if key in torch_state and torch_state[key] != lock["torch"][key]:
            failures.append(
                f"torch.{key}={torch_state[key]!r}, expected {lock['torch'][key]!r}"
            )

    fingerprint: dict = {}
    try:
        fingerprint = environment_fingerprint()
        for pool in fingerprint["threadpools"]:
            threads = pool.get("num_threads")
            if threads is not None and int(threads) != 1:
                failures.append(
                    f"threadpool {pool.get('internal_api', pool.get('prefix'))} "
                    f"uses {threads} threads, expected 1"
                )
    except Exception as error:
        failures.append(
            f"environment fingerprint failed: {type(error).__name__}: {error}"
        )

    if failures:
        joined = "\n  - ".join(failures)
        raise RuntimeError(f"canonical reproduction environment check failed:\n  - {joined}")
    fingerprint["lock_path"] = str(lock_path.relative_to(repo_root))
    fingerprint["lock_sha256"] = sha256_file(lock_path)
    fingerprint["requirements_sha256"] = requirements["sha256"]
    fingerprint["locked_packages"] = dict(sorted(actual_locked_packages.items()))
    fingerprint["verified"] = True
    return fingerprint
