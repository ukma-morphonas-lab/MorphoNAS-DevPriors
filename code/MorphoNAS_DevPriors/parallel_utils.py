"""
Shared utilities for multiprocessing-based sweep scripts.

The single public function here — configure_worker_threads() — must be called
at the top of every worker initializer and at the top of main() in any script
that launches a multiprocessing pool. Without it, libraries like OpenBLAS,
MKL, and PyTorch spawn one thread per CPU core inside each worker process,
causing severe thread contention on multi-core servers.
"""

from __future__ import annotations


def configure_worker_threads() -> None:
    """Pin this process to single-threaded operation.

    OpenBLAS, MKL, and PyTorch each spawn up to N_CPU threads by default.
    In a multiprocessing pool with W workers, that produces W × N_CPU threads
    competing for N_CPU cores — catastrophic contention. Setting these
    environment variables to "1" and calling torch.set_num_threads(1) keeps
    each worker strictly single-threaded, so W workers actually run in parallel
    on W cores.

    Uses setdefault so an explicit environment variable set by the caller is
    respected (e.g. OMP_NUM_THREADS=4 for a CPU-bound non-pool context).
    """
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass
