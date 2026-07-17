"""
Reliable progress reporting for long, multi-step runs (fleet-friendly).

Writes two things, both flushed and fsync'd so an external orchestrator can
always read a consistent state and compute an ETA, even mid-run or if the job
is killed between writes:

  * <path> (e.g. progress.json) -- an atomically replaced JSON snapshot
    {phase, done, total, frac, elapsed_sec, rate_sec_per_unit, eta_sec,
     eta_clock, updated_at, counts, meta}.
  * <path-without-.json>.heartbeat.jsonl -- an append-only log of every
    snapshot, so rate/ETA can be recomputed from history if needed.

The atomic snapshot is written to a temp file then os.replace()'d, which is
atomic on POSIX, so a reader never sees a half-written file. Call update()
every N units (not every unit) to keep fsync overhead negligible.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional


class ProgressWriter:
    def __init__(
        self,
        path: str,
        total: int,
        *,
        phase: str = "run",
        meta: Optional[dict] = None,
        heartbeat: bool = True,
    ) -> None:
        self.path = path
        self.total = int(total)
        self.phase = phase
        self.meta = dict(meta or {})
        self.t0 = time.time()
        self.hb_path: Optional[str] = None
        if heartbeat:
            stem = path[:-5] if path.endswith(".json") else path
            self.hb_path = stem + ".heartbeat.jsonl"
            # truncate any stale heartbeat from a previous run
            with open(self.hb_path, "w") as f:
                f.flush()
                os.fsync(f.fileno())
        self.update(0, phase="start")

    def _atomic_write(self, obj: dict) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def _snapshot(self, done: int, phase: Optional[str], counts: dict) -> dict:
        now = time.time()
        elapsed = now - self.t0
        done = int(done)
        rate = elapsed / done if done > 0 else 0.0
        remaining = max(self.total - done, 0)
        eta = rate * remaining if done > 0 else None
        return {
            "phase": phase or self.phase,
            "done": done,
            "total": self.total,
            "frac": (done / self.total) if self.total else 0.0,
            "elapsed_sec": round(elapsed, 2),
            "rate_sec_per_unit": round(rate, 5),
            "eta_sec": round(eta, 1) if eta is not None else None,
            "eta_clock": (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now + eta))
                if eta is not None
                else None
            ),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "counts": counts,
            "meta": self.meta,
        }

    def update(self, done: int, *, phase: Optional[str] = None, **counts) -> None:
        snap = self._snapshot(done, phase, counts)
        self._atomic_write(snap)
        if self.hb_path is not None:
            with open(self.hb_path, "a") as f:
                f.write(json.dumps(snap) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def done(self, **counts) -> None:
        self.update(self.total, phase="complete", **counts)
