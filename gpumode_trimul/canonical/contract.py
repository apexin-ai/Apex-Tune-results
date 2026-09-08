"""Immutable TriMul benchmark contract used by every Apex evaluation.

The evaluator is copied from the pinned TTT-Discover source tree.  This module
contains only metadata and parsing-independent policy; it deliberately does
not contain a candidate-specific optimization or a local timing calibration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

CANONICAL_UPSTREAM = "https://github.com/test-time-training/discover"
CANONICAL_UPSTREAM_PIN = "6c40e82dab9d5de7416ac873ad5cd3106084aaed"
TASK_NAME = "trimul"
HARDWARE = "H100"
OFFICIAL_MODE = "leaderboard"
PRIVATE_MODE = "private"
OFFICIAL_TIMEOUT_SECONDS = 1200

# This is the public paper snapshot, not a live leaderboard query.  It is a
# starting incumbent only; an official receipt is required before claiming SOTA.
PAPER_INCUMBENT_US = 1161.2


@dataclass(frozen=True)
class BenchmarkCase:
    seqlen: int
    bs: int
    dim: int
    hiddendim: int
    seed: int
    nomask: bool
    distribution: str

    def to_spec(self) -> str:
        return ";".join(
            (
                f"seqlen: {self.seqlen}",
                f"bs: {self.bs}",
                f"dim: {self.dim}",
                f"hiddendim: {self.hiddendim}",
                f"seed: {self.seed}",
                f"nomask: {'True' if self.nomask else 'False'}",
                f"distribution: {self.distribution}",
            )
        )


def canonical_manifest_sha256(root: Path | None = None) -> str:
    """Hash the copied evaluator files in a stable order."""

    root = root or Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("task.yml", "task.py", "utils.py", "reference.py", "eval.py"):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def promotion_threshold(incumbent_us: float, relative_improvement: float = 0.005) -> float:
    """Return the strict upper latency bound for a candidate promotion."""

    if incumbent_us <= 0 or not 0 < relative_improvement < 1:
        raise ValueError("incumbent and relative_improvement must be valid")
    return incumbent_us * (1.0 - relative_improvement)

