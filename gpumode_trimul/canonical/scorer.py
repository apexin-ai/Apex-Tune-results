"""Exact parser and aggregator for the official GPUMode TriMul evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


class TriMulScoreError(ValueError):
    """Raised when an evaluator trace cannot support a score."""


@dataclass(frozen=True)
class TriMulScore:
    latency_us: float
    benchmark_latencies_us: tuple[float, ...]
    benchmark_specs: tuple[str, ...]
    correctness_passed: bool
    private_seed_used: bool
    raw_fields: Mapping[str, str]

    @property
    def valid(self) -> bool:
        return self.correctness_passed and math.isfinite(self.latency_us) and self.latency_us > 0


def parse_popcorn_fields(text: str) -> dict[str, str]:
    """Parse one evaluator phase, rejecting ambiguous duplicate fields."""

    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        key = key.strip()
        if not key:
            continue
        if key in fields:
            raise TriMulScoreError(f"duplicate evaluator field: {key}")
        fields[key] = value.strip()
    return fields


def _validate_ordered_specs(
    fields: Mapping[str, str],
    *,
    prefix: str,
    count: int,
    expected_specs: Sequence[str] | None,
) -> tuple[str, ...]:
    specs = tuple(fields.get(f"{prefix}.{index}.spec", "") for index in range(count))
    if any(not spec for spec in specs):
        missing = next(index for index, spec in enumerate(specs) if not spec)
        raise TriMulScoreError(f"missing {prefix}.{missing}.spec")
    if expected_specs is not None and specs != tuple(expected_specs):
        raise TriMulScoreError(f"{prefix} specs do not match the canonical ordered manifest")
    return specs


def _positive_float(value: str, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TriMulScoreError(f"{field} is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise TriMulScoreError(f"{field} must be finite and positive")
    return number


def geometric_mean(values: tuple[float, ...]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise TriMulScoreError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def score_popcorn_output(
    text: str,
    *,
    private_seed_used: bool = False,
    expected_specs: Sequence[str] | None = None,
) -> TriMulScore:
    fields = parse_popcorn_fields(text)
    if fields.get("check") != "pass":
        raise TriMulScoreError(f"official evaluator did not pass: {fields.get('check')!r}")
    try:
        count = int(fields["benchmark-count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TriMulScoreError("missing or invalid benchmark-count") from exc
    if count != 7:
        raise TriMulScoreError(f"TriMul contract requires 7 benchmarks, got {count}")

    means_ns: list[float] = []
    specs = _validate_ordered_specs(
        fields,
        prefix="benchmark",
        count=count,
        expected_specs=expected_specs,
    )
    for index in range(count):
        means_ns.append(_positive_float(fields.get(f"benchmark.{index}.mean", ""), f"benchmark.{index}.mean"))
    latencies_us = tuple(value / 1000.0 for value in means_ns)
    return TriMulScore(
        latency_us=geometric_mean(latencies_us),
        benchmark_latencies_us=latencies_us,
        benchmark_specs=specs,
        correctness_passed=True,
        private_seed_used=private_seed_used,
        raw_fields=fields,
    )


def validate_test_output(
    text: str, *, expected_specs: Sequence[str] | None = None
) -> tuple[str, ...]:
    """Validate the complete correctness phase and return its ordered specs."""

    fields = parse_popcorn_fields(text)
    if fields.get("check") != "pass":
        raise TriMulScoreError(f"correctness evaluator did not pass: {fields.get('check')!r}")
    try:
        count = int(fields["test-count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TriMulScoreError("missing or invalid test-count") from exc
    if count != 18:
        raise TriMulScoreError(f"TriMul contract requires 18 correctness tests, got {count}")
    specs = _validate_ordered_specs(
        fields,
        prefix="test",
        count=count,
        expected_specs=expected_specs,
    )
    failed = [index for index in range(count) if fields.get(f"test.{index}.status") != "pass"]
    if failed:
        raise TriMulScoreError(f"correctness cases did not pass: {failed}")
    return specs


def parse_test_output(text: str, *, expected_specs: Sequence[str] | None = None) -> bool:
    try:
        validate_test_output(text, expected_specs=expected_specs)
    except TriMulScoreError:
        return False
    return True


def beats_incumbent(candidate_us: float, incumbent_us: float, *, min_relative_improvement: float = 0.005) -> bool:
    if candidate_us <= 0 or incumbent_us <= 0:
        raise ValueError("latencies must be positive")
    if not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must be in [0, 1)")
    return candidate_us < incumbent_us * (1.0 - min_relative_improvement)
