"""Usage aggregation helpers (pi's `harness/utils/usage.ts`)."""

from __future__ import annotations

from karen_ai import Usage
from karen_ai.types import UsageCost


def empty_usage() -> Usage:
    return Usage()


def _sum_optional(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


def add_usage(left: Usage, right: Usage) -> Usage:
    return Usage(
        input=left.input + right.input,
        output=left.output + right.output,
        cache_read=left.cache_read + right.cache_read,
        cache_write=left.cache_write + right.cache_write,
        cache_write1h=_sum_optional(left.cache_write1h, right.cache_write1h),
        reasoning=_sum_optional(left.reasoning, right.reasoning),
        total_tokens=left.total_tokens + right.total_tokens,
        cost=UsageCost(
            input=left.cost.input + right.cost.input,
            output=left.cost.output + right.cost.output,
            cache_read=left.cost.cache_read + right.cost.cache_read,
            cache_write=left.cost.cache_write + right.cost.cache_write,
            total=left.cost.total + right.cost.total,
        ),
    )
