from __future__ import annotations

from datetime import datetime, timezone


def validate_probability(p: float) -> tuple[bool, str | None]:
    if p < 0 or p > 1:
        return False, f"probability out of bounds: {p}"
    return True, None


def validate_tick_ts(ts: datetime) -> tuple[bool, str | None]:
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return False, "tick_ts must be timezone-aware"
    if ts > now:
        return False, "tick_ts is in the future"
    return True, None

