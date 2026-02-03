"""Rate limiting and retry utilities for API connectors."""
from __future__ import annotations

import asyncio
import time
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx
import structlog

log = structlog.get_logger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Simple rate limiter using token bucket algorithm."""

    def __init__(self, max_calls: int, period_seconds: float):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            period_seconds: Time period in seconds
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls: list[float] = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a call, waiting if necessary."""
        async with self.lock:
            now = time.time()
            # Remove calls older than the period
            self.calls = [t for t in self.calls if now - t < self.period_seconds]

            if len(self.calls) >= self.max_calls:
                # Need to wait until the oldest call expires
                oldest = min(self.calls)
                wait_time = self.period_seconds - (now - oldest) + 0.1  # Small buffer
                if wait_time > 0:
                    log.debug("rate_limit_wait", wait_seconds=wait_time)
                    await asyncio.sleep(wait_time)
                    # Re-check after waiting
                    now = time.time()
                    self.calls = [t for t in self.calls if now - t < self.period_seconds]

            self.calls.append(time.time())


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException),
):
    """
    Decorator to retry async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        log.warning(
                            "api_retry",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        log.error(
                            "api_retry_exhausted",
                            function=func.__name__,
                            attempts=max_retries + 1,
                            error=str(e),
                        )
                        raise
                except Exception as e:
                    # Don't retry on non-retryable exceptions
                    log.error("api_non_retryable_error", function=func.__name__, error=str(e))
                    raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


# Default rate limiters for common APIs
# Kalshi: Typically allows ~100 requests/minute for public endpoints
KALSHI_RATE_LIMITER = RateLimiter(max_calls=100, period_seconds=60.0)

# Polymarket: GraphQL endpoints typically allow ~1000 requests/minute
POLYMARKET_RATE_LIMITER = RateLimiter(max_calls=1000, period_seconds=60.0)

