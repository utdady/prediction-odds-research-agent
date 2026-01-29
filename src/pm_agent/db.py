from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from pm_agent.config import settings

_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """Get a singleton async engine with explicit connection pooling.

    Notes
    -----
    - pool_pre_ping: Detects stale connections before using them
    - pool_size: Number of persistent connections in the pool
    - max_overflow: Extra connections allowed above pool_size
    - pool_recycle: Recycle connections after N seconds to avoid server timeouts
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url_async,
            pool_pre_ping=True,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_recycle=settings.db_pool_recycle,
            echo_pool=settings.debug,
        )
    return _engine


# Create sessionmaker bound to the singleton engine
AsyncSessionLocal = async_sessionmaker(get_engine(), expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncSession:
    """Async context manager that yields a pooled session."""
    async with AsyncSessionLocal() as session:
        yield session

