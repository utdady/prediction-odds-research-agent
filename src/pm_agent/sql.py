from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def execute(session: AsyncSession, sql: str, params: dict[str, Any] | None = None) -> None:
    await session.execute(text(sql), params or {})


async def fetch_all(session: AsyncSession, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    res = await session.execute(text(sql), params or {})
    return [dict(r._mapping) for r in res.fetchall()]


async def fetch_one(session: AsyncSession, sql: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    res = await session.execute(text(sql), params or {})
    row = res.fetchone()
    return dict(row._mapping) if row else None

