from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.engine import get_session
from api.db.models import Verdict

router = APIRouter(prefix="/api/v1/stats", tags=["stats"])


@router.get("/recent")
async def recent_stats(
    guild_id: str | None = None,
    hours: int = 24,
    session: AsyncSession = Depends(get_session),
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    stmt = (
        select(
            Verdict.action,
            Verdict.source_kind,
            func.count().label("count"),
            func.avg(Verdict.score).label("avg_score"),
            func.avg(Verdict.inference_ms).label("avg_ms"),
        )
        .where(Verdict.created_at >= since)
        .group_by(Verdict.action, Verdict.source_kind)
    )
    if guild_id:
        stmt = stmt.where(Verdict.guild_id == guild_id)

    rows = (await session.execute(stmt)).all()
    return [
        {
            "action": r.action, "source": r.source_kind,
            "count": r.count, "avg_score": float(r.avg_score),
            "avg_ms": float(r.avg_ms) if r.avg_ms else None,
        }
        for r in rows
    ]


@router.get("/cache-efficiency")
async def cache_efficiency(
    hours: int = 24,
    session: AsyncSession = Depends(get_session),
):
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    stmt = select(
        Verdict.cache_hit,
        func.count().label("count"),
    ).where(Verdict.created_at >= since).group_by(Verdict.cache_hit)
    rows = (await session.execute(stmt)).all()
    total = sum(r.count for r in rows)
    hits = next((r.count for r in rows if r.cache_hit), 0)
    return {
        "total": total, "cache_hits": hits,
        "hit_rate": hits / total if total > 0 else 0.0,
    }