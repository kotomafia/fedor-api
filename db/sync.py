import asyncio
from contextlib import asynccontextmanager

from api.db.engine import async_session_factory


@asynccontextmanager
async def session_scope():
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def run_async(coro):
    """В Celery-таске запускаем coroutine через свежий event loop."""
    return asyncio.run(coro)