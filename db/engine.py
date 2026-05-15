from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)

from api.config import api_settings


_engine = create_async_engine(
    api_settings.database_url,
    echo=False,             # True помогает на старте смотреть SQL
    pool_size=10,
    max_overflow=5,
    pool_pre_ping=True,     # ловит мертвые соединения после рестарта Postgres
    pool_recycle=1800,
)

async_session_factory = async_sessionmaker(
    _engine, expire_on_commit=False, class_=AsyncSession,
)


async def get_session() -> AsyncSession:
    """Для DI в FastAPI."""
    async with async_session_factory() as session:
        yield session


def get_engine():
    return _engine