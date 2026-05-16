from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.engine import get_session
from api.db.repositories import (
    GuildIgnoredChannelRepository,
    GuildIgnoredRoleRepository,
    GuildSettingsRepository,
    GuildWhitelistRepository,
)
from api.schemas import (
    GuildSettingsPatchRequest,
    GuildSettingsResponse,
    IgnoredChannelAddRequest,
    IgnoredChannelEntry,
    IgnoredRoleAddRequest,
    IgnoredRoleEntry,
    LogLevelEnum,
    WhitelistAddRequest,
    WhitelistEntry,
)

router = APIRouter(prefix="/api/v1/guilds", tags=["guilds"])


@router.get("/{guild_id}/settings", response_model=GuildSettingsResponse)
async def get_guild_settings(
    guild_id: str,
    session: AsyncSession = Depends(get_session),
) -> GuildSettingsResponse:
    """Агрегированный snapshot для бота: настройки + игнор-списки.

    Бот кеширует ответ (TTL ~60 с) и инвалидирует на любых мутирующих
    командах. Один HTTP-запрос на гильдию вместо четырёх.
    """
    settings_repo = GuildSettingsRepository(session)
    channels_repo = GuildIgnoredChannelRepository(session)
    roles_repo = GuildIgnoredRoleRepository(session)
    whitelist_repo = GuildWhitelistRepository(session)

    gs = await settings_repo.get(guild_id)
    ignored_channels = await channels_repo.get_channel_ids(guild_id)
    ignored_roles = await roles_repo.get_role_ids(guild_id)
    whitelist = await whitelist_repo.get_phrases(guild_id)

    return GuildSettingsResponse(
        guild_id=guild_id,
        review_channel_id=gs.review_channel_id if gs else None,
        log_channel_id=gs.log_channel_id if gs else None,
        log_level=LogLevelEnum(gs.log_level) if gs else LogLevelEnum.BLOCK_FLAG,
        ignored_channel_ids=ignored_channels,
        ignored_role_ids=ignored_roles,
        whitelist_count=len(whitelist),
    )


@router.patch("/{guild_id}/settings", response_model=GuildSettingsResponse)
async def patch_guild_settings(
    guild_id: str,
    payload: GuildSettingsPatchRequest,
    session: AsyncSession = Depends(get_session),
) -> GuildSettingsResponse:
    settings_repo = GuildSettingsRepository(session)

    fields_set = payload.model_fields_set
    kwargs: dict = {}
    if "review_channel_id" in fields_set:
        kwargs["review_channel_id"] = payload.review_channel_id
    if "log_channel_id" in fields_set:
        kwargs["log_channel_id"] = payload.log_channel_id
    if "log_level" in fields_set:
        kwargs["log_level"] = (
            payload.log_level.value if payload.log_level else LogLevelEnum.BLOCK_FLAG.value
        )

    await settings_repo.patch(guild_id, **kwargs)
    await session.commit()

    return await get_guild_settings(guild_id, session)


@router.get(
    "/{guild_id}/whitelist",
    response_model=list[WhitelistEntry],
)
async def list_whitelist(
    guild_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[WhitelistEntry]:
    repo = GuildWhitelistRepository(session)
    items = await repo.list(guild_id)
    return [
        WhitelistEntry(
            id=item.id,
            guild_id=item.guild_id,
            phrase=item.phrase,
            added_by=item.added_by,
            added_at=item.added_at,
        )
        for item in items
    ]


@router.post(
    "/{guild_id}/whitelist",
    response_model=WhitelistEntry,
    status_code=status.HTTP_201_CREATED,
)
async def add_whitelist(
    guild_id: str,
    payload: WhitelistAddRequest,
    session: AsyncSession = Depends(get_session),
) -> WhitelistEntry:
    repo = GuildWhitelistRepository(session)
    entry = await repo.add(guild_id, payload.phrase, payload.added_by)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="phrase already in whitelist",
        )
    await session.commit()
    return WhitelistEntry(
        id=entry.id,
        guild_id=entry.guild_id,
        phrase=entry.phrase,
        added_by=entry.added_by,
        added_at=entry.added_at,
    )


@router.delete(
    "/{guild_id}/whitelist/{phrase_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_whitelist(
    guild_id: str,
    phrase_id: int,
    session: AsyncSession = Depends(get_session),
) -> None:
    repo = GuildWhitelistRepository(session)
    deleted = await repo.delete(guild_id, phrase_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="whitelist entry not found",
        )
    await session.commit()


@router.get(
    "/{guild_id}/ignored-channels",
    response_model=list[IgnoredChannelEntry],
)
async def list_ignored_channels(
    guild_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[IgnoredChannelEntry]:
    repo = GuildIgnoredChannelRepository(session)
    items = await repo.list(guild_id)
    return [
        IgnoredChannelEntry(
            id=item.id,
            guild_id=item.guild_id,
            channel_id=item.channel_id,
            added_by=item.added_by,
            added_at=item.added_at,
        )
        for item in items
    ]


@router.post(
    "/{guild_id}/ignored-channels",
    response_model=IgnoredChannelEntry,
    status_code=status.HTTP_201_CREATED,
)
async def add_ignored_channel(
    guild_id: str,
    payload: IgnoredChannelAddRequest,
    session: AsyncSession = Depends(get_session),
) -> IgnoredChannelEntry:
    repo = GuildIgnoredChannelRepository(session)
    entry = await repo.add(guild_id, payload.channel_id, payload.added_by)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="channel already ignored",
        )
    await session.commit()
    return IgnoredChannelEntry(
        id=entry.id,
        guild_id=entry.guild_id,
        channel_id=entry.channel_id,
        added_by=entry.added_by,
        added_at=entry.added_at,
    )


@router.delete(
    "/{guild_id}/ignored-channels/{channel_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_ignored_channel(
    guild_id: str,
    channel_id: str,
    session: AsyncSession = Depends(get_session),
) -> None:
    repo = GuildIgnoredChannelRepository(session)
    deleted = await repo.delete(guild_id, channel_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ignored channel entry not found",
        )
    await session.commit()


@router.get(
    "/{guild_id}/ignored-roles",
    response_model=list[IgnoredRoleEntry],
)
async def list_ignored_roles(
    guild_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[IgnoredRoleEntry]:
    repo = GuildIgnoredRoleRepository(session)
    items = await repo.list(guild_id)
    return [
        IgnoredRoleEntry(
            id=item.id,
            guild_id=item.guild_id,
            role_id=item.role_id,
            added_by=item.added_by,
            added_at=item.added_at,
        )
        for item in items
    ]


@router.post(
    "/{guild_id}/ignored-roles",
    response_model=IgnoredRoleEntry,
    status_code=status.HTTP_201_CREATED,
)
async def add_ignored_role(
    guild_id: str,
    payload: IgnoredRoleAddRequest,
    session: AsyncSession = Depends(get_session),
) -> IgnoredRoleEntry:
    repo = GuildIgnoredRoleRepository(session)
    entry = await repo.add(guild_id, payload.role_id, payload.added_by)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="role already ignored",
        )
    await session.commit()
    return IgnoredRoleEntry(
        id=entry.id,
        guild_id=entry.guild_id,
        role_id=entry.role_id,
        added_by=entry.added_by,
        added_at=entry.added_at,
    )


@router.delete(
    "/{guild_id}/ignored-roles/{role_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_ignored_role(
    guild_id: str,
    role_id: str,
    session: AsyncSession = Depends(get_session),
) -> None:
    repo = GuildIgnoredRoleRepository(session)
    deleted = await repo.delete(guild_id, role_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ignored role entry not found",
        )
    await session.commit()
