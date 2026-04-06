"""Strategy profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...schemas.strategy_profile import StrategyProfileDetail, StrategyProfileListResponse
from ...services.strategy_profile_service import StrategyProfileService

router = APIRouter()


def _get_strategy_profile_service() -> StrategyProfileService:
    return StrategyProfileService()


@router.get("", response_model=StrategyProfileListResponse, include_in_schema=False)
@router.get("/", response_model=StrategyProfileListResponse)
async def list_strategy_profiles(
    service: StrategyProfileService = Depends(_get_strategy_profile_service),
) -> StrategyProfileListResponse:
    """Return all available strategy profiles."""

    return service.list_profiles()


@router.get("/{profile}", response_model=StrategyProfileDetail)
async def get_strategy_profile(
    profile: str,
    service: StrategyProfileService = Depends(_get_strategy_profile_service),
) -> StrategyProfileDetail:
    """Return one strategy profile, falling back to default if unknown."""

    return service.get_profile(profile)
