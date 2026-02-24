"""API endpoints for application configuration including LLM settings."""
import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from typing import Optional
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.app_settings import AppSetting
from ...schemas.config import (
    LLMConfigResponse,
    LLMModelUpdate,
    OllamaSettings,
    ThemePolicyConfigResponse,
    ThemePolicyRevertRequest,
    ThemePolicyUpdateRequest,
    ThemePolicyUpdateResponse,
    ThemePolicyVersionSummary,
    ThemePolicyMatcherConfig,
    ThemePolicyLifecycleConfig,
)
from ...config.pipeline_config import get_pipeline_config
from ...services.theme_extraction_service import ThemeExtractionService
from ...services.llm.config import AVAILABLE_MODELS, get_model_by_id
from ...config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def require_admin(
    x_admin_key: str = Header(default=None, alias="X-Admin-Key"),
    authorization: str = Header(default=None),
):
    """Simple admin guard for configuration endpoints."""
    admin_key = settings.admin_api_key
    if not admin_key:
        logger.error("ADMIN_API_KEY not configured; config endpoints disabled")
        raise HTTPException(status_code=503, detail="Admin API key not configured")

    provided = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()
    if not provided:
        provided = x_admin_key

    if not provided or provided != admin_key:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return True


def get_setting(db: Session, key: str, default: str = None) -> Optional[str]:
    """Get a setting value by key."""
    setting = db.query(AppSetting).filter(AppSetting.key == key).first()
    return setting.value if setting else default


def set_setting(db: Session, key: str, value: str, category: str = None, description: str = None):
    """Set a setting value."""
    setting = db.query(AppSetting).filter(AppSetting.key == key).first()
    if setting:
        setting.value = value
        if category:
            setting.category = category
        if description:
            setting.description = description
    else:
        setting = AppSetting(key=key, value=value, category=category, description=description)
        db.add(setting)
    db.commit()
    return setting


def _get_setting_json(db: Session, key: str, default):
    raw = get_setting(db, key)
    if not raw:
        return deepcopy(default)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return deepcopy(default)
    if isinstance(default, dict) and not isinstance(parsed, dict):
        return deepcopy(default)
    if isinstance(default, list) and not isinstance(parsed, list):
        return deepcopy(default)
    return parsed


def _set_setting_json(db: Session, key: str, value, category: str, description: str):
    return set_setting(db, key, json.dumps(value), category=category, description=description)


def _theme_policy_defaults(pipeline: str) -> dict:
    cfg = get_pipeline_config(pipeline)

    def _resolve(override_map: dict[str, float], default_value: float) -> float:
        return float(override_map.get(pipeline, default_value))

    matcher_defaults = {
        "match_default_threshold": float(ThemeExtractionService.MATCH_THRESHOLD_CONFIG.resolve_threshold(pipeline=pipeline)),
        "fuzzy_attach_threshold": _resolve(
            ThemeExtractionService.FUZZY_ATTACH_THRESHOLD_PIPELINE_OVERRIDES,
            ThemeExtractionService.FUZZY_ATTACH_THRESHOLD_DEFAULT,
        ),
        "fuzzy_review_threshold": _resolve(
            ThemeExtractionService.FUZZY_REVIEW_THRESHOLD_PIPELINE_OVERRIDES,
            ThemeExtractionService.FUZZY_REVIEW_THRESHOLD_DEFAULT,
        ),
        "fuzzy_ambiguity_margin": _resolve(
            ThemeExtractionService.FUZZY_AMBIGUITY_MARGIN_PIPELINE_OVERRIDES,
            ThemeExtractionService.FUZZY_AMBIGUITY_MARGIN_DEFAULT,
        ),
        "embedding_attach_threshold": _resolve(
            ThemeExtractionService.EMBEDDING_ATTACH_THRESHOLD_PIPELINE_OVERRIDES,
            ThemeExtractionService.EMBEDDING_ATTACH_THRESHOLD_DEFAULT,
        ),
        "embedding_review_threshold": _resolve(
            ThemeExtractionService.EMBEDDING_REVIEW_THRESHOLD_PIPELINE_OVERRIDES,
            ThemeExtractionService.EMBEDDING_REVIEW_THRESHOLD_DEFAULT,
        ),
        "embedding_ambiguity_margin": _resolve(
            ThemeExtractionService.EMBEDDING_AMBIGUITY_MARGIN_PIPELINE_OVERRIDES,
            ThemeExtractionService.EMBEDDING_AMBIGUITY_MARGIN_DEFAULT,
        ),
    }
    lifecycle_defaults = {
        "promotion_min_mentions_7d": int(cfg.promotion_min_mentions_7d),
        "promotion_min_source_diversity_7d": int(cfg.promotion_min_source_diversity_7d),
        "promotion_min_avg_confidence_30d": float(cfg.promotion_min_avg_confidence_30d),
        "promotion_min_persistence_days": int(cfg.promotion_min_persistence_days),
        "dormancy_inactivity_days": int(cfg.dormancy_inactivity_days),
        "dormancy_min_mentions_30d": int(cfg.dormancy_min_mentions_30d),
        "dormancy_min_silence_days": int(cfg.dormancy_min_silence_days),
        "reactivation_min_mentions_7d": int(cfg.reactivation_min_mentions_7d),
        "reactivation_min_source_diversity_7d": int(cfg.reactivation_min_source_diversity_7d),
        "reactivation_min_avg_confidence_30d": float(cfg.reactivation_min_avg_confidence_30d),
        "relationship_subset_overlap_ratio": float(cfg.relationship_subset_overlap_ratio),
        "relationship_related_jaccard_threshold": float(cfg.relationship_related_jaccard_threshold),
        "relationship_min_overlap_constituents": int(cfg.relationship_min_overlap_constituents),
    }
    return {"matcher": matcher_defaults, "lifecycle": lifecycle_defaults}


def _clean_override_payload(request: ThemePolicyUpdateRequest) -> dict:
    matcher = request.matcher.model_dump(exclude_none=True) if request.matcher else {}
    lifecycle = request.lifecycle.model_dump(exclude_none=True) if request.lifecycle else {}
    return {"matcher": matcher, "lifecycle": lifecycle}


def _sanitize_override_sections(payload: dict) -> dict:
    matcher_payload = payload.get("matcher", {}) if isinstance(payload, dict) else {}
    lifecycle_payload = payload.get("lifecycle", {}) if isinstance(payload, dict) else {}
    matcher_allowed = set(ThemePolicyMatcherConfig.model_fields.keys())
    lifecycle_allowed = set(ThemePolicyLifecycleConfig.model_fields.keys())
    matcher = {
        key: value
        for key, value in (matcher_payload.items() if isinstance(matcher_payload, dict) else [])
        if key in matcher_allowed
    }
    lifecycle = {
        key: value
        for key, value in (lifecycle_payload.items() if isinstance(lifecycle_payload, dict) else [])
        if key in lifecycle_allowed
    }
    return {"matcher": matcher, "lifecycle": lifecycle}


def _diff_keys(before: dict, after: dict) -> list[str]:
    changed = []
    for section in ("matcher", "lifecycle"):
        before_section = before.get(section, {})
        after_section = after.get(section, {})
        section_keys = sorted(set(before_section.keys()) | set(after_section.keys()))
        for key in section_keys:
            if before_section.get(key) != after_section.get(key):
                changed.append(f"{section}.{key}")
    return changed


def _effective_policy(defaults: dict, overrides: dict) -> dict:
    output = {"matcher": dict(defaults.get("matcher", {})), "lifecycle": dict(defaults.get("lifecycle", {}))}
    for section in ("matcher", "lifecycle"):
        section_overrides = overrides.get(section, {}) if isinstance(overrides, dict) else {}
        for key, value in section_overrides.items():
            output[section][key] = value
    if output["matcher"].get("fuzzy_review_threshold", 0) > output["matcher"].get("fuzzy_attach_threshold", 1):
        output["matcher"]["fuzzy_review_threshold"] = output["matcher"]["fuzzy_attach_threshold"]
    if output["matcher"].get("embedding_review_threshold", 0) > output["matcher"].get("embedding_attach_threshold", 1):
        output["matcher"]["embedding_review_threshold"] = output["matcher"]["embedding_attach_threshold"]
    return output


def _history_row(entry: dict) -> ThemePolicyVersionSummary:
    return ThemePolicyVersionSummary(
        version_id=str(entry.get("version_id") or ""),
        pipeline=str(entry.get("pipeline") or ""),
        updated_at=str(entry.get("updated_at") or ""),
        updated_by=str(entry.get("updated_by") or "admin"),
        note=entry.get("note"),
    )


async def check_ollama_status(api_base: str) -> str:
    """Check if Ollama is reachable and return status."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{api_base}/api/tags")
            if response.status_code == 200:
                return "connected"
            return "error"
    except httpx.ConnectError:
        return "disconnected"
    except httpx.TimeoutException:
        return "timeout"
    except Exception as e:
        logger.warning(f"Ollama check failed: {e}")
        return "error"


@router.get("/config/llm", response_model=LLMConfigResponse)
async def get_llm_config(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_admin),
):
    """
    Get current LLM configuration.

    Returns:
        - Current extraction model
        - Current merge model
        - Ollama connection status
        - Available models list
    """
    # Get current settings
    extraction_model_id = get_setting(db, "llm_extraction_model", "groq/llama-3.3-70b-versatile")
    merge_model_id = get_setting(db, "llm_merge_model", "groq/llama-3.3-70b-versatile")
    ollama_api_base = get_setting(db, "ollama_api_base", "http://localhost:11434")

    # Also check environment variable override
    env_ollama_base = os.environ.get("OLLAMA_API_BASE")
    if env_ollama_base:
        ollama_api_base = env_ollama_base

    # Get model info
    extraction_model_info = get_model_by_id(extraction_model_id)
    merge_model_info = get_model_by_id(merge_model_id)

    # Check Ollama status
    ollama_status = await check_ollama_status(ollama_api_base)

    return LLMConfigResponse(
        extraction={
            "current_model": extraction_model_id,
            "model_info": extraction_model_info or {"id": extraction_model_id, "name": extraction_model_id, "provider": "unknown"},
        },
        merge={
            "current_model": merge_model_id,
            "model_info": merge_model_info or {"id": merge_model_id, "name": merge_model_id, "provider": "unknown"},
        },
        ollama_status=ollama_status,
        ollama_api_base=ollama_api_base,
        available_models=AVAILABLE_MODELS,
    )


@router.post("/config/llm")
async def update_llm_model(
    request: LLMModelUpdate,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_admin),
):
    """
    Update LLM model selection.

    Args:
        request: Model ID and use case (extraction or merge)

    Returns:
        Updated configuration
    """
    # Validate model exists
    model_info = get_model_by_id(request.model_id)
    if not model_info:
        # Allow custom model IDs but warn
        logger.warning(f"Unknown model ID: {request.model_id}")

    # Update setting based on use case
    if request.use_case == "extraction":
        set_setting(db, "llm_extraction_model", request.model_id, "llm", "Model used for theme extraction")
    elif request.use_case == "merge":
        set_setting(db, "llm_merge_model", request.model_id, "llm", "Model used for theme merge verification")
    else:
        raise HTTPException(status_code=400, detail=f"Invalid use_case: {request.use_case}")

    return {
        "status": "success",
        "use_case": request.use_case,
        "model_id": request.model_id,
        "model_info": model_info,
    }


@router.post("/config/ollama")
async def update_ollama_settings(
    request: OllamaSettings,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_admin),
):
    """
    Update Ollama API base URL.

    Args:
        request: Ollama API base URL

    Returns:
        Updated configuration and connection status
    """
    # Validate URL format
    api_base = request.api_base.rstrip("/")
    # Basic URL validation
    if not api_base.startswith("http://") and not api_base.startswith("https://"):
        raise HTTPException(status_code=400, detail="Ollama API base must be http:// or https://")

    # Test connection
    status = await check_ollama_status(api_base)

    # Save setting
    set_setting(db, "ollama_api_base", api_base, "llm", "Ollama API base URL")

    # Also set environment variable for LiteLLM
    os.environ["OLLAMA_API_BASE"] = api_base

    return {
        "status": "success",
        "ollama_api_base": api_base,
        "ollama_status": status,
    }


@router.get("/config/ollama/models")
async def get_ollama_models(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_admin),
):
    """
    Get list of available Ollama models.

    Returns:
        List of models installed in Ollama
    """
    ollama_api_base = get_setting(db, "ollama_api_base", "http://localhost:11434")
    env_ollama_base = os.environ.get("OLLAMA_API_BASE")
    if env_ollama_base:
        ollama_api_base = env_ollama_base

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_api_base}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return {
                    "status": "connected",
                    "models": [
                        {
                            "id": f"ollama_chat/{m.get('name', '')}",
                            "name": m.get("name", ""),
                            "size": m.get("size", 0),
                            "modified_at": m.get("modified_at", ""),
                        }
                        for m in models
                    ],
                }
            return {"status": "error", "models": [], "message": "Failed to fetch models"}
    except httpx.ConnectError:
        return {"status": "disconnected", "models": [], "message": "Cannot connect to Ollama"}
    except httpx.TimeoutException:
        return {"status": "timeout", "models": [], "message": "Connection timed out"}
    except Exception as e:
        logger.warning(f"Failed to get Ollama models: {e}")
        return {"status": "error", "models": [], "message": str(e)}


@router.get("/config/theme-policies", response_model=ThemePolicyConfigResponse)
async def get_theme_policy_config(
    pipeline: str,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_admin),
):
    if pipeline not in {"technical", "fundamental"}:
        raise HTTPException(status_code=400, detail="pipeline must be technical or fundamental")

    defaults = _theme_policy_defaults(pipeline)
    overrides_all = _get_setting_json(db, "theme_policy_overrides", {})
    staged_all = _get_setting_json(db, "theme_policy_staged", {})
    history_all = _get_setting_json(db, "theme_policy_history", [])

    pipeline_override = overrides_all.get(pipeline, {}) if isinstance(overrides_all, dict) else {}
    effective = _effective_policy(defaults, pipeline_override)
    metadata = pipeline_override.get("metadata", {}) if isinstance(pipeline_override, dict) else {}
    staged = staged_all.get(pipeline) if isinstance(staged_all, dict) else None
    history = [_history_row(row) for row in history_all if row.get("pipeline") == pipeline][:25]

    return ThemePolicyConfigResponse(
        pipeline=pipeline,
        active_version_id=metadata.get("version_id"),
        active_updated_at=metadata.get("updated_at"),
        active_updated_by=metadata.get("updated_by"),
        defaults=defaults,
        overrides={k: v for k, v in pipeline_override.items() if k in {"matcher", "lifecycle"}},
        effective=effective,
        staged=staged,
        history=history,
    )


@router.post("/config/theme-policies", response_model=ThemePolicyUpdateResponse)
async def update_theme_policy(
    request: ThemePolicyUpdateRequest,
    x_admin_actor: str = Header(default="admin", alias="X-Admin-Actor"),
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_admin),
):
    defaults = _theme_policy_defaults(request.pipeline)
    clean_payload = _clean_override_payload(request)
    overrides_all = _get_setting_json(db, "theme_policy_overrides", {})
    pipeline_override = overrides_all.get(request.pipeline, {}) if isinstance(overrides_all, dict) else {}
    base_override = {
        "matcher": dict(pipeline_override.get("matcher", {})) if isinstance(pipeline_override, dict) else {},
        "lifecycle": dict(pipeline_override.get("lifecycle", {})) if isinstance(pipeline_override, dict) else {},
    }
    proposed_override = deepcopy(base_override)
    for section in ("matcher", "lifecycle"):
        for key, value in clean_payload[section].items():
            proposed_override[section][key] = value

    current_effective = _effective_policy(defaults, base_override)
    preview_effective = _effective_policy(defaults, proposed_override)
    diff_keys = _diff_keys(current_effective, preview_effective)

    if request.mode == "preview":
        return ThemePolicyUpdateResponse(
            status="preview",
            pipeline=request.pipeline,
            mode=request.mode,
            preview_effective=preview_effective,
            diff_keys=diff_keys,
        )

    version_id = f"tp-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
    updated_at = datetime.utcnow().isoformat()
    metadata = {
        "version_id": version_id,
        "updated_at": updated_at,
        "updated_by": x_admin_actor or "admin",
        "note": request.note,
    }

    if request.mode == "stage":
        staged_all = _get_setting_json(db, "theme_policy_staged", {})
        staged_all[request.pipeline] = {
            "version_id": version_id,
            "pipeline": request.pipeline,
            "updated_at": updated_at,
            "updated_by": x_admin_actor or "admin",
            "note": request.note,
            "overrides": proposed_override,
            "effective": preview_effective,
            "diff_keys": diff_keys,
        }
        _set_setting_json(
            db,
            "theme_policy_staged",
            staged_all,
            category="theme",
            description="Staged theme policy override candidates",
        )
        return ThemePolicyUpdateResponse(
            status="staged",
            pipeline=request.pipeline,
            mode=request.mode,
            version_id=version_id,
            preview_effective=preview_effective,
            diff_keys=diff_keys,
        )

    if not isinstance(overrides_all, dict):
        overrides_all = {}
    overrides_all[request.pipeline] = {
        "matcher": proposed_override["matcher"],
        "lifecycle": proposed_override["lifecycle"],
        "metadata": metadata,
    }
    _set_setting_json(
        db,
        "theme_policy_overrides",
        overrides_all,
        category="theme",
        description="Active theme policy overrides by pipeline",
    )

    history_all = _get_setting_json(db, "theme_policy_history", [])
    if not isinstance(history_all, list):
        history_all = []
    history_all.insert(
        0,
        {
            "version_id": version_id,
            "pipeline": request.pipeline,
            "updated_at": updated_at,
            "updated_by": x_admin_actor or "admin",
            "note": request.note,
            "previous": base_override,
            "next": proposed_override,
            "diff_keys": diff_keys,
        },
    )
    history_all = history_all[:100]
    _set_setting_json(
        db,
        "theme_policy_history",
        history_all,
        category="theme",
        description="Theme policy audit log with version history",
    )

    staged_all = _get_setting_json(db, "theme_policy_staged", {})
    if isinstance(staged_all, dict) and request.pipeline in staged_all:
        staged_all.pop(request.pipeline, None)
        _set_setting_json(
            db,
            "theme_policy_staged",
            staged_all,
            category="theme",
            description="Staged theme policy override candidates",
        )

    return ThemePolicyUpdateResponse(
        status="applied",
        pipeline=request.pipeline,
        mode=request.mode,
        version_id=version_id,
        preview_effective=preview_effective,
        diff_keys=diff_keys,
    )


@router.post("/config/theme-policies/promote-staged", response_model=ThemePolicyUpdateResponse)
async def promote_staged_theme_policy(
    pipeline: str,
    note: str | None = None,
    x_admin_actor: str = Header(default="admin", alias="X-Admin-Actor"),
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_admin),
):
    staged_all = _get_setting_json(db, "theme_policy_staged", {})
    staged = staged_all.get(pipeline) if isinstance(staged_all, dict) else None
    if not staged:
        raise HTTPException(status_code=404, detail="No staged theme policy for pipeline")

    sanitized = _sanitize_override_sections(staged.get("overrides", {}))
    request = ThemePolicyUpdateRequest(
        pipeline=pipeline,
        matcher=sanitized.get("matcher", {}),
        lifecycle=sanitized.get("lifecycle", {}),
        note=note or staged.get("note"),
        mode="apply",
    )
    return await update_theme_policy(request=request, x_admin_actor=x_admin_actor, db=db, _auth=True)


@router.post("/config/theme-policies/revert", response_model=ThemePolicyUpdateResponse)
async def revert_theme_policy(
    request: ThemePolicyRevertRequest,
    x_admin_actor: str = Header(default="admin", alias="X-Admin-Actor"),
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_admin),
):
    history_all = _get_setting_json(db, "theme_policy_history", [])
    target = next(
        (
            row
            for row in history_all
            if row.get("pipeline") == request.pipeline and row.get("version_id") == request.version_id
        ),
        None,
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Version not found")

    selected_snapshot = target.get("next", {})
    sanitized = _sanitize_override_sections(selected_snapshot)
    update_request = ThemePolicyUpdateRequest(
        pipeline=request.pipeline,
        matcher=sanitized.get("matcher", {}),
        lifecycle=sanitized.get("lifecycle", {}),
        note=request.note or f"Reverted from {request.version_id}",
        mode="apply",
    )
    return await update_theme_policy(request=update_request, x_admin_actor=x_admin_actor, db=db, _auth=True)
