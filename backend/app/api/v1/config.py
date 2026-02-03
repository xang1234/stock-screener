"""API endpoints for application configuration including LLM settings."""
import logging
import os
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.app_settings import AppSetting
from ...services.llm.config import AVAILABLE_MODELS, get_model_by_id
from ...config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class LLMModelUpdate(BaseModel):
    """Request body for updating LLM model selection."""
    model_id: str
    use_case: str = "extraction"  # "extraction" or "merge"


class OllamaSettings(BaseModel):
    """Request body for updating Ollama settings."""
    api_base: str


class LLMConfigResponse(BaseModel):
    """Response for LLM configuration."""
    extraction: dict
    merge: dict
    ollama_status: str
    ollama_api_base: str
    available_models: list


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
