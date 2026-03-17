"""
API endpoints for Filter Presets feature.
Handles CRUD operations for saved filter configurations.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
import json
import logging

from ...database import get_db
from ...models.filter_preset import FilterPreset
from ...schemas.filter_preset import (
    FilterPresetCreate, FilterPresetUpdate, FilterPresetResponse,
    FilterPresetListResponse, ReorderPresetsRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ================= Preset CRUD =================

@router.get("", response_model=FilterPresetListResponse, include_in_schema=False)
@router.get("/", response_model=FilterPresetListResponse)
async def list_presets(db: Session = Depends(get_db)):
    """Get all filter presets ordered by position."""
    presets = db.query(FilterPreset).order_by(FilterPreset.position).all()

    # Parse JSON filters for each preset
    preset_responses = []
    for preset in presets:
        try:
            filters_dict = json.loads(preset.filters) if isinstance(preset.filters, str) else preset.filters
        except json.JSONDecodeError:
            filters_dict = {}

        preset_responses.append(FilterPresetResponse(
            id=preset.id,
            name=preset.name,
            description=preset.description,
            filters=filters_dict,
            sort_by=preset.sort_by,
            sort_order=preset.sort_order,
            position=preset.position,
            created_at=preset.created_at,
            updated_at=preset.updated_at,
        ))

    return FilterPresetListResponse(
        presets=preset_responses,
        total=len(preset_responses)
    )


@router.post("", response_model=FilterPresetResponse, include_in_schema=False)
@router.post("/", response_model=FilterPresetResponse)
async def create_preset(data: FilterPresetCreate, db: Session = Depends(get_db)):
    """Create a new filter preset."""
    existing = db.query(FilterPreset).filter(FilterPreset.name == data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Preset with this name already exists")

    max_pos = db.query(func.max(FilterPreset.position)).scalar() or -1

    preset = FilterPreset(
        name=data.name,
        description=data.description,
        filters=json.dumps(data.filters),
        sort_by=data.sort_by,
        sort_order=data.sort_order,
        position=max_pos + 1
    )
    db.add(preset)
    db.commit()
    db.refresh(preset)

    return FilterPresetResponse(
        id=preset.id,
        name=preset.name,
        description=preset.description,
        filters=data.filters,
        sort_by=preset.sort_by,
        sort_order=preset.sort_order,
        position=preset.position,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
    )


@router.put("/{preset_id}", response_model=FilterPresetResponse)
async def update_preset(preset_id: int, updates: FilterPresetUpdate, db: Session = Depends(get_db)):
    """Update preset properties."""
    preset = db.query(FilterPreset).filter(FilterPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    # Check for duplicate name if name is being changed
    if updates.name is not None and updates.name != preset.name:
        existing = db.query(FilterPreset).filter(FilterPreset.name == updates.name).first()
        if existing:
            raise HTTPException(status_code=400, detail="Preset with this name already exists")
        preset.name = updates.name

    if updates.description is not None:
        preset.description = updates.description
    if updates.filters is not None:
        preset.filters = json.dumps(updates.filters)
    if updates.sort_by is not None:
        preset.sort_by = updates.sort_by
    if updates.sort_order is not None:
        preset.sort_order = updates.sort_order
    if updates.position is not None:
        preset.position = updates.position

    db.commit()
    db.refresh(preset)

    try:
        filters_dict = json.loads(preset.filters) if isinstance(preset.filters, str) else preset.filters
    except json.JSONDecodeError:
        filters_dict = {}

    return FilterPresetResponse(
        id=preset.id,
        name=preset.name,
        description=preset.description,
        filters=filters_dict,
        sort_by=preset.sort_by,
        sort_order=preset.sort_order,
        position=preset.position,
        created_at=preset.created_at,
        updated_at=preset.updated_at,
    )


@router.delete("/{preset_id}")
async def delete_preset(preset_id: int, db: Session = Depends(get_db)):
    """Delete a filter preset."""
    preset = db.query(FilterPreset).filter(FilterPreset.id == preset_id).first()
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")

    db.delete(preset)
    db.commit()
    return {"status": "deleted", "preset_id": preset_id}


@router.put("/reorder")
async def reorder_presets(
    reorder_data: ReorderPresetsRequest,
    db: Session = Depends(get_db)
):
    """Reorder presets by updating their position values."""
    for idx, preset_id in enumerate(reorder_data.preset_ids):
        preset = db.query(FilterPreset).filter(FilterPreset.id == preset_id).first()
        if preset:
            preset.position = idx
    db.commit()
    return {"status": "reordered"}
