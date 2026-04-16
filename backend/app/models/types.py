"""Shared SQLAlchemy column types for backend models."""

from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB


JsonColumn = JSON().with_variant(JSONB(), "postgresql")
