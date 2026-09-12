"""Validated records produced while preparing evidence for review."""

from enum import Enum

from pydantic import Field, model_validator

from .records import Record


class ImageType(str, Enum):
    CHART = "chart"
    TABLE = "table"
    DOCUMENT = "document"
    PHOTO = "photo"
    OTHER = "other"


class ImageObservation(Record):
    transcription: str
    observations: list[str] = Field(default_factory=list)
    image_type: ImageType
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_observed_content_or_uncertainty(self) -> "ImageObservation":
        values = [self.transcription, *self.observations, *self.uncertainties]
        if not any(value.strip() for value in values):
            raise ValueError("image_observation_empty")
        return self
