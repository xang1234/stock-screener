"""Stable ContentItem identity shared by legacy and Social X ingestion."""

from hashlib import md5


def twitter_external_id(provider_post_id: str) -> str:
    return md5(f"twitter:{provider_post_id}".encode("utf-8")).hexdigest()


__all__ = ["twitter_external_id"]
