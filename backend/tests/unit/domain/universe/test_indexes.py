from __future__ import annotations

import pytest

from app.domain.universe.indexes import IndexDefinition, IndexRegistry, index_registry


def test_index_registry_normalizes_aliases_and_exposes_market() -> None:
    assert index_registry.normalize("s&p 500") == "SP500"
    assert index_registry.normalize("nikkei 225") == "NIKKEI225"
    assert index_registry.market_for("sti") == "SG"
    assert index_registry.normalize("ASX 200") == "ASX200"
    assert index_registry.normalize("S&P ASX 200") == "ASX200"
    assert index_registry.normalize("XJO") == "ASX200"
    assert index_registry.normalize("AXJO") == "ASX200"
    assert index_registry.market_for("ASX200") == "AU"
    assert index_registry.get("ASX200").label == "S&P/ASX 200"


def test_index_registry_rejects_duplicate_aliases() -> None:
    with pytest.raises(ValueError, match="Duplicate index alias"):
        IndexRegistry(
            (
                IndexDefinition(key="AAA", label="AAA", market="US", aliases=("DUP",)),
                IndexDefinition(key="BBB", label="BBB", market="US", aliases=("dup",)),
            )
        )


def test_index_registry_rejects_unknown_market() -> None:
    with pytest.raises(ValueError, match="Unsupported index market"):
        IndexRegistry((IndexDefinition(key="FTSE100", label="FTSE 100", market="GB"),))
