import json
from pathlib import Path

import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.services.static_artifact_combiner import (
    StaticArtifactCombiner,
    StaticArtifactFormulaError,
)
from app.services.static_site_errors import NoPublishedStaticMarketArtifact
from app.services.static_site_export_service import (
    STATIC_DEFAULT_MARKET,
    STATIC_MARKET_METADATA_FILENAME,
    STATIC_SITE_SCHEMA_VERSION,
    STATIC_SUPPORTED_MARKETS,
)


def write_market_artifact(
    root: Path,
    *,
    market: str,
    formula: str,
    scan_formula: str | None = None,
    chunk_formula: str | None = None,
) -> Path:
    # actions/upload-artifact uploads the contents of the selected market
    # directory, so download-artifact restores manifest.market.json directly
    # beneath the artifact-name directory.
    market_dir = root / f"static-market-{market}"
    chunk_dir = market_dir / "scan" / "chunks"
    chunk_dir.mkdir(parents=True)
    chunk_path = f"markets/{market.lower()}/scan/chunks/chunk-0001.json"
    (chunk_dir / "chunk-0001.json").write_text(
        json.dumps(
            {
                "rs_formula_version": chunk_formula or formula,
                "rows": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (market_dir / "scan" / "manifest.json").write_text(
        json.dumps(
            {
                "rs_formula_version": scan_formula or formula,
                "chunks": [{"path": chunk_path, "count": 0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    entry = {
        "market": market,
        "display_name": market,
        "as_of_date": "2026-04-10",
        "rs_formula_version": formula,
        "features": {
            "scan": True,
            "breadth": False,
            "groups": False,
            "charts": False,
        },
        "pages": {
            "scan": {"path": f"markets/{market.lower()}/scan/manifest.json"}
        },
        "assets": {},
    }
    (market_dir / STATIC_MARKET_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": STATIC_SITE_SCHEMA_VERSION,
                "generated_at": "2026-04-10T22:00:00Z",
                "market": market,
                "entry": entry,
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    return root


def combiner() -> StaticArtifactCombiner:
    return StaticArtifactCombiner(
        schema_version=STATIC_SITE_SCHEMA_VERSION,
        supported_markets=STATIC_SUPPORTED_MARKETS,
        default_market=STATIC_DEFAULT_MARKET,
    )


def test_combiner_accepts_downloaded_market_artifact_layout(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
    )

    result = combiner().combine(
        artifacts_dir=current,
        fallback_artifacts_dir=None,
        output_dir=tmp_path / "out",
        required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
        clean=True,
    )

    assert result.manifest["markets"]["US"]["rs_formula_version"] == (
        BALANCED_RS_FORMULA_VERSION
    )
    assert (
        tmp_path / "out" / "markets" / "us" / "scan" / "chunks" / "chunk-0001.json"
    ).is_file()


def test_combiner_rejects_wrong_formula_current_without_using_fallback(tmp_path):
    current = write_market_artifact(
        tmp_path / "current", market="US", formula=LEGACY_RS_FORMULA_VERSION
    )
    fallback = write_market_artifact(
        tmp_path / "fallback", market="US", formula=BALANCED_RS_FORMULA_VERSION
    )
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("last-good", encoding="utf-8")
    with pytest.raises(StaticArtifactFormulaError, match="US current"):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=fallback,
            output_dir=output,
            required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "last-good"


def test_combiner_rejects_wrong_formula_fallback(tmp_path):
    fallback = write_market_artifact(
        tmp_path / "fallback", market="HK", formula=LEGACY_RS_FORMULA_VERSION
    )
    with pytest.raises(StaticArtifactFormulaError, match="HK fallback"):
        combiner().combine(
            artifacts_dir=tmp_path / "empty-current",
            fallback_artifacts_dir=fallback,
            output_dir=tmp_path / "out",
            required_formula_by_market={"HK": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def test_combiner_rejects_swapped_artifact_name_and_manifest_market(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="AU",
        formula=BALANCED_RS_FORMULA_VERSION,
    )
    artifact_dir = current / "static-market-AU"
    manifest_path = artifact_dir / STATIC_MARKET_METADATA_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["market"] = "US"
    manifest["entry"]["market"] = "US"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="market 'US'; expected 'AU'"):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={"AU": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def test_combiner_allows_legacy_fallback_outside_explicit_formula_policy(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
    )
    fallback = write_market_artifact(
        tmp_path / "fallback",
        market="HK",
        formula=LEGACY_RS_FORMULA_VERSION,
    )

    result = combiner().combine(
        artifacts_dir=current,
        fallback_artifacts_dir=fallback,
        output_dir=tmp_path / "out",
        required_formula_by_market={
            "US": BALANCED_RS_FORMULA_VERSION,
            "HK": BALANCED_RS_FORMULA_VERSION,
        },
        fallback_required_formula_by_market={},
        clean=True,
    )

    assert result.manifest["markets"]["US"]["rs_formula_version"] == (
        BALANCED_RS_FORMULA_VERSION
    )
    assert result.manifest["markets"]["HK"]["rs_formula_version"] == (
        LEGACY_RS_FORMULA_VERSION
    )


def test_combiner_omits_optional_market_without_current_or_fallback_artifact(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
    )

    result = combiner().combine(
        artifacts_dir=current,
        fallback_artifacts_dir=tmp_path / "empty-fallback",
        output_dir=tmp_path / "out",
        required_formula_by_market={
            "US": BALANCED_RS_FORMULA_VERSION,
            "CN": BALANCED_RS_FORMULA_VERSION,
        },
        optional_markets={"CN"},
        clean=True,
    )

    assert result.manifest["supported_markets"] == ["US"]
    assert "CN" not in result.manifest["markets"]
    assert not (tmp_path / "out" / "markets" / "cn").exists()
    assert any("CN was omitted" in warning for warning in result.manifest["warnings"])


def test_combiner_validates_optional_market_formula_when_artifact_exists(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="CN",
        formula=LEGACY_RS_FORMULA_VERSION,
    )

    with pytest.raises(StaticArtifactFormulaError, match="CN current"):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={"CN": BALANCED_RS_FORMULA_VERSION},
            optional_markets={"CN"},
            clean=True,
        )


def test_combiner_rejects_wrong_scan_manifest_formula(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
        scan_formula=LEGACY_RS_FORMULA_VERSION,
    )

    with pytest.raises(
        StaticArtifactFormulaError,
        match="Scan manifest='legacy-linear-v1'",
    ):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def test_combiner_rejects_wrong_scan_chunk_formula(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
        chunk_formula=LEGACY_RS_FORMULA_VERSION,
    )

    with pytest.raises(
        StaticArtifactFormulaError,
        match="Scan chunk chunk-0001.json='legacy-linear-v1'",
    ):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def test_combiner_requires_every_market_named_by_formula_map(tmp_path):
    current = write_market_artifact(
        tmp_path / "current", market="US", formula=BALANCED_RS_FORMULA_VERSION
    )
    with pytest.raises(NoPublishedStaticMarketArtifact) as exc_info:
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={
                "US": BALANCED_RS_FORMULA_VERSION,
                "HK": BALANCED_RS_FORMULA_VERSION,
            },
            clean=True,
        )
    assert exc_info.value.markets == ("HK",)
