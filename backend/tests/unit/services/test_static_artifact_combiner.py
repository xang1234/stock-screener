import json
from datetime import date
from pathlib import Path

import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.services.static_artifact_combiner import (
    StaticArtifactCombiner,
    StaticArtifactFormulaError,
    annotate_publication_lag,
)
from app.services.static_options_section import StaticOptionsSection
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
    breadth_revision: int | None = None,
    breadth_source_revision: str | None = None,
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
            "breadth": breadth_revision is not None,
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
    if breadth_revision is not None:
        (market_dir / "breadth.json").write_text(
            json.dumps(
                {
                    "schema_version": STATIC_SITE_SCHEMA_VERSION,
                    "available": True,
                    "source_revision": (
                        breadth_source_revision
                        or f"2026-04-10|breadth-r{breadth_revision}"
                    ),
                    "payload": {
                        "current": {
                            "market": market,
                            "date": "2026-04-10",
                            "calculation_revision": breadth_revision,
                        }
                    },
                }
            )
            + "\n",
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
    with pytest.raises(NoPublishedStaticMarketArtifact, match="HK"):
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


def test_combiner_removes_stale_optional_market_directory_on_incremental_publish(
    tmp_path,
):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
    )
    output_dir = tmp_path / "out"
    stale_cn_scan_dir = output_dir / "markets" / "cn" / "scan"
    stale_cn_scan_dir.mkdir(parents=True)
    (stale_cn_scan_dir / "manifest.json").write_text("{}", encoding="utf-8")

    result = combiner().combine(
        artifacts_dir=current,
        fallback_artifacts_dir=tmp_path / "empty-fallback",
        output_dir=output_dir,
        required_formula_by_market={
            "US": BALANCED_RS_FORMULA_VERSION,
            "CN": BALANCED_RS_FORMULA_VERSION,
        },
        optional_markets={"CN"},
        clean=False,
    )

    assert result.manifest["supported_markets"] == ["US"]
    assert (output_dir / "markets" / "us").is_dir()
    assert not (output_dir / "markets" / "cn").exists()


def test_combiner_surfaces_stale_optional_market_cleanup_errors(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
    )
    output_dir = tmp_path / "out"
    stale_cn_path = output_dir / "markets" / "cn"
    stale_cn_path.parent.mkdir(parents=True)
    stale_cn_path.write_text("stale", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=tmp_path / "empty-fallback",
            output_dir=output_dir,
            required_formula_by_market={
                "US": BALANCED_RS_FORMULA_VERSION,
                "CN": BALANCED_RS_FORMULA_VERSION,
            },
            optional_markets={"CN"},
            clean=False,
        )

    assert stale_cn_path.is_file()
    assert not (output_dir / "markets" / "us").exists()


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


def test_combiner_accepts_current_breadth_revision_three(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
        breadth_revision=3,
    )

    result = combiner().combine(
        artifacts_dir=current,
        fallback_artifacts_dir=None,
        output_dir=tmp_path / "out",
        required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
        clean=True,
    )

    assert result.manifest["markets"]["US"]["features"]["breadth"] is True


def test_combiner_rejects_current_breadth_revision_two(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
        breadth_revision=2,
    )

    with pytest.raises(StaticArtifactFormulaError, match="breadth revision"):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def test_combiner_filters_revision_two_breadth_fallback_without_rs_policy(tmp_path):
    fallback = write_market_artifact(
        tmp_path / "fallback",
        market="HK",
        formula=LEGACY_RS_FORMULA_VERSION,
        breadth_revision=2,
    )

    with pytest.raises(NoPublishedStaticMarketArtifact, match="HK"):
        combiner().combine(
            artifacts_dir=tmp_path / "empty-current",
            fallback_artifacts_dir=fallback,
            output_dir=tmp_path / "out",
            required_formula_by_market={"HK": BALANCED_RS_FORMULA_VERSION},
            fallback_required_formula_by_market={},
            clean=True,
        )


def test_combiner_rejects_revision_three_with_revision_two_source_marker(tmp_path):
    current = write_market_artifact(
        tmp_path / "current",
        market="US",
        formula=BALANCED_RS_FORMULA_VERSION,
        breadth_revision=3,
        breadth_source_revision="2026-04-10|breadth-r2",
    )

    with pytest.raises(StaticArtifactFormulaError, match="breadth revision"):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def _rewrite_entry(root: Path, market: str, **changes) -> Path:
    path = root / f"static-market-{market}" / STATIC_MARKET_METADATA_FILENAME
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["entry"].update(changes)
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path.parent


def _validate_assets(market_dir: Path) -> None:
    entry = json.loads(
        (market_dir / STATIC_MARKET_METADATA_FILENAME).read_text(encoding="utf-8")
    )["entry"]
    StaticArtifactCombiner._validate_advertised_assets(
        market="US", source_label="fallback", entry=entry, market_dir=market_dir
    )


@pytest.mark.parametrize(
    "pages, files, message",
    [
        ({"home": {"path": "markets/us/home.json"}}, {}, "absent.*home.json"),
        (
            {"home": {"path": "markets/us/home.json"}},
            {"home.json": "{not json"},
            "home.json.*parse",
        ),
        ({"home": {"path": "markets/us/../../escape.json"}}, {}, "escapes"),
    ],
)
def test_advertised_page_paths_must_exist_inside_root_and_parse(
    tmp_path: Path, pages, files, message
) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = _rewrite_entry(tmp_path, "US", pages=pages)
    # The escape target exists, so only the containment check can reject it.
    (tmp_path / "escape.json").write_text("{}", encoding="utf-8")
    for name, text in files.items():
        (market_dir / name).write_text(text, encoding="utf-8")

    with pytest.raises(StaticArtifactFormulaError, match=message):
        _validate_assets(market_dir)


def test_chart_index_symbol_paths_must_stay_inside_root(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    (market_dir / "charts").mkdir()
    (market_dir / "charts" / "index.json").write_text(
        json.dumps({"symbols": [{"symbol": "X", "path": "markets/us/../../x.json"}]}),
        encoding="utf-8",
    )
    (tmp_path / "x.json").write_text("{}", encoding="utf-8")
    _rewrite_entry(
        tmp_path, "US", assets={"charts": {"path": "markets/us/charts/index.json"}}
    )

    with pytest.raises(StaticArtifactFormulaError, match="escapes"):
        _validate_assets(market_dir)


def test_chart_index_symbol_payload_must_exist(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    (market_dir / "charts").mkdir()
    (market_dir / "charts" / "index.json").write_text(
        json.dumps({"symbols": [{"symbol": "X", "path": "markets/us/charts/X.json"}]}),
        encoding="utf-8",
    )
    _rewrite_entry(
        tmp_path, "US", assets={"charts": {"path": "markets/us/charts/index.json"}}
    )

    with pytest.raises(StaticArtifactFormulaError, match="X.json"):
        _validate_assets(market_dir)


def test_valid_advertised_pages_and_charts_pass(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    (market_dir / "home.json").write_text("{}", encoding="utf-8")
    (market_dir / "charts").mkdir()
    (market_dir / "charts" / "X.json").write_text("{}", encoding="utf-8")
    (market_dir / "charts" / "index.json").write_text(
        json.dumps({"symbols": [{"symbol": "X", "path": "markets/us/charts/X.json"}]}),
        encoding="utf-8",
    )
    _rewrite_entry(
        tmp_path,
        "US",
        pages={
            "home": {"path": "markets/us/home.json"},
            "scan": {"path": "markets/us/scan/manifest.json"},
        },
        assets={"charts": {"path": "markets/us/charts/index.json"}},
    )

    _validate_assets(market_dir)


def test_combined_manifest_lists_unavailable_markets_and_sources(tmp_path: Path) -> None:
    current = write_market_artifact(
        tmp_path / "current", market="US", formula=BALANCED_RS_FORMULA_VERSION
    )
    fallback = write_market_artifact(
        tmp_path / "fallback", market="HK", formula=BALANCED_RS_FORMULA_VERSION
    )
    output = tmp_path / "out"
    (output / "markets" / "in").mkdir(parents=True)  # stale tree from an older bundle

    result = combiner().combine(
        artifacts_dir=current,
        fallback_artifacts_dir=fallback,
        output_dir=output,
        required_formula_by_market={},
        optional_markets=[m for m in STATIC_SUPPORTED_MARKETS if m != "US"],
        clean=False,
    )

    manifest = result.manifest
    assert manifest["supported_markets"] == ["US", "HK"]
    assert manifest["unavailable_markets"] == [
        m for m in STATIC_SUPPORTED_MARKETS if m not in {"US", "HK"}
    ]
    assert manifest["markets"]["US"]["publication"] == {
        "source": "current",
        "session_date": "2026-04-10",
    }
    assert manifest["markets"]["HK"]["publication"]["source"] == "fallback"
    assert not (output / "markets" / "in").exists()


class _Calendar:
    def __init__(self, last, sessions, broken=()):
        self.last, self.sessions, self.broken = last, sessions, set(broken)

    def last_completed_trading_day(self, market):
        if market in self.broken:
            raise RuntimeError("calendar coverage expired")
        return self.last

    def trading_days(self, market, start, end):
        return [d for d in self.sessions if start <= d <= end]


def test_publication_lag_marks_current_stale_and_unknown() -> None:
    sessions = [date(2026, 4, 9), date(2026, 4, 10), date(2026, 4, 13)]
    manifest = {
        "markets": {
            "US": {"publication": {"source": "current", "session_date": "2026-04-13"}},
            "HK": {"publication": {"source": "fallback", "session_date": "2026-04-09"}},
            "JP": {"publication": {"source": "fallback", "session_date": "2026-04-10"}},
            "KR": {"publication": {"source": "fallback", "session_date": None}},
        }
    }

    annotate_publication_lag(
        manifest, _Calendar(date(2026, 4, 13), sessions, broken={"JP"})
    )

    assert manifest["markets"]["US"]["publication"] == {
        "source": "current",
        "session_date": "2026-04-13",
        "session_lag": 0,
        "state": "current",
    }
    assert manifest["markets"]["HK"]["publication"]["session_lag"] == 2
    assert manifest["markets"]["HK"]["publication"]["state"] == "stale"
    for market in ("JP", "KR"):
        assert manifest["markets"][market]["publication"]["session_lag"] is None
        assert manifest["markets"][market]["publication"]["state"] == "unknown"


def test_publication_lag_counts_a_session_newer_than_the_close_as_current() -> None:
    # A run during the session can serve today's partial bar; that is not
    # behind the last completed session.
    manifest = {
        "markets": {
            "US": {"publication": {"source": "current", "session_date": "2026-04-14"}}
        }
    }
    sessions = [date(2026, 4, 13), date(2026, 4, 14)]

    annotate_publication_lag(manifest, _Calendar(date(2026, 4, 13), sessions))

    assert manifest["markets"]["US"]["publication"]["session_lag"] == 0
    assert manifest["markets"]["US"]["publication"]["state"] == "current"


def test_root_level_options_descriptor_is_not_validated_as_a_market_file(
    tmp_path: Path,
) -> None:
    # The US export advertises the root-level options bundle, which ships as
    # its own artifact (static-options-US), never inside the market artifact.
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    entry = json.loads(
        (market_dir / STATIC_MARKET_METADATA_FILENAME).read_text(encoding="utf-8")
    )["entry"]
    StaticOptionsSection._advertise(entry)

    StaticArtifactCombiner._validate_advertised_assets(
        market="US", source_label="current", entry=entry, market_dir=market_dir
    )


def test_advertised_path_with_nul_byte_is_rejected_not_crashing(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = _rewrite_entry(
        tmp_path, "US", pages={"home": {"path": "markets/us/ho\x00me.json"}}
    )

    with pytest.raises(StaticArtifactFormulaError, match="path is invalid"):
        _validate_assets(market_dir)


def test_damaged_current_artifact_falls_back_instead_of_aborting(tmp_path: Path) -> None:
    current = tmp_path / "current"
    write_market_artifact(current, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    write_market_artifact(current, market="HK", formula=BALANCED_RS_FORMULA_VERSION)
    _rewrite_entry(current, "HK", pages={"home": {"path": "markets/hk/home.json"}})
    fallback = write_market_artifact(
        tmp_path / "fallback", market="HK", formula=BALANCED_RS_FORMULA_VERSION
    )

    result = combiner().combine(
        artifacts_dir=current,
        fallback_artifacts_dir=fallback,
        output_dir=tmp_path / "out",
        required_formula_by_market={},
        optional_markets=[m for m in STATIC_SUPPORTED_MARKETS if m != "US"],
        clean=True,
    )

    assert result.manifest["markets"]["HK"]["publication"]["source"] == "fallback"
    assert result.manifest["markets"]["US"]["publication"]["source"] == "current"
    assert any("HK" in w and "home.json" in w for w in result.warnings)


def test_damaged_required_current_without_fallback_names_the_defect(
    tmp_path: Path,
) -> None:
    current = write_market_artifact(
        tmp_path / "current", market="US", formula=BALANCED_RS_FORMULA_VERSION
    )
    _rewrite_entry(current, "US", pages={"home": {"path": "markets/us/home.json"}})

    with pytest.raises(NoPublishedStaticMarketArtifact, match="home.json"):
        combiner().combine(
            artifacts_dir=current,
            fallback_artifacts_dir=None,
            output_dir=tmp_path / "out",
            required_formula_by_market={"US": BALANCED_RS_FORMULA_VERSION},
            clean=True,
        )


def test_market_root_relative_paths_are_still_validated(tmp_path: Path) -> None:
    # Older artifacts advertise paths relative to the market root.
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = _rewrite_entry(tmp_path, "US", pages={"home": {"path": "home.json"}})

    with pytest.raises(StaticArtifactFormulaError, match="absent.*home.json"):
        _validate_assets(market_dir)


def test_market_root_relative_chart_payloads_are_still_validated(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    (market_dir / "charts").mkdir()
    (market_dir / "charts" / "index.json").write_text(
        json.dumps({"symbols": [{"symbol": "X", "path": "charts/X.json"}]}),
        encoding="utf-8",
    )
    _rewrite_entry(tmp_path, "US", assets={"charts": {"path": "charts/index.json"}})

    with pytest.raises(StaticArtifactFormulaError, match="X.json"):
        _validate_assets(market_dir)


def test_another_markets_path_is_rejected(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = _rewrite_entry(
        tmp_path, "US", pages={"home": {"path": "markets/hk/home.json"}}
    )

    with pytest.raises(StaticArtifactFormulaError, match="another market"):
        _validate_assets(market_dir)


@pytest.mark.parametrize("chunk_text", [None, "{truncated"])
def test_scan_chunks_are_validated_without_a_formula_override(
    tmp_path: Path, chunk_text
) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    chunk = tmp_path / "static-market-US" / "scan" / "chunks" / "chunk-0001.json"
    if chunk_text is None:
        chunk.unlink()
    else:
        chunk.write_text(chunk_text, encoding="utf-8")

    with pytest.raises(StaticArtifactFormulaError, match="chunk-0001.json"):
        _validate_assets(tmp_path / "static-market-US")


def test_malformed_chart_payload_is_rejected(tmp_path: Path) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    (market_dir / "charts").mkdir()
    (market_dir / "charts" / "X.json").write_text("{truncated", encoding="utf-8")
    (market_dir / "charts" / "index.json").write_text(
        json.dumps({"symbols": [{"symbol": "X", "path": "markets/us/charts/X.json"}]}),
        encoding="utf-8",
    )
    _rewrite_entry(
        tmp_path, "US", assets={"charts": {"path": "markets/us/charts/index.json"}}
    )

    with pytest.raises(StaticArtifactFormulaError, match="X.json.*does not parse"):
        _validate_assets(market_dir)


@pytest.mark.parametrize("index", [{"symbols": 1}, [], {"symbols": [None]}])
def test_malformed_chart_index_is_rejected_not_crashing(tmp_path: Path, index) -> None:
    write_market_artifact(tmp_path, market="US", formula=BALANCED_RS_FORMULA_VERSION)
    market_dir = tmp_path / "static-market-US"
    (market_dir / "charts").mkdir()
    (market_dir / "charts" / "index.json").write_text(json.dumps(index), encoding="utf-8")
    _rewrite_entry(
        tmp_path, "US", assets={"charts": {"path": "markets/us/charts/index.json"}}
    )

    with pytest.raises(StaticArtifactFormulaError, match="chart index"):
        _validate_assets(market_dir)
