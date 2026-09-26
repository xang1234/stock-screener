"""Deterministic structural preparation of retained originals.

HTML: executable/embedded content is dropped; headings become a section
path; paragraphs, list items and tables become blocks. Tables keep their
header row, caption, detected unit/period labels and adjacent footnotes, so
a cell is never cited without its context.

Text PDFs: extracted page by page in a resource-limited subprocess (see
``pdf_extract.py``); physical page index and printed page label are kept
separately; pages beyond the limit are recorded as omitted. Table structure
recovered from PDF text layout is marked unverified rather than guessed.

Every block has exact offsets into one canonical document text, so a
selected passage can always be reconstructed from the retained original
plus this extractor version. No model is involved here.
"""

from __future__ import annotations

import json
import re
import resource
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import content_hash
from app.models.company_exposure import ExposureDocumentRevision, ExposurePassage
from app.services.theme_evaluation.multilingual_v2 import assess_language

PREPARATION_POLICY = "structure-v1"
HTML_EXTRACTOR = "html-bs4-lxml-v1"
PDF_EXTRACTOR = "pdf-pypdf-6.19.0-v1"
_PDF_SCRIPT = Path(__file__).with_name("pdf_extract.py")

_DROP_TAGS = (
    "script", "style", "noscript", "iframe", "object", "embed", "form",
    "template", "svg", "canvas", "button", "input", "select", "textarea",
)
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_TEXT_BLOCKS = {"p", "li", "blockquote", "pre", "dd", "dt", "figcaption"}
_UNIT_PATTERN = re.compile(
    r"(in\s+(?:thousands|millions|billions)[^)\n]{0,40}"
    r"|\((?:[A-Z]{3}|US\$|NT\$|HK\$|¥|\$)?\s*(?:thousand|million|billion)s?\)"
    r"|百万円|千円|億円|千元|萬元|万元|億元|亿元|單位[:：][^\s]{1,12}|单位[:：][^\s]{1,12})",
    re.IGNORECASE,
)
_PERIOD_PATTERN = re.compile(
    r"\b(?:FY|Q[1-4]\s*)?(?:19|20)\d{2}(?:/\d{2})?\b|(?:19|20)\d{2}年(?:度)?|民國\s*\d{2,3}年"
)
_FOOTNOTE_PATTERN = re.compile(r"^\s*(?:\(\d{1,2}\)|\[\d{1,2}\]|\*{1,3}|†|‡|注[\d０-９]?|Note\s+\d)")
_PDF_HEADING = re.compile(
    r"^(?:item\s+\d+[a-z]?\.|part\s+[ivx]+|第[一二三四五六七八九十\d]+[章節节部])", re.IGNORECASE
)
_SPEAKER = re.compile(r"^(?P<speaker>[A-Z][^:\n]{1,78}):\s")


class PreparationFailed(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class PreparationLimits:
    max_pages: int = 300
    max_passages: int = 24
    max_passage_chars: int = 4000
    pdf_timeout_seconds: float = 60.0
    pdf_memory_bytes: int = 768 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class QuestionSet:
    """What research is looking for; lexical terms incl. original-script names."""

    terms: tuple[str, ...]
    required_any: tuple[str, ...] = ()

    @property
    def hash(self) -> str:
        return content_hash({"terms": list(self.terms), "required": list(self.required_any)})


@dataclass(frozen=True, slots=True)
class PreparedBlock:
    ordinal: int
    kind: str
    text: str
    start: int
    end: int
    section_path: tuple[str, ...]
    page_index: int | None = None
    page_label: str | None = None
    table: dict | None = None
    speaker: str | None = None
    structure_verified: bool = True


@dataclass(frozen=True, slots=True)
class PreparedEvidence:
    revision_id: object
    revision_hash: str
    media_type: str
    extractor_version: str
    blocks: tuple[PreparedBlock, ...]
    document_text: str
    coverage: dict = field(default_factory=dict)

    @property
    def omitted_ranges(self) -> list:
        return list(self.coverage.get("omitted_ranges", []))

    @property
    def preparation_hash(self) -> str:
        return content_hash(
            {
                "revision": self.revision_hash,
                "policy": PREPARATION_POLICY,
                "extractor": self.extractor_version,
            }
        )

    def locator(self, block: PreparedBlock) -> dict:
        return {
            "revision_hash": self.revision_hash,
            "preparation_policy": PREPARATION_POLICY,
            "extractor_version": self.extractor_version,
            "page_index": block.page_index,
            "page_label": block.page_label,
            "section_path": list(block.section_path),
            "start": block.start,
            "end": block.end,
            "block_kind": block.kind,
            "structure_verified": block.structure_verified,
            "table": block.table,
        }


@dataclass(frozen=True, slots=True)
class PassageSelection:
    blocks: tuple[PreparedBlock, ...]
    omitted_matches: int
    policy: str = "lexical-v1"


def _clean(text: str) -> str:
    return re.sub(r"[ \t ]+", " ", re.sub(r"\s*\n\s*", "\n", text)).strip()


def _table_payload(table: Tag) -> dict:
    rows = []
    header: list[str] | None = None
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        values = [_clean(cell.get_text(" ")) for cell in cells]
        if not any(values):
            continue
        if header is None and (row.find("th") is not None or not rows):
            header = values
            if row.find("th") is not None:
                continue
        rows.append(values)
    caption_tag = table.find("caption")
    caption = _clean(caption_tag.get_text(" ")) if caption_tag else None
    joined = " ".join(filter(None, [caption or "", " ".join(header or [])]))
    return {
        "header": header or [],
        "rows": rows,
        "caption": caption,
        "units": sorted(set(_UNIT_PATTERN.findall(joined))),
        "periods": sorted(set(_PERIOD_PATTERN.findall(" ".join(header or [])))),
        "footnotes": [],
    }


def _render_table(payload: dict) -> str:
    lines = []
    if payload.get("caption"):
        lines.append(payload["caption"])
    if payload.get("header"):
        lines.append(" | ".join(payload["header"]))
    lines.extend(" | ".join(row) for row in payload["rows"])
    for note in payload.get("footnotes", []):
        lines.append(note)
    return "\n".join(lines)


def _html_blocks(data: bytes) -> list[dict]:
    soup = BeautifulSoup(data, "lxml")
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    root = soup.body or soup
    blocks: list[dict] = []
    path: list[tuple[int, str]] = []

    def section() -> tuple[str, ...]:
        return tuple(title for _, title in path)

    def visit(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                text = _clean(str(child))
                if text and node is root:
                    blocks.append({"kind": "paragraph", "text": text, "section": section()})
                continue
            if not isinstance(child, Tag):
                continue
            name = child.name.lower()
            if name in _HEADINGS:
                title = _clean(child.get_text(" "))
                if title:
                    level = _HEADINGS[name]
                    while path and path[-1][0] >= level:
                        path.pop()
                    path.append((level, title))
                    blocks.append({"kind": "heading", "text": title, "section": section()})
                continue
            if name == "table":
                payload = _table_payload(child)
                if payload["rows"] or payload["header"]:
                    blocks.append({"kind": "table", "table": payload, "section": section()})
                continue
            if name in _TEXT_BLOCKS:
                text = _clean(child.get_text(" "))
                if text:
                    if blocks and blocks[-1]["kind"] == "table" and _FOOTNOTE_PATTERN.match(text):
                        blocks[-1]["table"]["footnotes"].append(text)
                        continue
                    blocks.append({"kind": "paragraph", "text": text, "section": section()})
                continue
            has_block_children = child.find(
                list(_HEADINGS) + ["table", *_TEXT_BLOCKS, "div", "section", "article"]
            )
            if has_block_children:
                visit(child)
            else:
                text = _clean(child.get_text(" "))
                if text:
                    blocks.append({"kind": "paragraph", "text": text, "section": section()})

    visit(root)
    return blocks


def _run_pdf_extractor(data: bytes, limits: PreparationLimits) -> dict:
    def restrict() -> None:
        resource.setrlimit(
            resource.RLIMIT_AS, (limits.pdf_memory_bytes, limits.pdf_memory_bytes)
        )
        cpu = int(limits.pdf_timeout_seconds) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))

    try:
        completed = subprocess.run(
            [sys.executable, str(_PDF_SCRIPT), json.dumps({"max_pages": limits.max_pages})],
            input=data,
            capture_output=True,
            timeout=limits.pdf_timeout_seconds,
            preexec_fn=restrict,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise PreparationFailed("pdf_extraction_timeout") from None
    if completed.returncode != 0:
        raise PreparationFailed("pdf_extraction_resource_limit")
    try:
        result = json.loads(completed.stdout)
    except ValueError:
        raise PreparationFailed("pdf_extraction_invalid_output") from None
    if not result.get("ok"):
        raise PreparationFailed(result.get("failure", "pdf_extraction_failed"))
    return result


def _pdf_blocks(result: dict, max_chars: int) -> list[dict]:
    blocks: list[dict] = []
    section: tuple[str, ...] = ()
    for page in result["pages"]:
        buffer: list[str] = []

        def flush():
            text = _clean("\n".join(buffer))
            if text:
                blocks.append(
                    {
                        "kind": "paragraph",
                        "text": text,
                        "section": section,
                        "page_index": page["index"],
                        "page_label": page["label"],
                        "structure_verified": not _looks_tabular(text),
                    }
                )
            buffer.clear()

        for line in page["text"].splitlines():
            stripped = line.strip()
            if not stripped:
                flush()
                continue
            if _PDF_HEADING.match(stripped) and len(stripped) <= 120:
                flush()
                section = (stripped,)
                blocks.append(
                    {
                        "kind": "heading",
                        "text": stripped,
                        "section": section,
                        "page_index": page["index"],
                        "page_label": page["label"],
                    }
                )
                continue
            if sum(len(item) for item in buffer) + len(stripped) > max_chars:
                flush()
            buffer.append(stripped)
        flush()
    return blocks


def _looks_tabular(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    numeric = sum(1 for line in lines if len(re.findall(r"\d[\d,.]*", line)) >= 3)
    return len(lines) >= 3 and numeric >= max(2, len(lines) // 2)


class ExposureEvidencePreparer:
    def __init__(self, session: Session, store, *, limits: PreparationLimits | None = None):
        self.session = session
        self.store = store
        self.limits = limits or PreparationLimits()

    def prepare(self, revision: ExposureDocumentRevision, questions=None, limits=None) -> PreparedEvidence:
        del questions
        limits = limits or self.limits
        data = self.store.read(revision.blob_key)
        coverage: dict = {}
        if revision.media_type == "application/pdf":
            result = _run_pdf_extractor(data, limits)
            raw_blocks = _pdf_blocks(result, limits.max_passage_chars)
            extractor = PDF_EXTRACTOR
            coverage = {
                "page_count": result["page_count"],
                "processed_pages": min(result["page_count"], limits.max_pages),
                "omitted_ranges": result["omitted_ranges"],
                "failed_pages": result["failed_pages"],
            }
        elif revision.media_type in {"text/html", "application/xml", "text/plain"}:
            if revision.media_type == "text/plain":
                raw_blocks = [
                    {"kind": "paragraph", "text": _clean(chunk), "section": ()}
                    for chunk in re.split(r"\n\s*\n", data.decode("utf-8", "replace"))
                    if _clean(chunk)
                ]
            else:
                raw_blocks = _html_blocks(data)
            extractor = HTML_EXTRACTOR
            coverage = {"page_count": None, "processed_pages": None, "omitted_ranges": []}
        else:
            raise PreparationFailed("unsupported_media_type")

        blocks: list[PreparedBlock] = []
        pieces: list[str] = []
        cursor = 0
        for ordinal, raw in enumerate(raw_blocks):
            text = raw["text"] if raw["kind"] != "table" else _render_table(raw["table"])
            if cursor:
                pieces.append("\n\n")
                cursor += 2
            start = cursor
            pieces.append(text)
            cursor += len(text)
            speaker = None
            if raw["kind"] == "paragraph":
                match = _SPEAKER.match(text)
                if match and len(match.group("speaker").split()) <= 8:
                    speaker = match.group("speaker").strip()
            blocks.append(
                PreparedBlock(
                    ordinal=ordinal,
                    kind=raw["kind"],
                    text=text,
                    start=start,
                    end=cursor,
                    section_path=tuple(raw.get("section", ())),
                    page_index=raw.get("page_index"),
                    page_label=raw.get("page_label"),
                    table=raw.get("table"),
                    speaker=speaker,
                    structure_verified=raw.get("structure_verified", True),
                )
            )
        document_text = "".join(pieces)
        coverage["block_count"] = len(blocks)
        return PreparedEvidence(
            revision_id=revision.id,
            revision_hash=revision.content_hash,
            media_type=revision.media_type,
            extractor_version=extractor,
            blocks=tuple(blocks),
            document_text=document_text,
            coverage=coverage,
        )


def select_passages(
    prepared: PreparedEvidence, questions: QuestionSet, limit: int = 24
) -> PassageSelection:
    """Deterministic lexical selection; tables keep their full context."""

    terms = [term.casefold() for term in questions.terms if term.strip()]
    required = [term.casefold() for term in questions.required_any if term.strip()]
    scored = []
    for block in prepared.blocks:
        if block.kind == "heading":
            continue
        haystack = " ".join([*block.section_path, block.text]).casefold()
        if required and not any(term in haystack for term in required):
            continue
        score = sum(haystack.count(term) for term in terms)
        if score == 0:
            continue
        if block.kind == "table":
            score += 1
        scored.append((score, block.ordinal, block))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted((item[2] for item in scored[:limit]), key=lambda b: b.ordinal)
    return PassageSelection(
        blocks=tuple(chosen), omitted_matches=max(0, len(scored) - limit)
    )


def persist_passages(
    session: Session, prepared: PreparedEvidence, selection: PassageSelection
) -> list[ExposurePassage]:
    """Store selected passages; the same locator is stored once."""

    stored = []
    by_ordinal = {block.ordinal: block for block in prepared.blocks}
    for block in selection.blocks:
        locator = prepared.locator(block)
        locator_hash = content_hash(locator)
        existing = session.execute(
            select(ExposurePassage).where(
                ExposurePassage.document_revision_id == prepared.revision_id,
                ExposurePassage.preparation_policy == PREPARATION_POLICY,
                ExposurePassage.locator_hash == locator_hash,
            )
        ).scalar_one_or_none()
        if existing is not None:
            stored.append(existing)
            continue
        previous = by_ordinal.get(block.ordinal - 1)
        following = by_ordinal.get(block.ordinal + 1)
        decision = assess_language(block.text)
        passage = ExposurePassage(
            document_revision_id=prepared.revision_id,
            revision_content_hash=prepared.revision_hash,
            preparation_policy=PREPARATION_POLICY,
            extractor_version=prepared.extractor_version,
            locator=locator,
            locator_hash=locator_hash,
            original_text=block.text,
            text_hash=content_hash({"text": block.text}),
            context={
                "section_path": list(block.section_path),
                "previous_text": None if previous is None else previous.text[:300],
                "next_text": None if following is None else following.text[:300],
                "speaker": block.speaker,
                "language_warnings": list(decision.warnings),
            },
            language=decision.source_language,
            page_index=block.page_index,
        )
        try:
            with session.begin_nested():
                session.add(passage)
                session.flush()
        except IntegrityError:
            passage = session.execute(
                select(ExposurePassage).where(
                    ExposurePassage.document_revision_id == prepared.revision_id,
                    ExposurePassage.preparation_policy == PREPARATION_POLICY,
                    ExposurePassage.locator_hash == locator_hash,
                )
            ).scalar_one()
        stored.append(passage)
    return stored
