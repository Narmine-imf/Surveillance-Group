"""
toc_extractor.py
================
Two-track extractor for IMF-style PDFs:

  Track 1 — Box & Annex titles
    Pure PyMuPDF (fitz). Uses the embedded PDF outline (bookmarks) as primary
    source. Falls back to text scanning the TOC page.
    Runtime: < 1 s per file.

  Track 2 — Structural Benchmarks
    Uses Docling on a targeted sub-PDF (pages mentioning "Structural Benchmark"
    only). Table HTML is parsed via _html_to_dict() from docling_extractor —
    which correctly handles merged cells, multi-row headers, and colspan/rowspan.
    Falls back to DataFrame export if HTML is unavailable.

    Three table layouts are supported:
      Format A (TZA) — Mixed "Prior Actions + Structural Benchmarks" table.
                       Row key = benchmark action text.
                       Signal  = row whose key == "Structural Benchmarks".
      Format B (BEN) — Dedicated SB table; action text lives in a column.
                       Row key = reform area category.
                       Signal  = column named "Structural_benchmark" (or similar).
      Format C (fallback) — Title mentions "Structural Benchmark" but layout
                       doesn't match A or B. All non-header rows collected.

  Track 2b — Vision API fallback (image-embedded tables)
    Some IMF PDFs embed the SB table as a raster PNG image rather than as
    PDF text objects. Docling, pdfplumber, and fitz all return zero rows in
    that case because there is literally no text in the table area.

    This fallback is triggered in TWO situations:
      a) Docling found zero results on pages that had SB text nearby
      b) The document is a program REQUEST file where the entire SB table
         is image-embedded and _find_benchmark_pages() returns [] because
         there is no "Structural Benchmark" text anywhere in the PDF text layer.

    In case (b), _find_benchmark_pages_with_image_scan() is called to locate
    pages with large embedded images that could be an SB table, even without
    any text signal.
"""

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pdf_extractor.toc_extractor")

# ---------------------------------------------------------------------------
# Patterns — TOC / outline
# ---------------------------------------------------------------------------

_BOX_PARENT_RE      = re.compile(r"^\s*BOX(ES)?\s*$", re.IGNORECASE)
_ANNEX_PARENT_RE    = re.compile(r"^\s*ANNEX(ES)?\s*$", re.IGNORECASE)
_FIGURE_PARENT_RE   = re.compile(r"^\s*FIGURES?\s*$", re.IGNORECASE)
_TABLE_PARENT_RE    = re.compile(r"^\s*TABLES?\s*$", re.IGNORECASE)
_APPENDIX_PARENT_RE = re.compile(r"^\s*APPENDIX\s*", re.IGNORECASE)
_SKIP_PARENTS       = (_FIGURE_PARENT_RE, _TABLE_PARENT_RE, _APPENDIX_PARENT_RE)

_TOC_HEADING_RE = re.compile(
    r"(table\s+of\s+contents|^\s*CONTENTS\s*$)", re.IGNORECASE | re.MULTILINE
)
_BOX_HEADER_RE   = re.compile(r"^\s*BOX(ES)?\s*$", re.IGNORECASE)
_ANNEX_HEADER_RE = re.compile(r"^\s*ANNEX(ES)?\s*$", re.IGNORECASE)
_ANY_HEADER_RE   = re.compile(
    r"^\s*(BOXES?|ANNEX(ES)?|FIGURES?|TABLES?|APPENDIX|APPENDICES|CONTENTS)\s*$",
    re.IGNORECASE,
)
_TOC_LINE_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<title>\S.{1,150}?)"
    r"[\s._\u2026\u00B7\u2013\u2014]{2,}"
    r"(?P<page>\d{1,4})\s*$",
)

# ---------------------------------------------------------------------------
# Patterns — Structural Benchmark table detection
# ---------------------------------------------------------------------------

_SB_TEXT_RE    = re.compile(r"structural\s+benchmark", re.IGNORECASE)
_SB_TITLE_RE   = re.compile(r"structural\s+benchmark", re.IGNORECASE)
_SB_SECTION_RE = re.compile(r"^structural\s+benchmarks?\s*$", re.IGNORECASE)

# Matches the column name that CONTAINS the benchmark action text (BEN-style).
# Docling sometimes produces fully-qualified names like:
#   "Table 11b. Benin: Structural Benchmarks, 2022-23.Structural benchmark"
# so we match against the SHORT suffix after the last "." as well as the full name.
_SB_COL_RE = re.compile(r"^structural.?benchmarks?$", re.IGNORECASE)


def _col_short(col: str) -> str:
    """Return the part of a column name after the last '.', stripped."""
    return col.rsplit(".", 1)[-1].strip()


def _is_sb_col(col: str) -> bool:
    """True if this column (possibly namespace-prefixed) holds benchmark text."""
    return bool(_SB_COL_RE.match(_col_short(col)))


def _is_date_col(col: str) -> bool:
    return bool(re.search(r"date", _col_short(col), re.IGNORECASE))


def _is_status_col(col: str) -> bool:
    return bool(re.search(r"status", _col_short(col), re.IGNORECASE))

# Matches reform-area / policy-area column names
_REFORM_COL_RE = re.compile(
    r"^(reform.?area|policy.?area|area|sector|theme|subject)$", re.IGNORECASE
)

# Reform area prefix patterns
_REFORM_PREFIX_RE = re.compile(
    r"^(AML[/-]?CFT"
    r"|Governance and Transparency"
    r"|Governance"
    r"|Financial Inclusion"
    r"|Revenue Mobilization"
    r"|Public Financial Management"
    r"|Food security"
    r"|Land titling"
    r"|Social Safety Nets?"
    r"|Statistics"
    r"|Monetary Policy"
    r"|Financial Sector"
    r"|Fiscal \w+"
    r"|Banking \w+"
    r"|Trade"
    r"|Investment"
    r"|Labor \w+"
    r"|Climate \w+"
    r"|Health \w+"
    r"|Education \w+"
    r"|Tax \w+"
    r"|Energy \w+"
    r"|Debt \w+"
    r"|Gender \w+"
    r"|Poverty \w+"
    r"|Infrastructure \w*"
    r")",
    re.IGNORECASE,
)

# Minimum image dimensions to be considered a "real" table image
_IMG_MIN_WIDTH_PX   = 400
_IMG_MIN_HEIGHT_PX  = 150
_IMG_MAX_ASPECT     = 5.0

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TocTarget:
    category: str   # "Box" | "Annex"
    title: str
    page: int
    level: int = 1
    source_pdf: str = ""

    def __str__(self):
        return f"{'  '*(self.level-1)}[{self.category}] {self.title}  ->  p.{self.page}"

    def to_dict(self):
        return {"category": self.category, "title": self.title, "page": self.page}


@dataclass
class BenchmarkResult:
    table_title: str
    page_num: int
    columns: list[str]
    rows: dict   # {benchmark_text: {col: value}}

    def to_dict(self):
        return {
            "table_title": self.table_title,
            "page_num":    self.page_num,
            "columns":     self.columns,
            "rows":        self.rows,
        }


# ---------------------------------------------------------------------------
# Track 1 — Box & Annex from TOC  (pure fitz, < 1 s)
# ---------------------------------------------------------------------------

def _extract_from_outline(outline: list, source_pdf: str) -> list[TocTarget]:
    results: list[TocTarget] = []
    current_category: str | None = None
    parent_level: int | None = None

    for lvl, title, page, *_ in outline:
        title_s = title.strip()

        if _BOX_PARENT_RE.match(title_s):
            current_category, parent_level = "Box", lvl
            continue
        elif _ANNEX_PARENT_RE.match(title_s):
            current_category, parent_level = "Annex", lvl
            continue
        elif any(p.match(title_s) for p in _SKIP_PARENTS):
            current_category, parent_level = None, lvl
            continue

        if parent_level is not None and lvl <= parent_level:
            current_category, parent_level = None, None

        if current_category and parent_level is not None and lvl == parent_level + 1 and title_s:
            results.append(TocTarget(
                category=current_category, title=title_s,
                page=page, level=1, source_pdf=source_pdf,
            ))

    return results


def _extract_from_text(doc, source_pdf: str) -> list[TocTarget]:
    results: list[TocTarget] = []
    toc_pages = [
        i for i in range(min(30, len(doc)))
        if _TOC_HEADING_RE.search(doc[i].get_text())
    ]
    if not toc_pages:
        return []

    for page_idx in toc_pages:
        text = doc[page_idx].get_text()
        current_category: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            if _ANY_HEADER_RE.match(stripped):
                if _BOX_HEADER_RE.match(stripped):
                    current_category = "Box"
                elif _ANNEX_HEADER_RE.match(stripped):
                    current_category = "Annex"
                else:
                    current_category = None
                continue
            if current_category is None:
                continue
            m = _TOC_LINE_RE.match(line)
            if not m:
                continue
            title = m.group("title").strip().rstrip("_. ")
            page  = int(m.group("page"))
            level = 1 + len(m.group("indent")) // 2
            if title:
                results.append(TocTarget(
                    category=current_category, title=title,
                    page=page, level=level, source_pdf=source_pdf,
                ))
    return results


def extract_toc_targets(pdf_path: str) -> list[TocTarget]:
    import fitz
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)

    outline = doc.get_toc(simple=False)
    if outline:
        results = _extract_from_outline(outline, pdf_path)
        if results:
            doc.close()
            logger.info("[%s] %d entries from embedded outline", pdf_path, len(results))
            return results
        logger.debug("[%s] Outline has no Box/Annex; trying text scan", pdf_path)

    results = _extract_from_text(doc, pdf_path)
    doc.close()

    if results:
        logger.info("[%s] %d entries from text scan", pdf_path, len(results))
    else:
        logger.warning("[%s] No Box/Annex entries found", pdf_path)
    return results


# ---------------------------------------------------------------------------
# Track 2 — Structural Benchmarks via targeted Docling
# ---------------------------------------------------------------------------

def _page_has_large_image(doc, page_idx: int) -> bool:
    """Return True if this page contains a raster image large enough to be a table."""
    page = doc[page_idx]
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            info = doc.extract_image(xref)
        except Exception:
            continue
        w, h = info["width"], info["height"]
        aspect = w / h if h > 0 else 999
        if w >= _IMG_MIN_WIDTH_PX and h >= _IMG_MIN_HEIGHT_PX and aspect <= _IMG_MAX_ASPECT:
            return True
    return False


def _find_benchmark_pages(pdf_path: str) -> tuple[list[int], bool]:
    """
    Locate pages relevant to Structural Benchmarks.

    Returns (page_nums, image_only) where:
      page_nums  — 1-based page numbers to include in the sub-PDF
      image_only — True when no text signal was found and pages were chosen
                   purely from embedded-image heuristics (program request files)

    Strategy:
      1. Scan all pages for "structural benchmark" text (catches review files
         and request files where the table title is in the text layer).
      2. If text scan finds nothing, scan for pages with large embedded images
         that are plausible SB table candidates (catches fully image-embedded
         request files where there is no searchable text at all).
         In this mode we cap at the first 5 qualifying pages to avoid
         processing the entire document.
    """
    import fitz
    doc = fitz.open(pdf_path)
    total = len(doc)

    # ── Step 1: text-based scan (covers reviews + most requests) ─────────────
    text_pages = [
        i + 1
        for i in range(total)
        if _SB_TEXT_RE.search(doc[i].get_text())
    ]

    if text_pages:
        doc.close()
        logger.info("[%s] SB pages (text): %s", pdf_path, text_pages)
        return text_pages, False

    # ── Step 2: image-based scan (covers fully image-embedded request files) ──
    # Look for pages with a large image in the first 60% of the document
    # (SB tables appear in the body, not the appendix in request files).
    search_limit = max(10, int(total * 0.6))
    image_pages = [
        i + 1
        for i in range(min(search_limit, total))
        if _page_has_large_image(doc, i)
    ][:5]  # cap at 5 pages to stay efficient

    doc.close()

    if image_pages:
        logger.info(
            "[%s] No SB text found — using image-page heuristic: %s",
            pdf_path, image_pages,
        )
        return image_pages, True

    logger.info("[%s] No SB pages found (text or image scan)", pdf_path)
    return [], False


def _write_subpdf(pdf_path: str, page_nums: list[int], out_path: str) -> None:
    import fitz
    doc = fitz.open(pdf_path)
    total = len(doc)
    expanded = sorted({pp for p in page_nums for pp in (p-1, p, p+1) if 1 <= pp <= total})
    sub = fitz.open()
    for p in expanded:
        sub.insert_pdf(doc, from_page=p-1, to_page=p-1)
    sub.save(out_path)
    sub.close()
    doc.close()
    logger.info("Sub-PDF: %d pages -> %s", len(expanded), out_path)


# ---------------------------------------------------------------------------
# Table parsing — format detection and extraction
# ---------------------------------------------------------------------------

def _get_table_data(table, doc) -> tuple[dict, list[str], str]:
    html = ""
    try:
        if hasattr(table, "export_to_html"):
            try:
                html = table.export_to_html(doc=doc) or ""
            except TypeError:
                html = table.export_to_html() or ""
    except Exception:
        pass

    if html:
        try:
            from docling_extractor import _html_to_dict
            data, columns = _html_to_dict(html)
            if data and columns:
                logger.debug("Table parsed via HTML (%d rows, %d cols)", len(data), len(columns))
                return data, columns, html
        except ImportError:
            logger.debug("docling_extractor not on path — using DataFrame fallback")
        except Exception as e:
            logger.debug("_html_to_dict failed: %s", e)

    try:
        import pandas as pd
        df = table.export_to_dataframe(doc=doc)
        if df is None:
            return {}, [], html
        try:
            is_empty = df.empty
        except ValueError:
            is_empty = False
        if is_empty or df.shape[1] < 2:
            return {}, [], html

        raw_cols = [str(c) for c in df.columns]
        columns  = raw_cols[1:]
        data: dict = {}
        for _, row in df.iterrows():
            key = str(row.iloc[0]).strip()
            if not key or key == "nan":
                continue
            data[key] = {
                col: (str(row[col]).strip()
                      if pd.notna(row[col]) and str(row[col]).strip() not in ("", "nan")
                      else None)
                for col in columns
            }
        if data:
            logger.debug("Table parsed via DataFrame (%d rows)", len(data))
            return data, columns, html
    except Exception as e:
        logger.warning("DataFrame export failed: %s", e)

    return {}, [], html


def _detect_sb_format(columns: list[str], data: dict, table_title: str) -> str:
    if any(_is_sb_col(c) for c in columns):
        return "B"
    for key in data:
        if _SB_SECTION_RE.match(key.strip()):
            return "A"
    if _SB_TITLE_RE.search(table_title):
        return "C"
    return "none"


def _extract_format_a(data: dict, columns: list[str]) -> tuple[dict, list[str]]:
    in_sb = False
    rows: dict = {}
    for key, val in data.items():
        if _SB_SECTION_RE.match(key.strip()):
            in_sb = True
            continue
        if in_sb:
            all_null = isinstance(val, dict) and all(v is None for v in val.values())
            if all_null and len(key.strip()) < 60 and not key.strip()[:1].isdigit():
                break
            rows[key.strip()] = {c: val.get(c) for c in columns}
    return rows, list(columns)


def _split_reform_and_benchmark(text: str) -> tuple[str, str]:
    text = text.strip()
    m = _REFORM_PREFIX_RE.match(text)
    if m:
        prefix = m.group(0).strip()
        rest = text[len(prefix):].strip()
        if len(rest) > 20:
            return prefix, rest
    return "", text


# ── Known reform area labels ──────────────────────────────────────────────────
_KNOWN_REFORM_AREAS: list[str] = sorted([
    "Governance and Transparency", "Financial Inclusion", "Revenue Mobilization",
    "Public Financial Management", "Food security", "Land titling", "AML/CFT",
    "Social Safety Nets", "Social Protection", "Statistics", "Monetary Policy",
    "Financial Sector", "Financial Stability", "Tax Policy", "Investment Climate",
    "Health", "Education", "Climate Change", "Poverty Reduction", "Gender",
    "Debt Management", "Fiscal Policy", "State-Owned Enterprises", "SOEs",
    "Banking Sector", "External Sector", "Trade Policy", "Exchange Rate",
    "Pension", "Insurance", "Business Environment", "Digitalization",
    "Infrastructure", "Agriculture", "Judiciary", "Labor Market",
    "Customs Administration", "Rule of law", "Anti-corruption",
], key=len, reverse=True)

_REFORM_FULL_RE = re.compile(
    r"^(" + "|".join(re.escape(r) for r in _KNOWN_REFORM_AREAS) + r")(\s+|$)",
    re.IGNORECASE,
)
_STATUS_SUFFIX_RE = re.compile(
    r"(?:[.\s])+(Met|Not\s+met|Ongoing|Proposed|Completed|Implemented|Delayed|Missed)\s*$",
    re.IGNORECASE,
)
_DATE_VAL_RE = re.compile(
    r"^(End-[A-Za-z]+\.?\s*\d{0,4}"
    r"|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
    re.IGNORECASE,
)
_STATUS_VAL_RE = re.compile(
    r"^(Met|Not\s+met|Ongoing|Proposed|Completed|Implemented|Delayed|"
    r"Newly\s+proposed|Missed|Partly\s+met)$",
    re.IGNORECASE,
)
_AREA_FIRST_WORDS: dict[str, list[str]] = {}
for _area in _KNOWN_REFORM_AREAS:
    _fw = _area.split()[0].lower()
    _AREA_FIRST_WORDS.setdefault(_fw, []).append(_area)


def _classify_sb_cell(cell: str) -> tuple[str, str, str]:
    cell = cell.strip()
    if not cell:
        return ("empty", "", "")
    if _DATE_VAL_RE.match(cell) and len(cell) < 40:
        return ("date", "", cell)
    if _STATUS_VAL_RE.match(cell):
        return ("status", "", cell)
    m = _REFORM_FULL_RE.match(cell)
    if m:
        area = m.group(1)
        rest = cell[len(area):].strip()
        return ("reform+bench", area, rest) if rest else ("reform", area, "")
    sm = _STATUS_SUFFIX_RE.search(cell)
    if sm:
        status = sm.group(1)
        prefix = cell[:sm.start()].strip()
        m2 = _REFORM_FULL_RE.match(prefix)
        if m2:
            return ("reform+status", m2.group(1), status)
        return ("bench+status", status, prefix)
    words = cell.split()
    if words and words[0].lower() in _AREA_FIRST_WORDS:
        if len(words) > 1 and words[1][0].isupper() and words[1].lower() not in ("and","or","of","the"):
            best_area = _AREA_FIRST_WORDS[words[0].lower()][0]
            return ("partial_reform+bench", best_area, " ".join(words[1:]))
    return ("bench", "", cell)


def _smart_join(frags: list[str]) -> str:
    frags = [f.strip() for f in frags if f and f.strip()]
    if not frags:
        return ""
    if len(frags) == 1:
        return frags[0]
    result = frags[0]
    for frag in frags[1:]:
        if frag.lower() in result.lower():
            continue
        overlap_found = False
        for n in range(min(len(result), len(frag), 30), 2, -1):
            if result.lower().endswith(frag[:n].lower()):
                result += frag[n:]
                overlap_found = True
                break
        if not overlap_found:
            result = result.rstrip() + " " + frag
    return result.strip()


def _normalize_sb_row(row_key: str, vals: dict) -> dict:
    all_cells = [str(row_key).strip()] + [
        str(v).strip() for v in vals.values()
        if v and str(v).strip() not in ("", "None", "nan")
    ]
    reform_area = ""
    date_val    = ""
    status_val  = ""
    bench_frags: list[str] = []

    for cell in all_cells:
        typ, area, text = _classify_sb_cell(cell)
        if   typ == "empty":              continue
        elif typ == "date":               date_val   = date_val   or text
        elif typ == "status":             status_val = status_val or text
        elif typ == "reform":
            if not reform_area: reform_area = area
        elif typ in ("reform+bench", "partial_reform+bench"):
            if not reform_area: reform_area = area
            if text: bench_frags.append(text)
        elif typ == "reform+status":
            if not reform_area: reform_area = area
            if not status_val:  status_val  = text
        elif typ == "bench+status":
            if not status_val:  status_val = area
            if text: bench_frags.append(text)
        elif typ == "bench":
            if text: bench_frags.append(text)

    cleaned_frags = []
    for frag in bench_frags:
        extracted = False
        for area in _KNOWN_REFORM_AREAS:
            if frag.lower().endswith(" " + area.lower()) or frag.lower().endswith("." + area.lower()):
                suffix_start = len(frag) - len(area)
                bench_part = frag[:suffix_start].rstrip(" .,;").strip()
                if not reform_area:
                    reform_area = area
                if bench_part:
                    cleaned_frags.append(bench_part)
                extracted = True
                break
            if frag.lower().endswith(" in " + area.lower()):
                bench_part = frag[:-(len(area)+4)].strip()
                if not reform_area:
                    reform_area = area
                if bench_part:
                    cleaned_frags.append(bench_part)
                extracted = True
                break
        if not extracted:
            cleaned_frags.append(frag)
    bench_frags = cleaned_frags

    final_frags = []
    for frag in bench_frags:
        found_prefix = False
        for area in _KNOWN_REFORM_AREAS:
            area_words = area.split()
            if len(area_words) > 1:
                for tail_len in range(len(area_words)-1, 0, -1):
                    tail = " ".join(area_words[-tail_len:])
                    if frag.lower().startswith(tail.lower() + " "):
                        rest = frag[len(tail):].strip()
                        if not reform_area:
                            reform_area = area
                        if rest:
                            final_frags.append(rest)
                        found_prefix = True
                        break
            if found_prefix:
                break
        if not found_prefix:
            final_frags.append(frag)
    bench_frags = final_frags

    bench_frags.sort(key=len, reverse=True)
    return {
        "benchmark":   _smart_join(bench_frags),
        "due_date":    date_val   or None,
        "status":      status_val or None,
        "reform_area": reform_area,
    }


def _identify_col_types(columns: list[str], data: dict) -> dict[str, str]:
    col_values: dict[str, list[str]] = {c: [] for c in columns}
    for row_vals in data.values():
        if not isinstance(row_vals, dict):
            continue
        for col, val in row_vals.items():
            if val and col in col_values:
                col_values[col].append(str(val).strip())

    result = {}
    for col, values in col_values.items():
        if not values:
            result[col] = "unknown"
            continue
        n = len(values)
        date_hits   = sum(1 for v in values if _DATE_VAL_RE.match(v) and len(v) < 50)
        status_hits = sum(1 for v in values if _STATUS_VAL_RE.match(v))
        if date_hits / n > 0.35:
            result[col] = "date"
        elif status_hits / n > 0.35:
            result[col] = "status"
        else:
            result[col] = "other"
    return result


def _sb_col_is_clean(col_name: str, data: dict) -> bool:
    values = [
        str(row.get(col_name, "") or "").strip()
        for row in data.values()
        if isinstance(row, dict)
    ]
    non_null = [v for v in values if v]
    if not non_null:
        return False
    avg_len = sum(len(v) for v in non_null) / len(non_null)
    if avg_len < 25:
        return False
    noise = sum(
        1 for v in non_null
        if _DATE_VAL_RE.match(v) or _STATUS_VAL_RE.match(v) or _REFORM_FULL_RE.match(v)
    )
    return noise / len(non_null) < 0.3


def _extract_format_b(data: dict, columns: list[str]) -> tuple[dict, list[str]]:
    sb_col     = next((c for c in columns if _is_sb_col(c)),     None)
    date_col   = next((c for c in columns if not _is_sb_col(c) and _is_date_col(c)),   None)
    status_col = next((c for c in columns if not _is_sb_col(c) and _is_status_col(c)), None)

    if sb_col and _sb_col_is_clean(sb_col, data):
        rows: dict = {}
        for row_key, val in data.items():
            if not isinstance(val, dict):
                continue
            benchmark = (val.get(sb_col) or "").strip()
            if not benchmark or len(benchmark) < 15:
                continue
            if not re.search(
                r"(adopt|prepare|develop|publish|implement|submit|establish|"
                r"operationalize|conduct|strengthen|update|finalize|complete|"
                r"introduce|design|digitalize|transpose|reconcile|appoint|"
                r"hire|interface|formalize|build|increase|reduce|reform|"
                r"review|assess|ensure|provide|support|create|integrate)",
                benchmark, re.IGNORECASE
            ):
                continue
            rows[benchmark] = {
                "Due date":    val.get(date_col)              if date_col   else None,
                "Status":      val.get(status_col) or "New"   if status_col else "New",
                "Reform Area": row_key.strip(),
            }
        if rows:
            return rows, ["Due date", "Status", "Reform Area"]

    rows = {}
    for row_key, val in data.items():
        if not isinstance(val, dict):
            continue
        normalized = _normalize_sb_row(row_key, val)
        benchmark  = normalized["benchmark"]
        if not benchmark or len(benchmark) < 15:
            continue
        if _REFORM_FULL_RE.match(benchmark.strip()):
            continue
        rows[benchmark] = {
            "Due date":    normalized["due_date"],
            "Status":      normalized["status"],
            "Reform Area": normalized["reform_area"],
        }
    return rows, ["Due date", "Status", "Reform Area"]


def _extract_format_c(data: dict, columns: list[str]) -> tuple[dict, list[str]]:
    rows: dict = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        all_null = all(v is None for v in val.values())
        if all_null and len(key.strip()) < 50:
            continue
        rows[key.strip()] = {c: val.get(c) for c in columns}
    return rows, list(columns)


def _infer_table_title(columns: list[str], html: str) -> str:
    for col in columns:
        if "." in col:
            candidate = col.rsplit(".", 1)[0].strip()
            if _SB_TITLE_RE.search(candidate) and len(candidate) > 10:
                return candidate

    if html:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            cap = soup.find("caption")
            if cap:
                return cap.get_text(strip=True)
        except Exception:
            pass

    return ""


def _parse_benchmark_tables(
    doc,
    original_page_nums: list[int],
    image_only_mode: bool = False,
) -> list[BenchmarkResult]:
    """
    Scan all Docling tables. For each that is a Structural Benchmark table,
    detect the format and extract rows.

    image_only_mode: when True, the sub-PDF pages were chosen via image
    heuristic (no SB text found), so we relax the signal_text requirement
    and trust the table title / format detection alone.
    """
    sorted_orig = sorted(original_page_nums)
    results: list[BenchmarkResult] = []

    for table in doc.tables:
        sub_page = 1
        if hasattr(table, "prov") and table.prov:
            prov = table.prov[0]
            if hasattr(prov, "page_no"):
                sub_page = prov.page_no
        orig_page = sorted_orig[min(sub_page - 1, len(sorted_orig) - 1)]

        data, columns, html = _get_table_data(table, doc)
        if not data or not columns:
            continue

        signal_text = " ".join(columns) + " " + " ".join(list(data.keys())[:10]) + " " + html[:500]

        if not image_only_mode and not _SB_TEXT_RE.search(signal_text):
            continue

        table_title = _infer_table_title(columns, html)

        fmt = _detect_sb_format(columns, data, table_title)
        logger.info("  p.%d  format=%s  cols=%s", orig_page, fmt, columns[:4])

        if fmt == "A":
            rows, out_cols = _extract_format_a(data, columns)
        elif fmt == "B":
            rows, out_cols = _extract_format_b(data, columns)
        elif fmt == "C":
            rows, out_cols = _extract_format_c(data, columns)
        else:
            logger.debug("  Skipping — not a SB table (format=none)")
            continue

        if not rows:
            logger.debug("  No benchmark rows extracted (format %s)", fmt)
            continue

        logger.info("  -> %d benchmark rows extracted", len(rows))
        results.append(BenchmarkResult(
            table_title=table_title,
            page_num=orig_page,
            columns=out_cols,
            rows=rows,
        ))

    seen_row_keys: set[frozenset] = set()
    deduped: list[BenchmarkResult] = []
    for br in results:
        key = frozenset(br.rows.keys())
        if key not in seen_row_keys:
            seen_row_keys.add(key)
            deduped.append(br)
        else:
            logger.info("  Duplicate table dropped (page %d, %d rows)", br.page_num, len(br.rows))
    return deduped


# ---------------------------------------------------------------------------
# Track 2b — Vision API fallback for image-embedded SB tables
# ---------------------------------------------------------------------------

_VISION_MODEL = "claude-sonnet-4-20250514"

_VISION_PROMPT = """You are extracting data from an IMF Structural Benchmarks table image.

The table always has these four columns (exact names may vary slightly):
  1. Reform area   — short category label (e.g. "Transparency", "Business climate")
  2. Structural benchmark — the full action text (a sentence or two describing the reform)
  3. Due date      — e.g. "End-June 2025", "End-April 2025"
  4. Status        — e.g. "Met", "On track", "Newly proposed SB", "Not met"

Rules:
- The Reform area cell often SPANS multiple rows (merged cell). Repeat the value for every row it covers.
- The Status cell may span multiple lines within one row (e.g. "Newly\nproposed\nSB" → "Newly proposed SB").
- Extract EVERY data row. Skip the header row.
- If a cell is empty or blank, use null.

Return ONLY a JSON array — no markdown, no explanation, no code fences.
Each element must have exactly these keys:
  "reform_area", "benchmark", "due_date", "status"

Example of a single element:
{"reform_area": "Transparency", "benchmark": "Publish the audit reports of three high-stake public contracts executed during 2022-24.", "due_date": "End-June 2025", "status": "On track"}
"""


def _page_has_image_table(pdf_path: str, page_num: int) -> bool:
    import fitz
    doc = fitz.open(pdf_path)
    try:
        return _page_has_large_image(doc, page_num - 1)
    finally:
        doc.close()


def _extract_image_from_page(pdf_path: str, page_num: int):
    import fitz, io
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        best = None
        best_pixels = 0
        for img in page.get_images(full=True):
            xref = img[0]
            info = doc.extract_image(xref)
            w, h = info["width"], info["height"]
            aspect = w / h if h > 0 else 999
            if w >= _IMG_MIN_WIDTH_PX and h >= _IMG_MIN_HEIGHT_PX and aspect <= _IMG_MAX_ASPECT:
                pixels = w * h
                if pixels > best_pixels:
                    best_pixels = pixels
                    best = info

        if best is not None:
            raw = best["image"]
            ext = best["ext"].lower()
            media_type = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "gif", "webp") else "image/png"
            if ext == "jpg":
                media_type = "image/jpeg"

            if best["width"] < 1200:
                try:
                    from PIL import Image
                    scale = max(2, 600 // best["width"])
                    img_pil = Image.open(io.BytesIO(raw))
                    img_pil = img_pil.resize(
                        (img_pil.width * scale, img_pil.height * scale),
                        Image.LANCZOS,
                    )
                    buf = io.BytesIO()
                    img_pil.save(buf, format="PNG")
                    raw = buf.getvalue()
                    media_type = "image/png"
                except Exception:
                    pass

            return raw, media_type

        logger.debug("  p.%d  no embedded image — rendering full page", page.number + 1)
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png"), "image/png"

    finally:
        doc.close()


def _call_vision_api(image_bytes: bytes, media_type: str) -> list[dict]:
    import base64, json, os as _os

    api_key = _os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — vision API fallback will not work. "
            "Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
        )
        return []

    b64  = base64.standard_b64encode(image_bytes).decode("ascii")
    body = {
        "model": _VISION_MODEL,
        "max_tokens": 2048,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": _VISION_PROMPT},
            ],
        }],
    }
    headers = {
        "content-type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }

    text = ""

    try:
        import requests as _req
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=body, headers=headers, timeout=90,
        )
        resp.raise_for_status()
        for block in resp.json().get("content", []):
            if block.get("type") == "text":
                text = block["text"].strip()
                break
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Vision API (requests) failed: %s", exc)
        return []

    if not text:
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=_VISION_MODEL, max_tokens=2048,
                messages=body["messages"],
            )
            text = msg.content[0].text.strip() if msg.content else ""
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Vision API (SDK) failed: %s", exc)
            return []

    if not text:
        try:
            import ssl, urllib.request
            try:
                import certifi as _certifi
                ctx = ssl.create_default_context(cafile=_certifi.where())
            except ImportError:
                ctx = ssl.create_default_context()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps(body).encode(),
                headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
                for block in json.loads(r.read()).get("content", []):
                    if block.get("type") == "text":
                        text = block["text"].strip()
                        break
        except Exception as exc:
            logger.warning("Vision API (urllib) failed: %s", exc)
            return []

    if not text:
        logger.warning("Vision API returned empty response")
        return []

    text = re.sub(r"^```[a-zA-Z]*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```$",          "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        rows = json.loads(text)
        if not isinstance(rows, list):
            logger.warning("Vision API returned non-list JSON: %s", type(rows))
            return []
        return rows
    except json.JSONDecodeError as exc:
        logger.warning("Vision API JSON parse error: %s  (text: %s...)", exc, text[:200])
        return []


def _vision_rows_to_benchmark_result(
    rows: list[dict], page_num: int, table_title: str
) -> "BenchmarkResult | None":
    _ACTION_VERB_RE = re.compile(
        r"\b(adopt|prepare|develop|publish|implement|submit|establish|"
        r"operationalize|conduct|strengthen|update|finalize|complete|"
        r"introduce|design|digitalize|transpose|reconcile|appoint|"
        r"hire|interface|formalize|build|increase|reduce|reform|"
        r"review|assess|ensure|provide|support|create|integrate|"
        r"extend|in line)\b",
        re.IGNORECASE,
    )

    result_rows: dict = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        benchmark = (item.get("benchmark") or "").strip()
        if not benchmark or len(benchmark) < 15:
            continue
        if not _ACTION_VERB_RE.search(benchmark):
            continue
        result_rows[benchmark] = {
            "Due date":    (item.get("due_date") or None),
            "Status":      (item.get("status")   or None),
            "Reform Area": (item.get("reform_area") or None),
        }

    if not result_rows:
        return None

    return BenchmarkResult(
        table_title=table_title,
        page_num=page_num,
        columns=["Due date", "Status", "Reform Area"],
        rows=result_rows,
    )


def _extract_sb_via_vision(
    pdf_path: str, benchmark_pages: list[int]
) -> list["BenchmarkResult"]:
    results: list[BenchmarkResult] = []
    seen_keys: set[frozenset] = set()

    for pg in sorted(set(benchmark_pages)):
        if not _page_has_image_table(pdf_path, pg):
            logger.debug("  p.%d  no qualifying image found — skipping vision", pg)
            continue

        logger.info("  p.%d  no Docling tables found — trying vision API", pg)
        image_bytes, media_type = _extract_image_from_page(pdf_path, pg)
        if image_bytes is None:
            logger.warning("  p.%d  could not extract image", pg)
            continue

        rows = _call_vision_api(image_bytes, media_type)
        if not rows:
            logger.warning("  p.%d  vision API returned no rows", pg)
            continue

        logger.info("  p.%d  vision API returned %d row(s)", pg, len(rows))
        br = _vision_rows_to_benchmark_result(rows, pg, table_title=f"SB table (vision, p.{pg})")
        if br is None:
            logger.warning("  p.%d  no valid benchmark rows after filtering", pg)
            continue

        key = frozenset(br.rows.keys())
        if key in seen_keys:
            logger.info("  p.%d  duplicate table dropped (vision)", pg)
            continue
        seen_keys.add(key)
        results.append(br)
        logger.info("  -> %d benchmark rows extracted (vision)", len(br.rows))

    return results


def extract_structural_benchmarks(
    pdf_path: str,
    table_mode: str = "accurate",
) -> list[BenchmarkResult]:
    """
    Extract Structural Benchmark tables using targeted Docling.

    Handles both program review files (SB text in PDF layer) and program
    request files (SB table may be fully image-embedded with no text signal).
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions, TableFormerMode, TableStructureOptions,
    )

    pdf_path = str(pdf_path)

    # ── Step 1: locate relevant pages ────────────────────────────────────────
    # Returns (pages, image_only_mode).
    # image_only_mode=True means no SB text was found anywhere — the file is
    # likely a program request with a fully image-embedded table.
    benchmark_pages, image_only_mode = _find_benchmark_pages(pdf_path)

    if not benchmark_pages:
        logger.info("[%s] No SB pages found (text or image scan) — skipping", pdf_path)
        return []

    # ── Step 2: build sub-PDF and run Docling ─────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        subpdf_path = tmp.name

    try:
        _write_subpdf(pdf_path, benchmark_pages, subpdf_path)

        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                mode=TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST,
                do_cell_matching=True,
            ),
            generate_page_images=False,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        logger.info("Running Docling on sub-PDF (%s) ...", subpdf_path)
        result = converter.convert(subpdf_path)
        logger.info("Docling complete")

        docling_results = _parse_benchmark_tables(
            result.document, benchmark_pages, image_only_mode=image_only_mode
        )

        # ── Step 3: vision API fallback ───────────────────────────────────────
        # Triggered when:
        #   a) image_only_mode: the whole file had no SB text — definitely
        #      image-embedded. Try vision on all candidate pages.
        #   b) Normal mode but Docling found nothing — table may still be
        #      image-embedded on those specific pages.
        if not docling_results:
            logger.info(
                "[%s] Docling found no tables (image_only=%s) — trying vision API fallback",
                pdf_path, image_only_mode,
            )
            vision_results = _extract_sb_via_vision(pdf_path, benchmark_pages)
            if vision_results:
                return vision_results
            logger.warning("[%s] Vision API fallback also found no rows", pdf_path)

        return docling_results

    finally:
        try:
            os.unlink(subpdf_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def extract_all_targets(pdf_path: str, table_mode: str = "accurate") -> dict:
    return {
        "toc":        [e.to_dict() for e in extract_toc_targets(pdf_path)],
        "benchmarks": [b.to_dict() for b in extract_structural_benchmarks(pdf_path, table_mode)],
    }


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def batch_extract_all(
    pdf_paths: list[str],
    workers: int = 2,
    table_mode: str = "accurate",
) -> dict[str, dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pdf_paths = [str(p) for p in pdf_paths]
    results: dict[str, dict] = {}

    if workers == 1:
        for path in pdf_paths:
            try:
                results[path] = extract_all_targets(path, table_mode=table_mode)
            except Exception as exc:
                logger.error("[%s] Failed: %s", path, exc)
                results[path] = {"toc": [], "benchmarks": []}
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_all_targets, p, table_mode): p for p in pdf_paths}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                results[path] = fut.result()
            except Exception as exc:
                logger.error("[%s] Failed: %s", path, exc)
                results[path] = {"toc": [], "benchmarks": []}

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python toc_extractor.py file1.pdf [file2.pdf ...]")
        sys.exit(1)

    all_results = batch_extract_all(sys.argv[1:])

    for pdf, res in all_results.items():
        print(f"\n{'='*60}\n  {Path(pdf).name}\n{'='*60}")
        print(f"\n  Box & Annex entries ({len(res['toc'])}):")
        for e in res["toc"]:
            print(f"    [{e['category']}] {e['title']}  ->  p.{e['page']}")
        print(f"\n  Structural Benchmark tables ({len(res['benchmarks'])}):")
        for b in res["benchmarks"]:
            print(f"\n    {b['table_title']}  (p.{b['page_num']})")
            for label, vals in b["rows"].items():
                date   = vals.get("Due date") or vals.get("Due_date") or vals.get("Target Date", "")
                status = vals.get("Status", "")
                print(f"      [{status}] {label[:80]}  ({date})")

    out_path = "toc_targets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")
