"""
run_extraction.py
=================
Batch extractor that walks a country-folder structure and extracts
Box titles, Annex titles, and Structural Benchmark rows from every PDF.

Expected input layout
---------------------
  data/
    Tanzania/
      TZA_2023_ECF_Review1.pdf
      TZA_2022_Article_IV.pdf
    Kenya/
      KEN_2023_ECF.pdf

Output layout
-------------
  output/
    Tanzania/
      TZA_2023_ECF_Review1_results.json
      TZA_2022_Article_IV_results.json
    Kenya/
      KEN_2023_ECF_results.json
    boxes.csv           <- one row per Box entry
    annexes.csv         <- one row per Annex entry
    benchmarks.csv      <- one row per Structural Benchmark action
    summary.csv         <- combined (all three types)
    summary.json        <- combined, JSON format

Usage
-----
  python run_extraction.py                          # data/ -> output/
  python run_extraction.py --input /pdfs --output /results
  python run_extraction.py --workers 1              # serial (safer for memory)
  python run_extraction.py --mode fast              # faster Docling
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_extractor")


# ---------------------------------------------------------------------------
# Worker — runs in a subprocess per PDF
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> dict:
    pdf_path, country, table_mode = args
    from toc_extractor import extract_all_targets
    t0 = time.time()
    try:
        result = extract_all_targets(str(pdf_path), table_mode=table_mode)
        return {
            "status":     "ok",
            "country":    country,
            "pdf":        str(pdf_path),
            "stem":       pdf_path.stem,
            "elapsed":    round(time.time() - t0, 1),
            "toc":        result["toc"],
            "benchmarks": result["benchmarks"],
        }
    except Exception as exc:
        return {
            "status":     "error",
            "country":    country,
            "pdf":        str(pdf_path),
            "stem":       pdf_path.stem,
            "elapsed":    round(time.time() - t0, 1),
            "error":      str(exc),
            "toc":        [],
            "benchmarks": [],
        }


def _log_result(r: dict):
    mark = "✓" if r["status"] == "ok" else "✗"
    n_toc = len(r.get("toc", []))
    n_sb  = sum(len(b["rows"]) for b in r.get("benchmarks", []))
    logger.info("%s [%s] %-45s  %d toc  %d sb-rows  (%.1fs)",
                mark, r["country"], Path(r["pdf"]).name[:45],
                n_toc, n_sb, r["elapsed"])


# ---------------------------------------------------------------------------
# DataFrame builders  (called before batching outputs to disk)
# ---------------------------------------------------------------------------

def build_boxes_df(all_results: list[dict]) -> pd.DataFrame:
    """
    One row per Box entry.
    Columns: country, file, page, title
    """
    rows = [
        {
            "country": r["country"],
            "file":    Path(r["pdf"]).name,
            "page":    e["page"],
            "title":   e["title"],
        }
        for r in all_results
        for e in r.get("toc", [])
        if e["category"] == "Box"
    ]
    return pd.DataFrame(rows, columns=["country", "file", "page", "title"])


def build_annexes_df(all_results: list[dict]) -> pd.DataFrame:
    """
    One row per Annex entry.
    Columns: country, file, page, title
    """
    rows = [
        {
            "country": r["country"],
            "file":    Path(r["pdf"]).name,
            "page":    e["page"],
            "title":   e["title"],
        }
        for r in all_results
        for e in r.get("toc", [])
        if e["category"] == "Annex"
    ]
    return pd.DataFrame(rows, columns=["country", "file", "page", "title"])


def build_benchmarks_df(all_results: list[dict]) -> pd.DataFrame:
    """
    One row per Structural Benchmark action.
    Columns: country, file, page, table_title, reform_area,
             benchmark, due_date, status

    Handles both column naming conventions:
      - New (toc_extractor): "Due date", "Status", "Reform Area"
      - Old (TZA-style):     "Target Date", "Status", "Macroeconomic Rationale"

    Drops rows where benchmark is a reform area label (short, no action verb).
    """
    import re
    _ACTION_VERB_RE = re.compile(
        r"\b(adopt|prepare|develop|publish|implement|submit|establish|"
        r"operationalize|conduct|strengthen|update|finalize|complete|"
        r"introduce|design|digitalize|transpose|reconcile|appoint|"
        r"hire|interface|formalize|build|increase|reduce|reform|"
        r"review|assess|ensure|provide|support|create|integrate)\b",
        re.IGNORECASE
    )

    rows = []
    for r in all_results:
        for b in r.get("benchmarks", []):
            for label, vals in b.get("rows", {}).items():
                # Skip rows where the label is a reform area, not an action
                if not label or len(label) < 15:
                    continue
                if not _ACTION_VERB_RE.search(label):
                    continue

                # Normalise column names across both formats
                due_date = (
                    vals.get("Due date")
                    or vals.get("Due_date")
                    or vals.get("Target Date")
                    or vals.get("Target_Date")
                    or ""
                )
                status = vals.get("Status") or vals.get("status") or ""
                reform_area = (
                    vals.get("Reform Area")
                    or vals.get("Reform_Area")
                    or vals.get("Macroeconomic Rationale", "")
                )

                rows.append({
                    "country":     r["country"],
                    "file":        Path(r["pdf"]).name,
                    "page":        b["page_num"],
                    "table_title": b["table_title"],
                    "reform_area": reform_area,
                    "benchmark":   label,
                    "due_date":    due_date,
                    "status":      status,
                })

    cols = ["country", "file", "page", "table_title",
            "reform_area", "benchmark", "due_date", "status"]
    return pd.DataFrame(rows, columns=cols)


def build_summary_df(all_results: list[dict]) -> pd.DataFrame:
    """
    Combined DataFrame — one row per entry regardless of type.
    Columns: country, file, type, title, page, benchmark,
             target_date, status, macroeconomic_rationale, table_title
    """
    frames = []

    boxes = build_boxes_df(all_results)
    if not boxes.empty:
        boxes = boxes.copy()
        boxes["type"] = "Box"
        boxes["benchmark"] = ""
        boxes["target_date"] = ""
        boxes["status"] = ""
        boxes["macroeconomic_rationale"] = ""
        boxes["table_title"] = ""
        frames.append(boxes)

    annexes = build_annexes_df(all_results)
    if not annexes.empty:
        annexes = annexes.copy()
        annexes["type"] = "Annex"
        annexes["benchmark"] = ""
        annexes["target_date"] = ""
        annexes["status"] = ""
        annexes["macroeconomic_rationale"] = ""
        annexes["table_title"] = ""
        frames.append(annexes)

    benchmarks = build_benchmarks_df(all_results)
    if not benchmarks.empty:
        benchmarks = benchmarks.copy()
        benchmarks["type"] = "Structural Benchmark"
        benchmarks["title"] = benchmarks["table_title"]
        frames.append(benchmarks)

    if not frames:
        return pd.DataFrame()

    col_order = ["country", "file", "type", "title", "page", "reform_area",
                 "benchmark", "due_date", "status", "table_title"]
    df = pd.concat(frames, ignore_index=True)
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["country", "file", "type", "page"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch PDF extractor — country folders")
    parser.add_argument("--input",   default="data",    help="Root folder with country subfolders")
    parser.add_argument("--output",  default="output",  help="Root folder for results")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode",    default="accurate", choices=["accurate", "fast"])
    parser.add_argument("--glob",    default="**/*.pdf")
    args = parser.parse_args()

    input_root  = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        logger.error("Input folder not found: %s", input_root)
        sys.exit(1)

    # ── Discover PDFs ─────────────────────────────────────────────────────────
    tasks: list[tuple] = []
    country_folders = sorted(p for p in input_root.iterdir() if p.is_dir())
    root_pdfs       = sorted(input_root.glob("*.pdf"))

    for pdf in root_pdfs:
        tasks.append((pdf, "_root", args.mode))
    for folder in country_folders:
        for pdf in sorted(folder.glob(args.glob)):
            tasks.append((pdf, folder.name, args.mode))

    if not tasks:
        logger.error("No PDFs found under %s", input_root)
        sys.exit(1)

    logger.info("Found %d PDF(s) in %d country folder(s)",
                len(tasks), len(country_folders) + (1 if root_pdfs else 0))

    # ── Run extraction ────────────────────────────────────────────────────────
    all_results: list[dict] = []
    t_total = time.time()

    if args.workers == 1:
        for task in tasks:
            result = _process_one(task)
            all_results.append(result)
            _log_result(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, t): t for t in tasks}
            for fut in as_completed(futures):
                result = fut.result()
                all_results.append(result)
                _log_result(result)

    elapsed_total = time.time() - t_total
    logger.info("Extraction done — %d files in %.0f s", len(all_results), elapsed_total)

    # ── Build DataFrames (before writing anything to disk) ────────────────────
    logger.info("Building DataFrames ...")
    df_boxes      = build_boxes_df(all_results)
    df_annexes    = build_annexes_df(all_results)
    df_benchmarks = build_benchmarks_df(all_results)
    df_summary    = build_summary_df(all_results)

    logger.info("  Boxes      : %d rows", len(df_boxes))
    logger.info("  Annexes    : %d rows", len(df_annexes))
    logger.info("  Benchmarks : %d rows", len(df_benchmarks))
    logger.info("  Summary    : %d rows", len(df_summary))

    # ── Write per-file JSON ───────────────────────────────────────────────────
    for r in all_results:
        out_folder = output_root / r["country"]
        out_folder.mkdir(parents=True, exist_ok=True)
        out_file = out_folder / f"{r['stem']}_results.json"
        payload  = {
            "country":    r["country"],
            "file":       Path(r["pdf"]).name,
            "status":     r["status"],
            "elapsed_s":  r["elapsed"],
            "toc":        r.get("toc", []),
            "benchmarks": r.get("benchmarks", []),
        }
        if r["status"] == "error":
            payload["error"] = r.get("error", "")
        out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ── Write CSVs (utf-8-sig so Excel opens them correctly) ─────────────────
    output_root.mkdir(parents=True, exist_ok=True)

    def _save_csv(df: pd.DataFrame, name: str):
        path = output_root / name
        if df.empty:
            path.write_text("No data extracted.\n")
        else:
            df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("  Saved %s (%d rows)", path, len(df))
        return path

    csv_boxes      = _save_csv(df_boxes,      "boxes.csv")
    csv_annexes    = _save_csv(df_annexes,     "annexes.csv")
    csv_benchmarks = _save_csv(df_benchmarks,  "benchmarks.csv")
    csv_summary    = _save_csv(df_summary,     "summary.csv")

    # ── Write summary JSON ────────────────────────────────────────────────────
    summary_json = output_root / "summary.json"
    summary_json.write_text(
        json.dumps(df_summary.to_dict(orient="records"), indent=2, ensure_ascii=False)
    )

    # ── Print report ─────────────────────────────────────────────────────────
    ok_count  = sum(1 for r in all_results if r["status"] == "ok")
    err_count = sum(1 for r in all_results if r["status"] == "error")

    print(f"\n{'='*60}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Files     : {len(all_results)}  ({ok_count} ok, {err_count} errors)")
    print(f"  Time      : {elapsed_total:.0f} s")
    print(f"  Output    : {output_root.resolve()}")
    print(f"\n  DataFrames written:")
    print(f"    boxes.csv       {len(df_boxes):>5} rows")
    print(f"    annexes.csv     {len(df_annexes):>5} rows")
    print(f"    benchmarks.csv  {len(df_benchmarks):>5} rows")
    print(f"    summary.csv     {len(df_summary):>5} rows  (all types combined)")

    if err_count:
        print(f"\n  ERRORS:")
        for r in all_results:
            if r["status"] == "error":
                print(f"    [{r['country']}] {Path(r['pdf']).name}: {r.get('error','')}")

    print(f"\n  PER-COUNTRY BREAKDOWN:")
    by_country: dict[str, list] = {}
    for r in all_results:
        by_country.setdefault(r["country"], []).append(r)
    for country, res in sorted(by_country.items()):
        n_box   = len(df_boxes[df_boxes["country"] == country])   if not df_boxes.empty   else 0
        n_annex = len(df_annexes[df_annexes["country"] == country]) if not df_annexes.empty else 0
        n_sb    = len(df_benchmarks[df_benchmarks["country"] == country]) if not df_benchmarks.empty else 0
        print(f"    {country:<22} {len(res)} file(s)  |  "
              f"{n_box:>3} boxes  {n_annex:>3} annexes  {n_sb:>3} SB rows")
    print()

    # Return DataFrames so the script can also be imported and called directly
    return df_boxes, df_annexes, df_benchmarks, df_summary


if __name__ == "__main__":
    main()
