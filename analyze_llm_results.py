#!/usr/bin/env python3
"""
Merge LLM extraction JSON results, forward-fill townlands, link consecutive pages
via fuzzy boundary matching, join ground-truth admin columns, classify one-page vs
multi-page townlands, and run land/total valuation consistency checks on contiguous
townland_linked groups (county, barony, parish, townland).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import pandas as pd

from results_paths import results_log_file
from table_operations import (
    check_if_correct,
    clean_townland,
    from_total_pence,
    json_to_df,
    to_total_pence,
)
from townland_grouping import (
    forward_fill_townland,
    get_multi_page_townland_ids_linked,
    get_one_page_townland_ids_linked,
    join_gv_columns,
    link_consecutive_page_townlands,
    resolve_gv_parish_fuzzy,
    write_multi_page_townlands_json,
    write_one_page_townlands_json_linked,
)

RESULTS_DIR = "results"
DEFAULT_GV_XLSX = "nathan_to_fix.xlsx"
SUPPORTED_LLMS = ("openai", "gemini")
TOWNLAND_GROUP_SCOPE_COLS = ("county_effective", "barony_effective", "parish_effective")
# Looser than the parish-join threshold: cross-page townland names often differ by
# one OCR'd character (e.g. CARRIVCASHEL vs CARRYCASHEL at 0.87), splitting one
# townland into two segments and stranding its total row.
PAGE_LINK_THRESHOLD = 0.77


def configure_logging(llm: str) -> None:
    log_path = results_log_file(
        "analysis", f"llm_results_{llm}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )


def _coerce_entry_int(value: Any, default: int) -> int:
    """Gemini sometimes returns '' for numeric flags; parquet needs real ints."""
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_json_data(data: dict) -> dict:
    """Ensure parishes/entries exist and default missing numeric flags."""
    if not data or "parishes" not in data:
        return {"parishes": []}
    for parish in data["parishes"]:
        for entry in parish.get("entries", []):
            entry["n_shared"] = _coerce_entry_int(entry.get("n_shared"), 1)
            entry["is_total"] = _coerce_entry_int(entry.get("is_total"), 0)
            entry["is_exemption"] = _coerce_entry_int(entry.get("is_exemption"), 0)
            entry["is_continued"] = _coerce_entry_int(entry.get("is_continued"), 0)
            entry.setdefault("townland", "")
            entry.setdefault("os", "")
            entry.setdefault("sublocation_1", "")
            entry.setdefault("sublocation_2", "")
    return data


def load_combined_extractions(llm: str, results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """Load all results/{llm}/*.jpg.json into one DataFrame in alphabetical page filename order."""
    llm_dir = Path(results_dir) / llm
    if not llm_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {llm_dir}")

    frames: list[pd.DataFrame] = []
    skipped = 0

    for json_path in sorted(llm_dir.glob("*.jpg.json")):
        page_name = json_path.name.replace(".json", "")
        try:
            with open(json_path, encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                logging.warning("Empty JSON file: %s", json_path)
                skipped += 1
                continue
            data = normalize_json_data(json.loads(raw))
        except json.JSONDecodeError as exc:
            logging.warning("Malformed JSON %s: %s", json_path, exc)
            skipped += 1
            continue

        try:
            df_page = json_to_df(data)
        except Exception as exc:
            logging.warning("Failed to flatten %s: %s", json_path, exc)
            skipped += 1
            continue

        if df_page.empty:
            skipped += 1
            continue

        df_page = df_page.copy()
        df_page["page_name"] = page_name
        df_page["target_filename"] = page_name
        df_page["llm"] = llm
        if "townland_raw" not in df_page.columns:
            df_page["townland_raw"] = df_page["townland"]
        df_page["townland_filled"] = forward_fill_townland(df_page)
        frames.append(df_page)

    if not frames:
        raise RuntimeError(f"No valid JSON extractions found under {llm_dir}")

    combined = pd.concat(frames, ignore_index=True)
    logging.info(
        "Loaded %d pages (%d rows); skipped %d files",
        combined["page_name"].nunique(),
        len(combined),
        skipped,
    )
    return combined


def _normalize_townland_key(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def _normalize_scope_value(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def _townland_group_key(row: pd.Series, townland_col: str = "townland_linked") -> tuple[str, ...]:
    """Scope A key: (county, barony, parish, townland_linked). Empty when townland is blank."""
    townland = _normalize_townland_key(row.get(townland_col))
    if not townland:
        return ()
    return tuple(_normalize_scope_value(row.get(col)) for col in TOWNLAND_GROUP_SCOPE_COLS) + (townland,)


def iter_townland_linked_groups(
    df: pd.DataFrame,
    *,
    townland_col: str = "townland_linked",
) -> Iterator[tuple[str, pd.DataFrame]]:
    """
    Yield (townland_label, segment_df) for contiguous runs of the same Scope A key.

    Rows are walked in existing DataFrame order (alphabetical page filenames at load,
    then JSON entry order within each page). Append rows while
    (county_effective, barony_effective, parish_effective, townland_linked) stays
    unchanged; start a new group when any component changes.
    """
    d = df.reset_index(drop=True)
    n = len(d)
    i = 0
    while i < n:
        key = _townland_group_key(d.loc[i], townland_col)
        if not key:
            i += 1
            continue
        label = d.loc[i, townland_col]
        j = i + 1
        while j < n and _townland_group_key(d.loc[j], townland_col) == key:
            j += 1
        yield label, d.iloc[i:j]
        i = j


def run_valuation_checks(df: pd.DataFrame) -> pd.DataFrame:
    """One row per contiguous townland_linked group (Scope A) with sum vs total (factor-of-2)."""
    records: list[dict[str, Any]] = []

    for townland, segment in iter_townland_linked_groups(df):
        is_total = pd.to_numeric(segment["is_total"], errors="coerce").fillna(0)
        total_rows = segment[is_total == 1]
        detail_rows = segment[is_total == 0]
        page_names = sorted(segment["page_name"].unique(), key=str)

        rec: dict[str, Any] = {
            "county_effective": _normalize_scope_value(segment["county_effective"].iloc[0]),
            "barony_effective": _normalize_scope_value(segment["barony_effective"].iloc[0]),
            "parish_effective": _normalize_scope_value(segment["parish_effective"].iloc[0]),
            "townland_linked": townland,
            "page_names": ",".join(page_names),
            "page_count": int(segment["page_name"].nunique()),
            "row_count": len(segment),
            "has_total_row": len(total_rows) > 0,
            "total_row_count": len(total_rows),
            "valuation_ok": False,
            "failure_reason": "",
        }

        if "is_one_page_townland" in segment.columns:
            rec["is_one_page_townland"] = bool(segment["is_one_page_townland"].iloc[0])

        if len(total_rows) == 0:
            rec["failure_reason"] = "no_total_row"
            records.append(rec)
            continue

        total_row = total_rows.iloc[-1]
        if len(total_rows) > 1:
            rec["multiple_total_rows"] = True

        total_land = to_total_pence(total_row["land_val"])
        total_total = to_total_pence(total_row["total_val"])
        sum_land = sum(to_total_pence(v) for v in detail_rows["land_val"])
        sum_total = sum(to_total_pence(v) for v in detail_rows["total_val"])

        rec["sum_land_val"] = from_total_pence(sum_land)
        rec["total_land_val"] = from_total_pence(total_land)
        rec["sum_total_val"] = from_total_pence(sum_total)
        rec["total_total_val"] = from_total_pence(total_total)
        rec["valuation_ok"] = check_if_correct(total_land, total_total, sum_land, sum_total)
        records.append(rec)

    return pd.DataFrame(records)


def _find_unflagged_total(segment: pd.DataFrame) -> dict[str, Any] | None:
    """
    Look for a row that behaves like a total (its total_val is within the pipeline's
    factor-of-2 tolerance of the sum of all other rows) but has is_total=0.
    Returns detail about the best candidate, or None.
    """
    if len(segment) < 3:
        return None

    pence = [to_total_pence(v) for v in segment["total_val"]]
    total_sum = sum(pence)
    best: tuple[int, int, int, int] | None = None

    for pos, value in enumerate(pence):
        rest = total_sum - value
        if value <= 0 or rest <= 0:
            continue
        if value <= 2 * rest and rest <= 2 * value:
            distance = abs(value - rest)
            if best is None or distance < best[0]:
                best = (distance, pos, value, rest)

    if best is None:
        return None

    _, pos, value, rest = best
    row = segment.iloc[pos]
    return {
        "candidate_row_pos": pos,
        "candidate_is_last_row": pos == len(segment) - 1,
        "candidate_occupier": str(row.get("occupier", "")),
        "candidate_total_val": from_total_pence(value),
        "sum_other_rows_total_val": from_total_pence(rest),
    }


def build_no_total_row_report(
    df: pd.DataFrame,
    *,
    neighbor_similarity_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    One row per no_total_row segment, classified by likely root cause:

    - unflagged_total:   a row in the segment sums like a total but is_total=0
                         (LLM extracted the total row without flagging it)
    - split_segment:     an adjacent segment in the same county/barony/parish has a
                         fuzzy-similar townland name and owns a total row (the
                         contiguous grouping likely split one townland in two)
    - tiny_segment:      <=2 rows; likely a spurious segment from townland-name noise
    - genuinely_missing: none of the above; total may be absent from the page or
                         the extraction missed it entirely
    """
    segments: list[dict[str, Any]] = []
    for townland, segment in iter_townland_linked_groups(df):
        is_total = pd.to_numeric(segment["is_total"], errors="coerce").fillna(0)
        segments.append(
            {
                "townland": str(townland),
                "scope": tuple(
                    _normalize_scope_value(segment[c].iloc[0]) for c in TOWNLAND_GROUP_SCOPE_COLS
                ),
                "segment": segment,
                "total_count": int((is_total == 1).sum()),
            }
        )

    records: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if seg["total_count"] > 0:
            continue
        segment = seg["segment"]
        page_names = sorted(segment["page_name"].unique(), key=str)

        rec: dict[str, Any] = {
            "county_effective": seg["scope"][0],
            "barony_effective": seg["scope"][1],
            "parish_effective": seg["scope"][2],
            "townland_linked": seg["townland"],
            "page_names": ",".join(page_names),
            "row_count": len(segment),
            "category": "genuinely_missing",
            "neighbor_direction": "",
            "neighbor_townland": "",
            "neighbor_similarity": "",
            "neighbor_total_rows": "",
            "candidate_row_pos": "",
            "candidate_is_last_row": "",
            "candidate_occupier": "",
            "candidate_total_val": "",
            "sum_other_rows_total_val": "",
        }

        unflagged = _find_unflagged_total(segment)
        if unflagged is not None:
            rec["category"] = "unflagged_total"
            rec.update(unflagged)
            records.append(rec)
            continue

        best_neighbor: dict[str, Any] | None = None
        for direction, j in (("prev", idx - 1), ("next", idx + 1)):
            if not (0 <= j < len(segments)):
                continue
            neighbor = segments[j]
            if neighbor["scope"] != seg["scope"] or neighbor["total_count"] == 0:
                continue
            similarity = SequenceMatcher(
                None, clean_townland(seg["townland"]), clean_townland(neighbor["townland"])
            ).ratio()
            if similarity >= neighbor_similarity_threshold and (
                best_neighbor is None or similarity > best_neighbor["neighbor_similarity"]
            ):
                best_neighbor = {
                    "neighbor_direction": direction,
                    "neighbor_townland": neighbor["townland"],
                    "neighbor_similarity": round(similarity, 3),
                    "neighbor_total_rows": neighbor["total_count"],
                }

        if best_neighbor is not None:
            rec["category"] = "split_segment"
            rec.update(best_neighbor)
        elif len(segment) <= 2:
            rec["category"] = "tiny_segment"

        records.append(rec)

    return pd.DataFrame(records)


def add_one_page_flags(df: pd.DataFrame, one_page_ids: pd.Index) -> pd.DataFrame:
    """Add is_one_page_townland column from (GV admin, townland_linked) group ids."""
    df = df.copy()
    one_page_set = set(
        one_page_ids.tolist() if isinstance(one_page_ids, pd.MultiIndex) else list(one_page_ids)
    )
    keys = list(
        zip(
            df["county_effective"],
            df["barony_effective"],
            df["parish_effective"],
            df["townland_linked"],
        )
    )
    df["is_one_page_townland"] = [k in one_page_set for k in keys]
    return df


def build_performance_summary(
    df: pd.DataFrame,
    valuation_df: pd.DataFrame,
    llm: str,
) -> dict[str, Any]:
    """Aggregate metrics for performance_summary.json."""
    with_total = valuation_df[valuation_df["has_total_row"]]
    passed = with_total[with_total["valuation_ok"]]

    def rate(num: int, den: int) -> float:
        return round(num / den * 100.0, 2) if den else 0.0

    summary: dict[str, Any] = {
        "llm": llm,
        "pages_processed": int(df["page_name"].nunique()),
        "entry_rows": int(len(df)),
        "townland_segments": int(len(valuation_df)),
        "segments_with_total_row": int(len(with_total)),
        "segments_with_total_row_pct": rate(len(with_total), len(valuation_df)),
        "valuation_pass_count": int(len(passed)),
        "valuation_pass_pct": rate(len(passed), len(with_total)),
    }

    if "is_one_page_townland" in valuation_df.columns:
        for label, subset in [
            ("one_page_townlands", valuation_df[valuation_df["is_one_page_townland"]]),
            ("multi_page_townlands", valuation_df[~valuation_df["is_one_page_townland"]]),
        ]:
            sub_with = subset[subset["has_total_row"]]
            sub_pass = sub_with[sub_with["valuation_ok"]]
            summary[f"{label}_segments"] = int(len(subset))
            summary[f"{label}_valuation_pass_pct"] = rate(len(sub_pass), len(sub_with))

    return summary


def save_dataframe(df: pd.DataFrame, base_path: Path) -> None:
    """Write parquet if available, always write CSV."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logging.info("Wrote %s", csv_path)

    parquet_path = base_path.with_suffix(".parquet")
    try:
        out = df.copy()
        for col in ("n_shared", "is_total", "is_exemption", "is_continued"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int64")
        out.to_parquet(parquet_path, index=False)
        logging.info("Wrote %s", parquet_path)
    except ImportError:
        logging.warning(
            "Parquet engine unavailable (install pyarrow); wrote CSV only at %s", csv_path
        )


def run_pipeline(
    llm: str,
    *,
    gv_xlsx: str = DEFAULT_GV_XLSX,
    threshold: float = 0.9,
    results_dir: str = RESULTS_DIR,
) -> dict[str, Any]:
    """Execute full analysis for one LLM; return performance summary dict."""
    out_dir = Path(results_dir) / llm

    df = load_combined_extractions(llm, results_dir)

    if not os.path.isfile(gv_xlsx):
        raise FileNotFoundError(f"Ground truth file not found: {gv_xlsx}")

    gv_df = pd.read_excel(gv_xlsx)
    df = join_gv_columns(df, gv_df)
    df, review_df = resolve_gv_parish_fuzzy(
        df,
        gv_df,
        threshold=threshold,
        review_path=str(out_dir / "gv_join_review.csv"),
        overrides_path=str(out_dir / "gv_join_overrides.csv"),
    )
    logging.info("GV join review rows: %d", len(review_df))
    df = link_consecutive_page_townlands(df, threshold=PAGE_LINK_THRESHOLD, inplace=True)

    df_gv_subset = df[
        df["county_effective"].astype(str).str.strip().replace("nan", "").ne("")
        & df["townland_linked"].astype(str).str.strip().ne("")
    ].copy()

    if not df_gv_subset.empty:
        one_page_ids = get_one_page_townland_ids_linked(df_gv_subset)
        multi_page_ids = get_multi_page_townland_ids_linked(df_gv_subset)
        df = add_one_page_flags(df, one_page_ids)
        write_one_page_townlands_json_linked(
            df_gv_subset,
            one_page_ids,
            output_path=str(out_dir / "one_page_townlands.json"),
        )
        write_multi_page_townlands_json(
            df_gv_subset,
            multi_page_ids,
            output_path=str(out_dir / "multi_page_townlands.json"),
        )
        logging.info(
            "Townland groups: %d one-page, %d multi-page",
            len(one_page_ids),
            len(multi_page_ids),
        )
    else:
        df["is_one_page_townland"] = False
        logging.warning("No GV-joined rows; skipping one-page townland detection")

    save_dataframe(df, out_dir / "combined_extractions")

    valuation_df = run_valuation_checks(df)
    valuation_path = out_dir / "townland_valuation_checks.csv"
    valuation_df.to_csv(valuation_path, index=False)
    logging.info("Wrote %s (%d segments)", valuation_path, len(valuation_df))

    no_total_df = build_no_total_row_report(df)
    no_total_path = out_dir / "no_total_row_debug.csv"
    no_total_df.to_csv(no_total_path, index=False)
    category_counts = (
        no_total_df["category"].value_counts().to_dict() if not no_total_df.empty else {}
    )
    logging.info(
        "Wrote %s (%d no-total segments): %s", no_total_path, len(no_total_df), category_counts
    )

    summary = build_performance_summary(df, valuation_df, llm)
    summary_path = out_dir / "performance_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logging.info("Wrote %s", summary_path)
    logging.info("Summary: %s", json.dumps(summary, indent=2))

    return summary


def print_cross_model_comparison(results_dir: str = RESULTS_DIR) -> None:
    """Print overlap stats when both models have performance summaries."""
    summaries: dict[str, dict] = {}
    pages_by_model: dict[str, set[str]] = {}

    for llm in SUPPORTED_LLMS:
        path = Path(results_dir) / llm / "performance_summary.json"
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                summaries[llm] = json.load(f)
        val_path = Path(results_dir) / llm / "townland_valuation_checks.csv"
        if val_path.is_file():
            vdf = pd.read_csv(val_path)
            pages: set[str] = set()
            for page_list in vdf["page_names"].dropna():
                pages.update(p.strip() for p in str(page_list).split(",") if p.strip())
            pages_by_model[llm] = pages

    if len(summaries) < 2:
        return

    common = pages_by_model.get("gemini", set()) & pages_by_model.get("openai", set())
    print("\nCross-model overlap (pages with valuation checks in both):")
    print(f"  Common pages: {len(common)}")
    for llm in SUPPORTED_LLMS:
        if llm in summaries:
            s = summaries[llm]
            print(
                f"  {llm}: valuation_pass_pct={s.get('valuation_pass_pct')}% "
                f"({s.get('valuation_pass_count')}/{s.get('segments_with_total_row')} segments with total row)"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze LLM extraction results")
    parser.add_argument(
        "--llm",
        choices=SUPPORTED_LLMS,
        help="Model subdirectory under results/ (omit with --compare-models to only compare)",
    )
    parser.add_argument("--gv-xlsx", default=DEFAULT_GV_XLSX, help="Ground truth Excel path")
    parser.add_argument("--threshold", type=float, default=0.9, help="Fuzzy match threshold")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Print cross-model overlap from existing summaries (after the run if --llm is given)",
    )
    args = parser.parse_args()

    if args.llm is None and not args.compare_models:
        parser.error("--llm is required unless --compare-models is given")

    if args.llm is not None:
        configure_logging(args.llm)
        run_pipeline(
            args.llm,
            gv_xlsx=args.gv_xlsx,
            threshold=args.threshold,
            results_dir=args.results_dir,
        )
    if args.compare_models:
        print_cross_model_comparison(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
