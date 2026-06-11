#!/usr/bin/env python3
"""
Build a CSV with column `target_filename` for use with:

  python batch_llm_extract.py prepare --csv <this_file> ...

Values are basenames (e.g. IRE_GRIFF_004_065.jpg). Exactly one --from-* source is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

_GRIFF_BASENAME = re.compile(
    r"^IRE_GRIFF_\d+_\d+\.jpe?g$", re.IGNORECASE
)


def normalize_to_basename(entry: str) -> str:
    entry = entry.strip()
    if not entry:
        return ""
    return os.path.basename(entry.replace("/", os.sep))


def collect_from_json(path: Path) -> List[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                out.append(normalize_to_basename(item))
            elif isinstance(item, dict):
                tf = item.get("target_filename")
                if tf is not None:
                    out.append(normalize_to_basename(str(tf)))
    elif isinstance(raw, dict) and "target_filename" in raw:
        out.append(normalize_to_basename(str(raw["target_filename"])))
    else:
        raise SystemExit(f"Unsupported JSON shape in {path}")
    return [x for x in out if x]


def collect_from_list(path: Path) -> List[str]:
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(normalize_to_basename(line))
    return [x for x in out if x]


def collect_from_scan(root: Path) -> List[str]:
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    out: List[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        out.append(p.name)
    return out


def collect_from_table(path: Path, column: Optional[str]) -> List[str]:
    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
    elif suf in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise SystemExit(f"Unsupported table format: {path}")

    col = column
    if not col:
        if "target_filename" in df.columns:
            col = "target_filename"
        else:
            raise SystemExit(
                f"No column specified and 'target_filename' not in columns: {list(df.columns)}"
            )

    if col not in df.columns:
        raise SystemExit(f"Column not found: {col!r}. Available: {list(df.columns)}")

    out: List[str] = []
    for v in df[col].dropna():
        out.append(normalize_to_basename(str(v)))
    return [x for x in out if x]


def apply_validate(names: Iterable[str], skip_invalid: bool) -> List[str]:
    kept: List[str] = []
    for n in names:
        if _GRIFF_BASENAME.match(n):
            kept.append(n)
        else:
            msg = f"Does not match IRE_GRIFF_*_*.jpg pattern: {n!r}"
            if skip_invalid:
                logging.warning("Skipping %s", msg)
            else:
                logging.warning(msg)
            if not skip_invalid:
                kept.append(n)
    return kept


def finalize_names(names: List[str], dedupe: bool) -> List[str]:
    if dedupe:
        names = list(dict.fromkeys(names))
        return sorted(names)
    return list(names)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-json", type=Path, metavar="PATH")
    src.add_argument("--from-list", type=Path, metavar="PATH")
    src.add_argument(
        "--scan-root",
        nargs="?",
        const=Path("Nanonets/analysis"),
        default=None,
        type=Path,
        metavar="PATH",
        help="Recurse for *.jpg; PATH defaults to Nanonets/analysis if omitted",
    )
    src.add_argument("--from-csv", type=Path, metavar="PATH")

    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    p.add_argument(
        "--column",
        default=None,
        help="Column name for --from-csv / table (default: target_filename if present)",
    )
    p.add_argument(
        "--dedupe",
        dest="dedupe",
        action="store_true",
        default=True,
        help="Unique basenames (default)",
    )
    p.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="Keep duplicate basenames in order",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Warn on basenames not matching IRE_GRIFF_*_*.jpg",
    )
    p.add_argument(
        "--skip-invalid",
        action="store_true",
        help="With --validate, drop non-matching basenames instead of keeping them",
    )

    args = p.parse_args()

    names: List[str] = []
    if args.from_json:
        names = collect_from_json(args.from_json)
    elif args.from_list:
        names = collect_from_list(args.from_list)
    elif args.scan_root is not None:
        root = args.scan_root
        names = collect_from_scan(root)
    elif args.from_csv:
        names = collect_from_table(args.from_csv, args.column)

    if not names:
        raise SystemExit("No filenames collected from source.")

    if args.validate:
        names = apply_validate(names, skip_invalid=args.skip_invalid)

    names = finalize_names(names, dedupe=args.dedupe)

    out = pd.DataFrame({"target_filename": names})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")
    logging.info("Wrote %s rows to %s", len(names), args.output)


if __name__ == "__main__":
    main()
