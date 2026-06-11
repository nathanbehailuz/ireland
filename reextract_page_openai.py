#!/usr/bin/env python3
"""
Throwaway: re-extract one Griffith page with OpenAI using openai_api_key_1 from .env.

Usage:
  python3 reextract_page_openai.py
  python3 reextract_page_openai.py --page IRE_GRIFF_260_147.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Must set key before extract.py reads LLM_HEADERS at import time.
load_dotenv()
_api_key = (os.getenv("openai_api_key_1") or "").strip()
if not _api_key:
    raise SystemExit("openai_api_key_1 is not set in .env")
os.environ["openai_api_key"] = _api_key

from batch_llm_extract import resolve_local_image_path  # noqa: E402
from extract import extract_table_data  # noqa: E402

DEFAULT_PAGE = "IRE_GRIFF_260_147.jpg"
RESULTS_DIR = Path("results/openai")


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-extract one page with OpenAI (openai_api_key_1)")
    parser.add_argument("--page", default=DEFAULT_PAGE, help="Target filename, e.g. IRE_GRIFF_260_147.jpg")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where to write {page}.jpg.json (default: results/openai)",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not backup existing JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    page = args.page.strip()
    if not page.lower().endswith(".jpg"):
        page = f"{page}.jpg"

    image_path = resolve_local_image_path(page)
    if not os.path.isfile(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{page}.json"

    if out_path.is_file() and not args.no_backup:
        backup = out_path.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(out_path, backup)
        logging.info("Backed up existing JSON to %s", backup)

    logging.info("Extracting %s via OpenAI (%s)", page, image_path)
    data = extract_table_data(image_path, "openai")
    if not data:
        logging.error("Extraction failed")
        return 1

    n_entries = sum(len(p.get("entries", [])) for p in data.get("parishes", []))
    logging.info(
        "Got %d parish(es), %d entries (input_tokens=%s, output_tokens=%s)",
        len(data.get("parishes", [])),
        n_entries,
        getattr(extract_table_data, "last_input_tokens", "?"),
        getattr(extract_table_data, "last_output_tokens", "?"),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logging.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
