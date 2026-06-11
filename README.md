# ireland

Griffith's Valuation extraction and analysis for Irish townland records.

## `analyze_llm_results.py`

This script takes the per-page JSON files produced by an LLM extractor (OpenAI or Gemini) and runs a full analysis pipeline: merge, clean, join to ground truth, link townlands across pages, and check whether extracted valuations are internally consistent.

### How to run

```bash
python3 analyze_llm_results.py --llm openai
python3 analyze_llm_results.py --llm gemini
python3 analyze_llm_results.py --compare-models
```

Input lives under `results/{llm}/*.jpg.json`. Ground-truth admin data comes from `nathan_to_fix.xlsx` by default.

---

### Step-by-step pipeline

#### 1. Load and merge all page extractions

The script reads every JSON file for the chosen model from `results/{llm}/`. Each file is one scanned page of Griffith's Valuation tables.

- Empty or malformed files are skipped and logged.
- Each page's nested parish/entry structure is flattened into table rows (occupier, land value, total value, flags, etc.).
- Rows with blank townland names on continuation lines are forward-filled from the townland printed above them within the same page and parish.
- All pages are concatenated into a single table in alphabetical page filename order, preserving JSON entry order within each page.

At this point each row has what the LLM read from the manuscript, plus a filled-in townland name for continuation rows.

#### 2. Join ground-truth administrative data

Each row is matched to the ground-truth spreadsheet (`nathan_to_fix.xlsx`) to attach county, barony, and parish from the authoritative record.

- The primary match is page filename + parish name as extracted by the LLM.
- If that fails, the script falls back to a page-level admin record.
- Rows that still cannot be matched are flagged; their admin columns stay empty.

The joined admin values are stored as the "effective" county, barony, and parish used for all later grouping.

#### 3. Resolve uncertain parish matches

When the LLM's parish spelling does not match the spreadsheet exactly, the script tries fuzzy matching against all parish candidates known for that page.

- Matches above the similarity threshold are applied automatically.
- Ambiguous or low-confidence cases are written to `gv_join_review.csv` for manual review.
- Manual corrections can be supplied via `gv_join_overrides.csv`.

#### 4. Link townlands across consecutive pages

A townland's occupier list often continues onto the next scanned page. The script compares the last townland name on one page with the first townland name on the next page (in page order).

- If the names are similar enough (fuzzy match) **and** both rows share the same county, barony, and parish, the leading rows on the next page are treated as the same townland.
- This produces a unified `townland_linked` name that can span multiple pages within one parish.

#### 5. Classify one-page vs multi-page townlands

Using the linked townland names and ground-truth admin, the script identifies which townlands appear on exactly one page versus those that span two or more pages.

- Each row is tagged as belonging to a one-page or multi-page townland.
- Two JSON inventories are written: `one_page_townlands.json` and `multi_page_townlands.json`.

#### 6. Save the merged dataset

The fully enriched table — all extraction rows plus admin columns, linked townland names, and one-page flags — is saved as:

- `combined_extractions.csv`
- `combined_extractions.parquet` (when pyarrow is available)

This is the main merged output of the pipeline.

#### 7. Group rows for valuation checks

Rows are already in page order: JSON files are loaded in alphabetical filename order (which matches increasing page number), with within-page rows in extraction order. The valuation walk uses that order directly — no additional re-sort by county, barony, or parish.

The script walks through the table and builds **contiguous townland groups**: consecutive rows that share the same county, barony, parish, and `townland_linked` name. If the same townland name appears again later after a different townland in between, that starts a new group.

Groups can span multiple pages when a townland continues across scans.

#### 8. Run valuation consistency checks

For each townland group:

- **Detail rows** are all entries that are not total/summary lines.
- **Total row** is the last summary line in the group (if a townland has intermediate totals on continuation pages, only the final one is used).
- The script sums land and total values from all detail rows across all pages in the group.
- Those sums are compared to the values on the total row.
- A group **passes** if both land and total are within a factor of two of each other (tolerance for pounds/shillings/pence rounding and OCR noise).
- Groups with no total row are recorded as failures with reason `no_total_row`.

Results are written to `townland_valuation_checks.csv`, one row per townland group.

#### 9. Build performance summary

Aggregate metrics are computed and saved to `performance_summary.json`, including:

- Pages and rows processed
- Number of townland groups checked
- How many groups had a total row, and what share passed the valuation check
- Separate pass rates for one-page vs multi-page townlands

#### 10. Optional cross-model comparison

With `--compare-models`, the script prints overlap and pass-rate stats when both OpenAI and Gemini summaries exist. It can run standalone (no `--llm` needed, reads the existing summaries) or be combined with `--llm` to compare right after a run.

---

### Output files (per model)

| File | Description |
|------|-------------|
| `combined_extractions.csv` / `.parquet` | Merged, enriched extraction table |
| `townland_valuation_checks.csv` | Per-townland-group valuation pass/fail |
| `performance_summary.json` | Aggregate metrics |
| `one_page_townlands.json` | Townlands confined to a single page |
| `multi_page_townlands.json` | Townlands spanning multiple pages |
| `gv_join_review.csv` | Parish matches needing manual review |
| `gv_join_overrides.csv` | Manual admin overrides (optional input) |

Logs are written under `results/log/analysis/`.
