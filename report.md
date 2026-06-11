# Griffith's Valuation Extraction — Project Report

*Nathan Behailu — June 2026*

## 1. Introduction & objective

Griffith's Valuation is a mid-19th-century survey of Irish land and property, printed as tabular records listing, for each townland, the occupiers, the immediate lessors, a description of the tenement, and valuations in pounds/shillings/pence, closed by a printed total line per townland.

The goal of this project is to turn scanned pages of these tables into structured, validated data. The corpus is **1,904 scanned pages** (named `IRE_GRIFF_{volume}_{page}.jpg`), defined by the `target_filename` column of `nathan_to_fix.xlsx` — the project's ground-truth spreadsheet, which holds 5,832 townland-level rows with authoritative county/barony/parish metadata (2,507 of which are flagged with `check_townland == 1` for townland-name review).

The work breaks into three parts, covered in the sections below:

1. An extraction pipeline that converts page images into per-page JSON via vision LLMs (Section 2).
2. An analysis pipeline that merges the JSON into a single enriched dataset and validates it (Section 3).
3. Results and a quality deep-dive into the remaining failure modes (Sections 4–5).

## 2. Pipeline: images → LLM extraction → JSON

The source scans live locally under `Nanonets/analysis/{volume}/IRE_GRIFF_*.jpg` (one subfolder per volume; the images are not tracked in git). The set of pages to process is driven by `nathan_to_fix.xlsx`: `build_batch_csv.py` reads its `target_filename` column and produces the batch job list.

Extraction is done with **vision LLMs rather than traditional OCR**. The flow is:

1. `upload_images_to_r2.py` uploads the page images to Cloudflare R2 so the batch APIs can reference them by URL.
2. `batch_llm_extract.py` orchestrates the batch jobs end to end (prepare → submit → poll → finalize), using the shared extraction prompt and JSON normalization in `extract.py`.
3. Two models were run over the full corpus:
   - **OpenAI** `gpt-5.2`
   - **Gemini** `gemini-3-pro-image-preview`

The output is one JSON file per page — `results/openai/*.jpg.json` and `results/gemini/*.jpg.json`, 1,904 files each. Each file contains the page's parishes and their entry rows: townland, occupier, lessor, the £/s/d valuation fields, and flags such as `is_total` (the row is a printed total line) and `is_continued` (the townland continues from a previous page).

## 3. Pipeline: JSON → combined dataframes & validation

The analysis entry point is `analyze_llm_results.py` (run as `python3 analyze_llm_results.py --llm openai|gemini`), supported by `table_operations.py` and `townland_grouping.py`. It runs the following steps:

1. **Flatten** every page JSON into table rows (`json_to_df` in `table_operations.py`), converting £/s/d valuations to pence and concatenating all pages in page order.
2. **Forward-fill continuation rows**: entries with a blank townland inherit the townland printed above them on the same page and parish.
3. **Join ground-truth admin metadata** from `nathan_to_fix.xlsx`, attaching the authoritative county, barony, and parish to each row by page + parish match.
4. **Fuzzy parish resolution**: when the LLM's parish spelling does not match the spreadsheet exactly, fuzzy matching resolves it; ambiguous cases are exported to `gv_join_review.csv` for manual review (zero unresolved cases in the latest run).
5. **Link townlands across consecutive pages**: a townland's occupier list often runs onto the next page, so the last townland of one page is fuzzy-matched against the first townland of the next (within the same county/barony/parish), producing a unified `townland_linked` name.
6. **Classify one-page vs multi-page townlands**, written to `one_page_townlands.json` and `multi_page_townlands.json`.

The fully enriched table is saved as **`combined_extractions.csv` / `.parquet`** — the main merged output of the project.

On top of it, the script builds **townland segments** (contiguous runs of rows sharing the same county, barony, parish, and linked townland name) and runs a **valuation consistency check** per segment: the land and total valuations of all detail rows are summed and compared against the printed total row, with a factor-of-two tolerance (`check_if_correct()`) to absorb £/s/d rounding and OCR noise. Per-segment results go to `townland_valuation_checks.csv`, and aggregate metrics to `performance_summary.json`.

## 4. Extraction results — coverage & model comparison

Both models processed **100% of the corpus** (1,904/1,904 pages). The headline metrics below come from each model's `performance_summary.json` (latest run):

| Metric | What it records | OpenAI | Gemini |
|--------|-----------------|--------|--------|
| `pages_processed` | Pages with at least one valid JSON extraction | 1,904 | 1,904 |
| `entry_rows` | Total table rows extracted across all pages (occupiers, totals, exemptions) | 90,667 | 98,790 |
| `townland_segments` | Contiguous runs of rows sharing the same county, barony, parish, and linked townland name — the unit every check runs on | 4,133 | 3,800 |
| `segments_with_total_row` | Segments containing at least one row flagged `is_total=1` | 3,157 | 3,154 |
| `segments_with_total_row_pct` | Share of all segments that have a total row (denominator = `townland_segments`) | 76.4% | 83.0% |
| `valuation_pass_count` | Segments where summed detail rows match the printed total row within a factor-of-2 tolerance, for both land and total valuation | 2,621 | 2,753 |
| `valuation_pass_pct` | Pass share among segments *with* a total row (denominator = `segments_with_total_row`) | 83.0% | 87.3% |
| `one_page_townlands_segments` | Segments belonging to townlands confined to a single page | 3,184 | 2,780 |
| `one_page_townlands_valuation_pass_pct` | Valuation pass rate within one-page townlands | 85.2% | 86.7% |
| `multi_page_townlands_segments` | Segments belonging to townlands spanning two or more pages | 949 | 1,020 |
| `multi_page_townlands_valuation_pass_pct` | Valuation pass rate within multi-page townlands | 75.2% | 88.9% |

Two observations stand out:

- **Volume:** Gemini extracts roughly 8k more rows than OpenAI from the same pages, indicating more complete row capture.
- **Takeaway:** Gemini is stronger on both total-row detection and valuation consistency, and the gap is largest on multi-page townlands (88.9% vs 75.2% pass rate) — the hardest case, since detail rows and the total line sit on different scans.

## 5. Quality deep-dive: totals, valuations, and failure modes

Two distinct gaps drive the quality losses, and they should not be conflated:

- **Gap A — segments with no total row at all:** 976 segments (OpenAI, 23.6%) and 646 (Gemini, 17.0%) cannot be checked because no row in the segment is flagged `is_total=1`.
- **Gap B — segments with a total that fails arithmetic:** having a total row does not guarantee passing the check. 536 OpenAI segments and 401 Gemini segments fail the valuation comparison despite having a total row.

### 5.1 Failure taxonomy for Gap A (`no_total_row_debug.csv`)

Every no-total segment is automatically classified by likely root cause:

| Category | OpenAI | Gemini | Meaning |
|----------|--------|--------|---------|
| `genuinely_missing` | 588 | 346 | No total candidate found in the segment (sub-broken-down in 5.3) |
| `unflagged_total` | 242 | 175 | A row's `total_val` matches the sum of all other rows — the LLM extracted the printed total line but left `is_total=0` |
| `tiny_segment` | 137 | 109 | ≤2 rows; likely a spurious segment created by townland-name noise |
| `split_segment` | 9 | 16 | An adjacent segment with a fuzzy-similar name owns the total — grouping split one townland in two |

### 5.2 Open issues

1. **Unflagged totals — 242 (OpenAI) / 175 (Gemini) segments.** The printed total line is extracted but left with `is_total=0`. Candidate rows are detected by sum-matching and exported for manual QA in `unflagged_total_checklist.csv`, tiered by confidence (last-row-of-segment with an empty occupier field first); spot-checks against the page images confirm the diagnosis. Still to do: a prompt tweak or a post-processing auto-repair that sets `is_total=1` on confirmed candidates.
2. **Valuation arithmetic failures (Gap B) — 536 (OpenAI) / 401 (Gemini) segments.** The total row exists but the summed detail rows don't match it within the factor-of-2 tolerance, pointing to misread digits or dropped rows. These need triage via `townland_valuation_checks.csv`, likely combining cross-model comparison with targeted re-extraction (`reextract_page_openai.py`).
3. **Tiny segments — 137 (OpenAI) / 109 (Gemini).** Segments of ≤2 rows created by townland-name noise; the grouping logic needs cleanup or these need to be merged back into their neighbours.
4. **Residual `split_segment` rows — 9 (OpenAI) / 16 (Gemini).** Mostly directional pairs (`GRANGE WEST`/`EAST`, `KNOCKDUFF UPPER`/`LOWER`, `TIRNAHINCH NEAR`/`FAR`) — genuinely different townlands that the heuristic flags only because their names are similar; the linker correctly does *not* merge them, so most of these are false alarms rather than errors. Listed in `split_segment_checklist.csv` for confirmation.

### 5.3 Sub-breakdown of `genuinely_missing` (`genuinely_missing_debug.csv`)

Cross-referencing each segment's position on the page, the corpus page coverage, and the other model's output shows that most of this bucket is **corpus structure, not LLM error**:

| Subcategory | OpenAI | Gemini | Meaning |
|-------------|--------|--------|---------|
| `continues_to_unscanned_page` | 186 | 210 | The townland runs off the page bottom and the next page is **not in the 1,904-page corpus** — the total is printed on an unscanned page |
| `ends_at_page_bottom_next_scanned` | 169 | 71 | Runs off the page bottom; the next page is scanned but linking did not connect (a mix of real townland changes and near-miss names like `TULLYGLUSH (NEVIN)` → `TULLYGLUSH`) |
| `extraction_miss_other_model_has_total` | 126 | 40 | Mid-page segment where the **other model found the total** — the strongest evidence of a true extraction miss |
| `missing_in_both_models_mid_page` | 107 | 25 | Neither model has a total; the page likely prints none for that townland (common for urban street listings) |

The cross-model comparison gives a useful triage principle: a townland missing its total in *both* models points to segmentation or the page itself, while a total missing in only one model points to that model's extraction. The actionable buckets are the extraction misses (126 OpenAI / 40 Gemini, recoverable via the other model or re-extraction) and the near-miss page links in `ends_at_page_bottom_next_scanned`; the unscanned-page cases are a corpus limitation, not an extraction error.

## 6. What was built

- **Batch extraction infrastructure**: R2 image upload (`upload_images_to_r2.py`), batch job construction (`build_batch_csv.py`), and full batch lifecycle orchestration for both the OpenAI and Gemini batch APIs (`batch_llm_extract.py`, `extract.py`).
- **Post-processing & validation pipeline**: `analyze_llm_results.py` with `table_operations.py` and `townland_grouping.py` — merge, ground-truth join, fuzzy resolution, cross-page townland linking, segment grouping, and valuation consistency checks.
- **Dual-model run and comparison** over the full 1,904-page corpus, with per-model metrics and a cross-model triage of failures.
- **Automated QA artifacts**: `no_total_row_debug.csv`, `genuinely_missing_debug.csv`, `unflagged_total_checklist.csv`, `townland_valuation_checks.csv`, and `performance_summary.json` per model.
- **Documentation**: `README.md` describing the full analysis pipeline step by step.

## 7. Appendix

### Key files

| Purpose | Path |
|---------|------|
| Ground truth / page list | `nathan_to_fix.xlsx` |
| Source images | `Nanonets/analysis/{volume}/IRE_GRIFF_*.jpg` |
| Extraction prompt & API logic | `extract.py` |
| Batch orchestration | `batch_llm_extract.py` |
| Analysis entry point | `analyze_llm_results.py` |
| Per-page JSON results | `results/{openai,gemini}/*.jpg.json` |
| Merged dataset | `results/{openai,gemini}/combined_extractions.csv` / `.parquet` |
| Per-segment valuation checks | `results/{openai,gemini}/townland_valuation_checks.csv` |
| Aggregate metrics | `results/{openai,gemini}/performance_summary.json` |
| No-total root-cause taxonomy | `results/{openai,gemini}/no_total_row_debug.csv` |
| Genuinely-missing sub-breakdown | `results/{openai,gemini}/genuinely_missing_debug.csv` |
| Manual QA checklists | `results/{openai,gemini}/unflagged_total_checklist.csv` |

### Reproducing the analysis

```bash
python3 analyze_llm_results.py --llm openai
python3 analyze_llm_results.py --llm gemini
python3 analyze_llm_results.py --compare-models
```
