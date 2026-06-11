"""
Fuzzy grouping for historical townland records.

Normalizes townland strings, builds canonical IDs via fuzzy matching (threshold 0.9)
scoped to (county, barony, parish), and provides drop-in replacements for
exact-match grouping so one-page townlands and pages_to_check are computed robustly.

Usage:
    from townland_grouping import add_canonical_townland, get_one_page_townland_ids, get_pages_to_check, write_one_page_townlands_json

    df_with_canonical = add_canonical_townland(df_filtered, threshold=0.9)
    one_page_ids = get_one_page_townland_ids(df_with_canonical)
    pages_to_check = get_pages_to_check(df_with_canonical, one_page_ids)
    write_one_page_townlands_json(df_with_canonical, one_page_ids, output_path="one_page_townlands.json")
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

try:
    from rapidfuzz import fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    from difflib import SequenceMatcher
    _RAPIDFUZZ_AVAILABLE = False
# Optional: pip install rapidfuzz for faster fuzzy matching; otherwise difflib is used.

# Sentinel for missing admin/townland (so groupby does not drop rows)
_NA_SENTINEL = ""

# Required columns for grouping
_REQUIRED_COLS = ["county_gv", "barony_gv", "parish_gv", "townland_gv", "target_filename"]


def _similarity(s1: str, s2: str) -> float:
    """Return similarity in [0, 100] for rapidfuzz or [0, 1] for SequenceMatcher."""
    if _RAPIDFUZZ_AVAILABLE:
        return fuzz.ratio(s1, s2)
    return SequenceMatcher(None, s1, s2).ratio() * 100.0


def normalize_townland(s: Union[str, float, None]) -> str:
    """
    Normalize one townland string: case, punctuation, whitespace, apostrophes.

    Handles NaN/None by returning _NA_SENTINEL so groupby does not drop rows.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return _NA_SENTINEL
    s = str(s).strip()
    if not s:
        return _NA_SENTINEL
    s = s.lower()
    # Apostrophe variants -> empty (Waller's Lot vs Wallers Lot)
    for ap in ("'", "'", "'", "`", "'"):
        s = s.replace(ap, "")
    # Remove punctuation that often varies
    s = re.sub(r"[.,;:]+", "", s)
    # Hyphen -> remove so Bally-More ≈ Ballymore
    s = s.replace("-", "")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else _NA_SENTINEL


def normalize_admin(s: Union[str, float, None]) -> str:
    """
    Normalize county/barony/parish for consistent grouping keys.
    Same rules as townland but used only when building canonical map.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return _NA_SENTINEL
    s = str(s).strip()
    if not s:
        return _NA_SENTINEL
    s = s.lower()
    for ap in ("'", "'", "'", "`", "'"):
        s = s.replace(ap, "")
    s = re.sub(r"[.,;:]+", "", s)
    s = s.replace("-", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else _NA_SENTINEL


def normalize_llm_parish(s: Union[str, float, None]) -> str:
    """Normalize LLM parish labels for matching (strip PARISH OF, part-of suffixes)."""
    norm = normalize_admin(s)
    if norm == _NA_SENTINEL:
        return norm
    if norm.startswith("parish of "):
        norm = norm[10:].strip()
    elif norm.startswith("parish "):
        norm = norm[7:].strip()
    for suffix in (" (part of)", " (part of.)", " part of", "—continued", " continued"):
        if norm.endswith(suffix):
            norm = norm[: -len(suffix)].strip()
    return norm if norm else _NA_SENTINEL


def _admin_nonempty(val: Any) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip().lower() not in ("", "nan")


def _coalesce_field(gv_val: Any, raw_val: Any) -> str:
    """Prefer _gv column; fall back to county/barony/parish when _gv is NaN."""
    if _admin_nonempty(gv_val):
        return str(gv_val).strip()
    if _admin_nonempty(raw_val):
        return str(raw_val).strip()
    return ""


def _gv_record_from_row(row: pd.Series) -> dict[str, str]:
    county_e = _coalesce_field(row.get("county_gv"), row.get("county"))
    barony_e = _coalesce_field(row.get("barony_gv"), row.get("barony"))
    parish_e = _coalesce_field(row.get("parish_gv"), row.get("parish"))
    return {
        "county_gv": county_e,
        "barony_gv": barony_e,
        "parish_gv": parish_e,
        "county_effective": county_e,
        "barony_effective": barony_e,
        "parish_effective": parish_e,
        "parish_display": str(row.get("parish", "") or "").strip(),
        "parish_gv_display": str(row.get("parish_gv", "") or "").strip(),
        "townland_display": str(row.get("townland", "") or "").strip(),
    }


def _fuzzy_cluster_canonical(
    names: List[str],
    threshold: float,
    raw_to_normalized: Dict[str, str] | None = None,
) -> Dict[str, str]:
    """
    Within one (county, barony, parish), map each normalized name to a canonical string.

    Uses union-find: any pair with similarity >= threshold is merged; each component
    gets one canonical name (min by string order for stability).

    threshold is in [0, 100] when using rapidfuzz (we scale SequenceMatcher to 0-100).
    """
    names = [n for n in names if n != _NA_SENTINEL]
    if not names:
        return {}
    names = sorted(set(names))

    # Union-find parent array (index = position in names)
    n = len(names)
    parent = list(range(n))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Scale threshold: plan says 0.9; rapidfuzz uses 0-100
    thresh = threshold if threshold > 1 else threshold * 100.0

    for i in range(n):
        for j in range(i + 1, n):
            if _similarity(names[i], names[j]) >= thresh:
                union(i, j)

    # For each component, canonical = min name in that component (by string order)
    component_min: Dict[int, str] = {}
    for i in range(n):
        root = find(i)
        if root not in component_min or names[i] < component_min[root]:
            component_min[root] = names[i]

    # Map each name -> canonical (same component's min)
    norm_to_canonical: Dict[str, str] = {}
    for i in range(n):
        root = find(i)
        canon = component_min[root]
        norm_to_canonical[names[i]] = canon

    # If we have raw -> normalized, canonical should be a raw form for display.
    # Plan says "first occurrence or most frequent raw"; we use first alphabetically
    # among normalized then map back to raw if provided. Here we only have normalized
    # names, so canonical is the chosen normalized rep (component min). Caller can
    # later map to a preferred raw form if desired. For now we use normalized as
    # canonical so grouping is stable.
    return norm_to_canonical


def build_canonical_townland_map(
    df: pd.DataFrame,
    *,
    threshold: float = 0.9,
    normalize_admin_cols: bool = True,
) -> Dict[Tuple[str, str, str, str], str]:
    """
    Build mapping (c, b, p, norm_townland) -> canonical_townland_str.

    Fuzzy clustering is run only within each (county, barony, parish).
    """
    for col in ["county_gv", "barony_gv", "parish_gv", "townland_gv"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'")

    if normalize_admin_cols:
        df = df.copy()
        df["_c_norm"] = df["county_gv"].map(normalize_admin)
        df["_b_norm"] = df["barony_gv"].map(normalize_admin)
        df["_p_norm"] = df["parish_gv"].map(normalize_admin)
    else:
        df = df.copy()
        df["_c_norm"] = df["county_gv"].fillna("").astype(str).str.strip().str.lower()
        df["_b_norm"] = df["barony_gv"].fillna("").astype(str).str.strip().str.lower()
        df["_p_norm"] = df["parish_gv"].fillna("").astype(str).str.strip().str.lower()

    df["_t_norm"] = df["townland_gv"].map(normalize_townland)

    # Unique (c, b, p) groups
    groups = df[["_c_norm", "_b_norm", "_p_norm"]].drop_duplicates()

    result: Dict[Tuple[str, str, str, str], str] = {}

    for _, row in groups.iterrows():
        c, b, p = row["_c_norm"], row["_b_norm"], row["_p_norm"]
        subset = df[(df["_c_norm"] == c) & (df["_b_norm"] == b) & (df["_p_norm"] == p)]
        unique_townlands = subset["_t_norm"].dropna().unique().tolist()
        unique_townlands = [u for u in unique_townlands if u != _NA_SENTINEL]
        if not unique_townlands:
            # All NA townlands in this group -> single canonical
            result[(c, b, p, _NA_SENTINEL)] = _NA_SENTINEL
            continue
        norm_to_canon = _fuzzy_cluster_canonical(unique_townlands, threshold=threshold)
        for norm_name, canon_name in norm_to_canon.items():
            result[(c, b, p, norm_name)] = canon_name
        if _NA_SENTINEL in subset["_t_norm"].values:
            result[(c, b, p, _NA_SENTINEL)] = _NA_SENTINEL

    return result


def add_canonical_townland(
    df: pd.DataFrame,
    *,
    threshold: float = 0.9,
    normalize_admin_cols: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add column townland_canonical using fuzzy clustering within (county, barony, parish).

    Returns a DataFrame with townland_canonical set; if inplace=True, modifies df
    and returns it. Otherwise returns a copy.
    """
    if not inplace:
        df = df.copy()
    for col in ["county_gv", "barony_gv", "parish_gv", "townland_gv"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column '{col}'")

    # Build map keyed by normalized (c, b, p, townland)
    if normalize_admin_cols:
        c_norm = df["county_gv"].map(normalize_admin)
        b_norm = df["barony_gv"].map(normalize_admin)
        p_norm = df["parish_gv"].map(normalize_admin)
    else:
        c_norm = df["county_gv"].fillna("").astype(str).str.strip().str.lower()
        b_norm = df["barony_gv"].fillna("").astype(str).str.strip().str.lower()
        p_norm = df["parish_gv"].fillna("").astype(str).str.strip().str.lower()

    t_norm = df["townland_gv"].map(normalize_townland)

    canon_map = build_canonical_townland_map(
        df, threshold=threshold, normalize_admin_cols=normalize_admin_cols
    )

    def lookup(r: pd.Series) -> str:
        key = (r["_c"], r["_b"], r["_p"], r["_t"])
        return canon_map.get(key, r["_t"])

    df["_c"] = c_norm
    df["_b"] = b_norm
    df["_p"] = p_norm
    df["_t"] = t_norm
    df["townland_canonical"] = df.apply(lookup, axis=1)
    df.drop(columns=["_c", "_b", "_p", "_t"], inplace=True)

    return df


def _page_column(df: pd.DataFrame) -> str:
    """Return column used for page identity (target_filename or page_name)."""
    if "target_filename" in df.columns:
        return "target_filename"
    if "page_name" in df.columns:
        return "page_name"
    raise ValueError("DataFrame must have 'target_filename' or 'page_name'")


def build_gv_lookup(
    gv_df: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, str], dict], Dict[str, dict], Dict[str, List[dict]]]:
    """
    Build lookups from ground-truth spreadsheet rows (coalesced admin).

    Returns:
        by_page_parish: (target_filename, parish_norm) -> admin record
        by_page_fallback: target_filename -> first row on page with nonempty county_effective
        by_page_candidates: target_filename -> list of unique candidate records (for fuzzy match)
    """
    by_page_parish: Dict[Tuple[str, str], dict] = {}
    by_page_fallback: Dict[str, dict] = {}
    by_page_candidates: Dict[str, List[dict]] = {}

    if "target_filename" not in gv_df.columns:
        raise ValueError("Ground truth must contain column 'target_filename'")

    for _, row in gv_df.iterrows():
        page = str(row["target_filename"]).strip()
        if not page:
            continue
        rec = _gv_record_from_row(row)

        if page not in by_page_candidates:
            by_page_candidates[page] = []
        by_page_candidates[page].append(rec)

        parish_norms: set[str] = set()
        if _admin_nonempty(row.get("parish_gv")):
            parish_norms.add(normalize_admin(row.get("parish_gv")))
        if _admin_nonempty(row.get("parish")):
            parish_norms.add(normalize_admin(row.get("parish")))
            parish_norms.add(normalize_llm_parish(row.get("parish")))

        for pnorm in parish_norms:
            if pnorm == _NA_SENTINEL:
                continue
            key = (page, pnorm)
            existing = by_page_parish.get(key)
            if existing is None or (
                _admin_nonempty(rec["county_effective"])
                and not _admin_nonempty(existing.get("county_effective"))
            ):
                by_page_parish[key] = rec

        if _admin_nonempty(rec["county_effective"]) and page not in by_page_fallback:
            by_page_fallback[page] = rec

    # Deduplicate candidates per page
    for page, recs in list(by_page_candidates.items()):
        unique: List[dict] = []
        seen: set[Tuple[str, str, str]] = set()
        for rec in recs:
            k = (rec["county_effective"], rec["barony_effective"], rec["parish_effective"])
            if k in seen:
                continue
            seen.add(k)
            unique.append(rec)
        by_page_candidates[page] = unique

    return by_page_parish, by_page_fallback, by_page_candidates


def _apply_gv_record_to_lists(
    rec: dict,
    counties: list,
    baronies: list,
    parishes: list,
    county_eff: list,
    barony_eff: list,
    parish_eff: list,
    methods: list,
    method: str,
) -> None:
    counties.append(rec.get("county_gv", ""))
    baronies.append(rec.get("barony_gv", ""))
    parishes.append(rec.get("parish_gv", ""))
    county_eff.append(rec.get("county_effective", ""))
    barony_eff.append(rec.get("barony_effective", ""))
    parish_eff.append(rec.get("parish_effective", ""))
    methods.append(method)


def _append_empty_gv_lists(
    counties: list,
    baronies: list,
    parishes: list,
    county_eff: list,
    barony_eff: list,
    parish_eff: list,
    methods: list,
    method: str,
) -> None:
    for lst in (counties, baronies, parishes, county_eff, barony_eff, parish_eff):
        lst.append("")
    methods.append(method)


def join_gv_columns(
    df: pd.DataFrame,
    gv_df: pd.DataFrame,
    *,
    parish_col: str = "parish",
    page_col: str = "page_name",
) -> pd.DataFrame:
    """Attach coalesced admin columns and townland_gv from townland_filled."""
    df = df.copy()
    by_page_parish, by_page_fallback, _ = build_gv_lookup(gv_df)

    counties: list = []
    baronies: list = []
    parishes: list = []
    county_eff: list = []
    barony_eff: list = []
    parish_eff: list = []
    methods: list = []

    for _, row in df.iterrows():
        page = str(row[page_col]).strip()
        parish_norm = normalize_llm_parish(row.get(parish_col, ""))
        rec = by_page_parish.get((page, parish_norm))
        method = "parish_key"
        if rec is None or not _admin_nonempty(rec.get("county_effective")):
            rec = by_page_fallback.get(page)
            method = "page_fallback" if rec else "unresolved"
        if rec and _admin_nonempty(rec.get("county_effective")):
            _apply_gv_record_to_lists(
                rec, counties, baronies, parishes, county_eff, barony_eff, parish_eff, methods, method
            )
        else:
            _append_empty_gv_lists(
                counties, baronies, parishes, county_eff, barony_eff, parish_eff, methods, "unresolved"
            )

    df["county_gv"] = counties
    df["barony_gv"] = baronies
    df["parish_gv"] = parishes
    df["county_effective"] = county_eff
    df["barony_effective"] = barony_eff
    df["parish_effective"] = parish_eff
    df["gv_join_method"] = methods
    if "townland_filled" in df.columns:
        df["townland_gv"] = df["townland_filled"]
    elif "townland" in df.columns:
        df["townland_gv"] = df["townland"]
    else:
        raise ValueError("DataFrame must have townland_filled or townland")

    return df


def _score_parish_match(llm_norm: str, candidate: dict) -> float:
    if llm_norm == _NA_SENTINEL:
        return 0.0
    labels: List[str] = []
    for key in ("parish_gv_display", "parish_display", "parish_effective"):
        val = candidate.get(key, "")
        if _admin_nonempty(val):
            labels.append(normalize_admin(val))
            labels.append(normalize_llm_parish(val))
    if not labels:
        return 0.0
    return max(_similarity(llm_norm, lab) for lab in set(labels))


def _load_gv_join_overrides(path: str | None) -> Dict[Tuple[str, str], dict]:
    """Map (page_name, llm_parish_norm) -> effective admin; '' parish_norm = whole page."""
    if not path or not os.path.isfile(path):
        return {}
    overrides: Dict[Tuple[str, str], dict] = {}
    odf = pd.read_csv(path)
    for _, row in odf.iterrows():
        page = str(row.get("page_name", "")).strip()
        if not page:
            continue
        llm_parish = normalize_llm_parish(row.get("llm_parish", ""))
        if llm_parish == _NA_SENTINEL:
            llm_parish = ""
        overrides[(page, llm_parish)] = {
            "county_effective": str(row.get("county_effective", "")).strip(),
            "barony_effective": str(row.get("barony_effective", "")).strip(),
            "parish_effective": str(row.get("parish_effective", "")).strip(),
        }
    return overrides


def resolve_gv_parish_fuzzy(
    df: pd.DataFrame,
    gv_df: pd.DataFrame,
    *,
    threshold: float = 0.9,
    page_col: str = "page_name",
    parish_col: str = "parish",
    review_path: str | None = None,
    overrides_path: str | None = None,
    margin: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fuzzy-match unresolved LLM parishes to GV rows per page; write ambiguous cases to review_path.

    Returns (updated df, review DataFrame).
    """
    df = df.copy()
    thresh = threshold if threshold > 1 else threshold * 100.0
    _, _, by_page_candidates = build_gv_lookup(gv_df)
    overrides = _load_gv_join_overrides(overrides_path)

    review_rows: List[dict[str, Any]] = []

    if "gv_join_method" not in df.columns:
        df["gv_join_method"] = "unresolved"

    for page, page_df in df.groupby(page_col, sort=False):
        page = str(page).strip()
        candidates = by_page_candidates.get(page, [])
        if not candidates:
            for llm_parish in page_df[parish_col].dropna().unique():
                review_rows.append(
                    {
                        "page_name": page,
                        "llm_parish": llm_parish,
                        "llm_parish_norm": normalize_llm_parish(llm_parish),
                        "status": "no_gv_rows",
                        "candidate_rank": "",
                        "score": "",
                        "county": "",
                        "barony": "",
                        "parish": "",
                        "parish_gv": "",
                        "county_effective": "",
                        "barony_effective": "",
                        "parish_effective": "",
                    }
                )
            continue

        parish_norms = page_df[parish_col].map(normalize_llm_parish)

        for llm_norm, grp in page_df.groupby(parish_norms, sort=False):
            if llm_norm == _NA_SENTINEL:
                continue
            llm_display = str(grp[parish_col].iloc[0])

            override_key = (page, llm_norm)
            override_whole_page = (page, "")
            if override_key in overrides:
                rec_o = overrides[override_key]
            elif override_whole_page in overrides:
                rec_o = overrides[override_whole_page]
            else:
                rec_o = None

            if rec_o and _admin_nonempty(rec_o.get("county_effective")):
                _apply_fuzzy_match_to_indices(df, grp.index, rec_o, "override")
                continue

            if grp["county_effective"].apply(_admin_nonempty).all():
                continue

            scored: List[Tuple[float, int, dict]] = []
            for rank, cand in enumerate(candidates, start=1):
                score = _score_parish_match(llm_norm, cand)
                scored.append((score, rank, cand))
            scored.sort(key=lambda x: (-x[0], x[1]))

            if not scored or scored[0][0] < thresh:
                status = "no_match" if not scored else "low_score"
                for score, rank, cand in scored[:3]:
                    review_rows.append(
                        _review_row(page, llm_display, llm_norm, status, rank, score, cand)
                    )
                if not scored:
                    review_rows.append(
                        _review_row(page, llm_display, llm_norm, "no_match", "", "", {})
                    )
                continue

            best_score, _, best_cand = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            if len(scored) > 1 and (best_score - second_score) < margin:
                for score, rank, cand in scored[:3]:
                    review_rows.append(
                        _review_row(page, llm_display, llm_norm, "ambiguous", rank, score, cand)
                    )
                continue

            _apply_fuzzy_match_to_indices(df, grp.index, best_cand, "fuzzy_parish")

    review_df = pd.DataFrame(review_rows)
    if review_path:
        os.makedirs(os.path.dirname(review_path) or ".", exist_ok=True)
        review_df.to_csv(review_path, index=False)

    return df, review_df


def _review_row(
    page: str,
    llm_parish: str,
    llm_norm: str,
    status: str,
    rank: Any,
    score: Any,
    cand: dict,
) -> dict[str, Any]:
    return {
        "page_name": page,
        "llm_parish": llm_parish,
        "llm_parish_norm": llm_norm,
        "status": status,
        "candidate_rank": rank,
        "score": round(float(score), 2) if score != "" and score is not None else "",
        "county": cand.get("county_effective", ""),
        "barony": cand.get("barony_effective", ""),
        "parish": cand.get("parish_display", ""),
        "parish_gv": cand.get("parish_gv_display", ""),
        "county_effective": cand.get("county_effective", ""),
        "barony_effective": cand.get("barony_effective", ""),
        "parish_effective": cand.get("parish_effective", ""),
    }


def _apply_fuzzy_match_to_indices(
    df: pd.DataFrame,
    indices: pd.Index,
    rec: dict,
    method: str,
) -> None:
    for col, key in (
        ("county_gv", "county_effective"),
        ("barony_gv", "barony_effective"),
        ("parish_gv", "parish_effective"),
        ("county_effective", "county_effective"),
        ("barony_effective", "barony_effective"),
        ("parish_effective", "parish_effective"),
    ):
        df.loc[indices, col] = rec.get(key, "")
    df.loc[indices, "gv_join_method"] = method


def _normalize_townland_key(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def forward_fill_townland(
    df: pd.DataFrame,
    *,
    page_col: str = "page_name",
    parish_col: str = "parish",
    raw_col: str = "townland_raw",
) -> pd.Series:
    """
    Within each (page, parish), forward-fill empty townland_raw from the previous
    non-empty value in table row order.
    """
    if raw_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{raw_col}'")

    out = pd.Series(index=df.index, dtype=object)
    for _, grp in df.groupby([page_col, parish_col], sort=False):
        prev: str | None = None
        for idx, raw in grp[raw_col].items():
            if _normalize_townland_key(raw):
                prev = str(raw).strip() if pd.notna(raw) else ""
            out.at[idx] = prev if prev else ""
    return out


_ADMIN_LINK_COLS = ["county_effective", "barony_effective", "parish_effective"]


def _admin_link_key(row: pd.Series, admin_cols: list[str]) -> tuple[str, ...]:
    return tuple(
        str(row.get(col, "") or "").strip()
        if pd.notna(row.get(col, ""))
        else ""
        for col in admin_cols
    )


def _last_nonempty_townland(page_df: pd.DataFrame, townland_col: str) -> str | None:
    last: str | None = None
    for val in page_df[townland_col]:
        if _normalize_townland_key(val):
            last = str(val).strip() if pd.notna(val) else ""
    return last


def _first_nonempty_townland(page_df: pd.DataFrame, townland_col: str) -> str | None:
    for val in page_df[townland_col]:
        if _normalize_townland_key(val):
            return str(val).strip() if pd.notna(val) else ""
    return None


def _last_nonempty_townland_info(
    page_df: pd.DataFrame,
    townland_col: str,
    admin_cols: list[str],
) -> tuple[str, tuple[str, ...]] | None:
    last: tuple[str, tuple[str, ...]] | None = None
    for _, row in page_df.iterrows():
        if _normalize_townland_key(row[townland_col]):
            last = (str(row[townland_col]).strip(), _admin_link_key(row, admin_cols))
    return last


def _first_nonempty_townland_info(
    page_df: pd.DataFrame,
    townland_col: str,
    admin_cols: list[str],
) -> tuple[str, tuple[str, ...]] | None:
    for _, row in page_df.iterrows():
        if _normalize_townland_key(row[townland_col]):
            return (str(row[townland_col]).strip(), _admin_link_key(row, admin_cols))
    return None


def link_consecutive_page_townlands(
    df: pd.DataFrame,
    *,
    threshold: float = 0.9,
    page_col: str = "page_name",
    townland_col: str = "townland_filled",
    out_col: str = "townland_linked",
    admin_cols: list[str] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Copy townland_col to out_col, then for each consecutive page pair (sorted page names),
    if similarity(last non-empty townland on A, first on B) >= threshold, replace the
    leading contiguous block on B (matching B's original first townland) with A's last name.

    When admin_cols (default: county/barony/parish_effective) are present, linking only
    occurs if the boundary rows share the same admin triple.
    """
    if not inplace:
        df = df.copy()
    if townland_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{townland_col}'")

    link_admin_cols = admin_cols if admin_cols is not None else _ADMIN_LINK_COLS
    use_admin_scope = all(col in df.columns for col in link_admin_cols)

    df[out_col] = df[townland_col]
    thresh = threshold if threshold > 1 else threshold * 100.0
    pages_sorted = sorted(df[page_col].dropna().unique())

    for i in range(len(pages_sorted) - 1):
        page_a, page_b = pages_sorted[i], pages_sorted[i + 1]
        order_a = df.loc[df[page_col] == page_a].sort_index()
        order_b = df.loc[df[page_col] == page_b].sort_index()
        if order_a.empty or order_b.empty:
            continue

        # Read page A's name from out_col (already linked) so the canonical name
        # propagates transitively across 3+ page chains with OCR spelling drift.
        if use_admin_scope:
            last_info = _last_nonempty_townland_info(order_a, out_col, link_admin_cols)
            first_info = _first_nonempty_townland_info(order_b, out_col, link_admin_cols)
            if not last_info or not first_info:
                continue
            last_a, admin_a = last_info
            first_b, admin_b = first_info
            if admin_a != admin_b:
                continue
        else:
            last_a = _last_nonempty_townland(order_a, out_col)
            first_b = _first_nonempty_townland(order_b, out_col)
            if not last_a or not first_b:
                continue

        score = _similarity(normalize_townland(last_a), normalize_townland(first_b))
        if score < thresh:
            continue

        first_b_norm = normalize_townland(first_b)
        boundary_admin = admin_b if use_admin_scope else None
        for idx in order_b.index:
            val_norm = normalize_townland(df.at[idx, out_col])
            if val_norm != first_b_norm:
                break
            if use_admin_scope and boundary_admin is not None:
                if _admin_link_key(df.loc[idx], link_admin_cols) != boundary_admin:
                    break
            df.at[idx, out_col] = last_a

    return df


_LINK_KEY_COLS = ["county_effective", "barony_effective", "parish_effective", "townland_linked"]


def get_one_page_townland_ids_linked(df: pd.DataFrame) -> pd.Index:
    """(county_gv, barony_gv, parish_gv, townland_linked) keys appearing on exactly one page."""
    if "townland_linked" not in df.columns:
        raise ValueError("DataFrame must have column 'townland_linked'")
    page_col = _page_column(df)
    grouped = df.groupby(_LINK_KEY_COLS)[page_col].nunique()
    return grouped[grouped == 1].index


def get_multi_page_townland_ids_linked(df: pd.DataFrame) -> pd.Index:
    """(county_gv, barony_gv, parish_gv, townland_linked) keys appearing on more than one page."""
    if "townland_linked" not in df.columns:
        raise ValueError("DataFrame must have column 'townland_linked'")
    page_col = _page_column(df)
    grouped = df.groupby(_LINK_KEY_COLS)[page_col].nunique()
    return grouped[grouped > 1].index


def write_one_page_townlands_json_linked(
    df: pd.DataFrame,
    one_page_ids: pd.Index,
    output_path: str = "one_page_townlands.json",
) -> str:
    """Write one-page townland records keyed on townland_linked."""
    return _write_townlands_scope_json(
        df, one_page_ids, _LINK_KEY_COLS, "townland_linked", output_path
    )


def write_multi_page_townlands_json(
    df: pd.DataFrame,
    multi_page_ids: pd.Index,
    output_path: str = "multi_page_townlands.json",
) -> str:
    """Write multi-page townland records keyed on townland_linked."""
    return _write_townlands_scope_json(
        df, multi_page_ids, _LINK_KEY_COLS, "townland_linked", output_path
    )


def _write_townlands_scope_json(
    df: pd.DataFrame,
    scope_ids: pd.Index,
    key_cols: List[str],
    townland_field: str,
    output_path: str,
) -> str:
    scope_set = set(
        scope_ids.tolist() if isinstance(scope_ids, pd.MultiIndex) else list(scope_ids)
    )
    keys = list(zip(*(df[c] for c in key_cols)))
    mask = [k in scope_set for k in keys]
    page_col = _page_column(df)
    out_cols = [c for c in (
        "county_effective",
        "barony_effective",
        "parish_effective",
        townland_field,
        page_col,
    ) if c in df.columns]
    scoped_df = df.loc[mask, out_cols].drop_duplicates()
    if page_col != "target_filename":
        scoped_df = scoped_df.rename(columns={page_col: "target_filename"})
    records = scoped_df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return output_path


def _iter_townland_segments(df: pd.DataFrame, townland_col: str = "townland_gv"):
    """Yield (townland_label, segment_df, start_idx, end_idx) for contiguous townland runs."""
    d = df.reset_index(drop=True)
    n = len(d)
    i = 0
    while i < n:
        key = _normalize_townland_key(d.loc[i, townland_col])
        if not key:
            i += 1
            continue
        tl_first = d.loc[i, townland_col]
        j = i + 1
        while j < n and _normalize_townland_key(d.loc[j, townland_col]) == key:
            j += 1
        yield tl_first, d.iloc[i:j], i, j - 1
        i = j


def _best_fuzzy_match(name: str, candidates: List[str], threshold: float) -> str | None:
    if not name or not candidates:
        return None
    norm_name = normalize_townland(name)
    if norm_name == _NA_SENTINEL:
        return None
    thresh = threshold if threshold > 1 else threshold * 100.0
    best: str | None = None
    best_score = -1.0
    for cand in candidates:
        norm_cand = normalize_townland(cand)
        if norm_cand == _NA_SENTINEL:
            continue
        score = _similarity(norm_name, norm_cand)
        if score > best_score:
            best_score = score
            best = cand
    if best_score >= thresh:
        return best
    return None


def resolve_continued_townlands(
    df: pd.DataFrame,
    *,
    threshold: float = 0.9,
    page_col: str = "page_name",
    townland_col: str = "townland_gv",
) -> pd.DataFrame:
    """
    For segments whose first row has is_continued, fuzzy-match to the last townland
    on the previous page (same county_gv, barony_gv, parish_gv) and unify townland_gv.
    """
    df = df.copy()
    if townland_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{townland_col}'")

    admin_cols = ["county_gv", "barony_gv", "parish_gv"]
    for c in admin_cols:
        if c not in df.columns:
            df[c] = ""

    pages_sorted = sorted(df[page_col].dropna().unique())

    # Last townland name per (admin triple, page) from final segment on that page
    last_townland_by_admin_page: Dict[Tuple[str, str, str, str], str] = {}

    for page_idx, page in enumerate(pages_sorted):
        page_mask = df[page_col] == page
        page_df = df.loc[page_mask]
        if page_df.empty:
            continue

        for (county, barony, parish), grp in page_df.groupby(admin_cols, dropna=False):
            c = str(county) if pd.notna(county) else ""
            b = str(barony) if pd.notna(barony) else ""
            p = str(parish) if pd.notna(parish) else ""

            prev_page_key = None
            for pi in range(page_idx - 1, -1, -1):
                prev_page = pages_sorted[pi]
                pk = (c, b, p, prev_page)
                if pk in last_townland_by_admin_page:
                    prev_page_key = pk
                    break

            segments = list(_iter_townland_segments(grp, townland_col=townland_col))
            for tl_first, segment, start_idx, end_idx in segments:
                global_indices = segment.index.tolist()
                is_cont = False
                if "is_continued" in segment.columns:
                    is_cont = (
                        pd.to_numeric(segment["is_continued"], errors="coerce").fillna(0).iloc[0] > 0
                    )
                if is_cont and prev_page_key is not None:
                    prior_name = last_townland_by_admin_page[prev_page_key]
                    matched = _best_fuzzy_match(str(tl_first), [prior_name], threshold)
                    if matched:
                        df.loc[global_indices, townland_col] = matched

            if segments:
                _, last_seg, _, _ = segments[-1]
                last_name = str(last_seg[townland_col].iloc[0])
                if last_name and last_name.lower() != "nan":
                    last_townland_by_admin_page[(c, b, p, page)] = last_name

    return df


def assign_townland_canonical_with_continued(
    df: pd.DataFrame,
    *,
    threshold: float = 0.9,
    page_col: str = "page_name",
    inplace: bool = False,
) -> pd.DataFrame:
    """Resolve cross-page continuations, then add townland_canonical via fuzzy clustering."""
    if not inplace:
        df = df.copy()

    has_gv = df[["county_gv", "barony_gv", "parish_gv"]].notna().any(axis=1) & (
        df["county_gv"].astype(str).str.strip() != ""
    )
    if not has_gv.any():
        df["townland_canonical"] = df.get("townland_gv", pd.Series(dtype=str)).map(normalize_townland)
        return df

    df_gv = df.loc[has_gv].copy()
    df_gv = resolve_continued_townlands(df_gv, threshold=threshold, page_col=page_col)
    df_gv = add_canonical_townland(df_gv, threshold=threshold, inplace=True)

    df["townland_canonical"] = ""
    df.loc[has_gv, "townland_gv"] = df_gv["townland_gv"]
    df.loc[has_gv, "townland_canonical"] = df_gv["townland_canonical"]

    return df


def get_one_page_townland_ids(df: pd.DataFrame) -> pd.Index:
    """
    Return the index of (county_gv, barony_gv, parish_gv, townland_canonical)
    where the townland appears on exactly one page.

    Expects df to have column townland_canonical (e.g. from add_canonical_townland).
    """
    if "townland_canonical" not in df.columns:
        raise ValueError("DataFrame must have column 'townland_canonical' (run add_canonical_townland first)")
    page_col = _page_column(df)
    grouped = (
        df.groupby(["county_gv", "barony_gv", "parish_gv", "townland_canonical"])[page_col]
        .nunique()
    )
    return grouped[grouped == 1].index


def get_pages_to_check(
    df: pd.DataFrame,
    one_page_ids: pd.Index,
) -> pd.Series:
    """
    Return unique target_filename values for rows whose (county_gv, barony_gv, parish_gv, townland_canonical)
    is in one_page_ids, sorted.
    """
    if "townland_canonical" not in df.columns:
        raise ValueError("DataFrame must have column 'townland_canonical'")
    key_cols = ["county_gv", "barony_gv", "parish_gv", "townland_canonical"]
    for c in key_cols:
        if c not in df.columns:
            raise ValueError(f"DataFrame must have column '{c}'")

    one_page_set = set(
        one_page_ids.tolist() if isinstance(one_page_ids, pd.MultiIndex) else list(one_page_ids)
    )
    keys = list(
        zip(
            df["county_gv"],
            df["barony_gv"],
            df["parish_gv"],
            df["townland_canonical"],
        )
    )
    mask = [k in one_page_set for k in keys]
    page_col = _page_column(df)
    pages = df.loc[mask, page_col].dropna().unique()
    return pd.Series(sorted(pages))


def write_townlands_excel(
    df: pd.DataFrame,
    one_page_ids: pd.Index,
    pages_to_check: pd.Series,
    output_path: str = "townlands_canonical.xlsx",
) -> str:
    """
    Write townlands summary and pages_to_check to a new Excel file.

    Sheets:
      - townlands: one row per canonical townland (county, barony, parish, townland_canonical)
        with page_count, is_one_page, and pages (comma-separated target_filename list).
      - pages_to_check: single column of target_filename values for one-page townlands.

    Returns the output_path written.
    """
    if "townland_canonical" not in df.columns:
        raise ValueError("DataFrame must have column 'townland_canonical'")

    grouped = df.groupby(["county_gv", "barony_gv", "parish_gv", "townland_canonical"])
    townlands_summary = (
        grouped["target_filename"]
        .agg([("pages", lambda s: ", ".join(sorted(s.dropna().unique()))), ("page_count", "nunique")])
        .reset_index()
    )
    townlands_summary["is_one_page"] = townlands_summary["page_count"] == 1

    pages_df = pd.DataFrame({"target_filename": pages_to_check})

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        townlands_summary.to_excel(writer, sheet_name="townlands", index=False)
        pages_df.to_excel(writer, sheet_name="pages_to_check", index=False)

    return output_path


def write_one_page_townlands_json(
    df: pd.DataFrame,
    one_page_ids: pd.Index,
    output_path: str = "one_page_townlands.json",
) -> str:
    """
    Write a list of one-page townlands to a JSON file.

    Each item is a dict with county_gv, barony_gv, parish_gv, townland_canonical,
    and target_filename. Returns the output_path written.
    """
    if "townland_canonical" not in df.columns:
        raise ValueError("DataFrame must have column 'townland_canonical'")
    one_page_set = set(
        one_page_ids.tolist() if isinstance(one_page_ids, pd.MultiIndex) else list(one_page_ids)
    )
    keys = list(
        zip(
            df["county_gv"],
            df["barony_gv"],
            df["parish_gv"],
            df["townland_canonical"],
        )
    )
    mask = [k in one_page_set for k in keys]
    page_col = _page_column(df)
    out_cols = ["county_gv", "barony_gv", "parish_gv", "townland_canonical", page_col]
    one_page_df = df.loc[mask, out_cols].drop_duplicates()
    if page_col != "target_filename":
        one_page_df = one_page_df.rename(columns={page_col: "target_filename"})
    records = one_page_df.to_dict(orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return output_path


if __name__ == "__main__":
    import os

    input_file = "nathan_to_fix.xlsx"
    if not os.path.exists(input_file):
        print(f"Demo: {input_file} not found. Skipping run.")
        print("Usage: ensure nathan_to_fix.xlsx exists, then run add_canonical_townland on df[df['check_townland']==1]")
        raise SystemExit(0)

    df = pd.read_excel(input_file)
    df_filtered = df[df["check_townland"] == 1].copy()
    print(f"Filtered rows (check_townland==1): {len(df_filtered)}")

    df_with_canonical = add_canonical_townland(df_filtered, threshold=0.9)
    one_page_ids = get_one_page_townland_ids(df_with_canonical)
    pages_to_check = get_pages_to_check(df_with_canonical, one_page_ids)

    print(f"One-page townland groups: {len(one_page_ids)}")
    print(f"Number of pages to check: {len(pages_to_check)}")
    print("Sample pages_to_check (first 10):")
    for p in pages_to_check.head(10).tolist():
        print(f"  {p}")

    output_excel = "townlands_canonical.xlsx"
    write_townlands_excel(df_with_canonical, one_page_ids, pages_to_check, output_path=output_excel)
    print(f"Wrote townlands to {output_excel}")

    # Save one-page townlands as JSON (list of records)
    one_page_json_path = "one_page_townlands.json"
    write_one_page_townlands_json(df_with_canonical, one_page_ids, output_path=one_page_json_path)
    print(f"Wrote one-page townlands to {one_page_json_path}")

    # Also save one-page townlands to Excel for backward compatibility
    one_page_set = set(
        one_page_ids.tolist() if isinstance(one_page_ids, pd.MultiIndex) else list(one_page_ids)
    )
    keys = list(
        zip(
            df_with_canonical["county_gv"],
            df_with_canonical["barony_gv"],
            df_with_canonical["parish_gv"],
            df_with_canonical["townland_canonical"],
        )
    )
    mask = [k in one_page_set for k in keys]
    one_page_df = (
        df_with_canonical.loc[mask, ["county_gv", "barony_gv", "parish_gv", "townland_canonical", "target_filename"]]
        .drop_duplicates()
    )
    onepage_path = "onepage_townlands.xlsx"
    one_page_df.to_excel(onepage_path, index=False)
    print(f"Wrote one-page townlands to {onepage_path}")
