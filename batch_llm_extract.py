#!/usr/bin/env python3
"""
Prepare / submit / poll / finalize batch valuation extraction via OpenAI Batch API or Gemini Batch API.

Images are referenced by HTTPS URLs built from env + R2-style keys (see upload_images_to_r2.py).
Requires one of: CLOUDFLARE_IMAGE_BASE_URL, R2_PUBLIC_BASE_URL, IMAGE_PUBLIC_BASE_URL (unless --local-base64).

The Cloudflare R2 *public development* URL (r2.dev) is rate-limited; use ``prepare --verify-public-urls`` or
``verify-urls`` to probe URLs with 429 backoff, or use a custom domain for production. See:
https://developers.cloudflare.com/r2/buckets/public-buckets/

Environment:
  openai_api_key     — OpenAI (submit/finalize for openai)
  gemini_api_key     — Gemini (submit/finalize for gemini; not needed for ``refinalize`` with local JSONL)

``refinalize`` re-parses saved batch output JSONL (e.g. gemini_output.jsonl) and writes
results/<provider>/*.jpg.json without calling provider APIs. Use after fixing normalize/parse logic.

Terminal progress bars use tqdm (disable with ``--no-progress`` if output is piped or cluttered).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from extract import (
    EXTRACTION_PROMPT,
    LLM_MODELS,
    build_gemini_generate_content_request_for_base64_jpeg,
    build_gemini_generate_content_request_for_image_url,
    build_openai_chat_completions_body_for_base64_jpeg,
    build_openai_chat_completions_body_for_image_url,
    encode_image_to_base64,
    extract_json_from_content,
    normalize_extraction_result,
)
from results_paths import results_log_file

LOG_FAILURE = results_log_file("analysis", "batch_extract_failures.log")

_TERMINAL_OPENAI = {"completed", "failed", "expired", "cancelled"}
_TERMINAL_GEMINI = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_openai_api_key() -> str:
    """Same var as extract.py (`openai_api_key`); also accepts OPENAI_API_KEY."""
    key = (os.getenv("openai_api_key") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "Set openai_api_key in .env or export OPENAI_API_KEY for the OpenAI SDK."
        )
    return key


def openai_client():
    from openai import OpenAI

    return OpenAI(api_key=get_openai_api_key())


def folder_from_griffith_name(filename: str) -> str:
    """e.g. IRE_GRIFF_004_065.jpg -> 004"""
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[2].zfill(3)
    return "000"


def resolve_local_image_path(target_filename: str) -> str:
    """Same layout as compare_llms.get_image_path (cwd-relative)."""
    folder = folder_from_griffith_name(target_filename)
    return os.path.join("Nanonets", "analysis", folder, os.path.basename(target_filename))


def posix_relpath_from_cwd(path: str) -> str:
    abs_path = os.path.abspath(path)
    cwd = os.getcwd()
    try:
        rel = os.path.relpath(abs_path, cwd)
    except ValueError:
        rel = os.path.basename(abs_path)
    return rel.replace(os.sep, "/")


def infer_r2_key(local_path: str, basename_for_infer: Optional[str] = None) -> str:
    """Object key under project (forward slashes), aligned with upload_images_to_r2 paths."""
    if os.path.isfile(local_path):
        return posix_relpath_from_cwd(local_path)
    base = basename_for_infer or os.path.basename(local_path)
    return resolve_local_image_path(base).replace(os.sep, "/")


def _strip_env_value(raw: str) -> str:
    """Trim whitespace and optional matching quotes from .env values."""
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1]
    return v.strip().rstrip("/")


def get_public_base_url(env_override: Optional[str] = None) -> Tuple[str, str]:
    """Return (env_var_name_used, base_url_without_trailing_slash)."""
    candidates: List[str] = []
    if env_override:
        candidates.append(env_override)
    candidates.extend(
        ["CLOUDFLARE_IMAGE_BASE_URL", "R2_PUBLIC_BASE_URL", "IMAGE_PUBLIC_BASE_URL"]
    )
    seen = set()
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        v = _strip_env_value(os.environ.get(name, "") or "")
        if v:
            return name, v
    raise SystemExit(
        "Set one of CLOUDFLARE_IMAGE_BASE_URL, R2_PUBLIC_BASE_URL, or IMAGE_PUBLIC_BASE_URL "
        "(or pass --public-base-env / use --local-base64)."
    )


def _sleep_for_429(
    response: requests.Response, attempt: int, base_delay: float, max_delay: float
) -> None:
    ra = response.headers.get("Retry-After")
    if ra is not None:
        try:
            delay = float(ra)
        except ValueError:
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    else:
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    delay = min(delay + random.uniform(0, 2.0), max_delay)
    time.sleep(delay)


def check_public_image_url_accessible(
    url: str,
    *,
    max_attempts: int = 15,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    timeout: float = 60.0,
) -> None:
    """
    Confirm a public R2 (or CDN) object URL is reachable. Retries on 429 with
    exponential backoff and optional Retry-After (r2.dev is rate-limited).
    """
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            with requests.get(
                url,
                stream=True,
                timeout=timeout,
                headers={"Range": "bytes=0-8191"},
                allow_redirects=True,
            ) as resp:
                if resp.status_code == 429:
                    logging.warning(
                        "429 Too Many Requests for %s (attempt %s/%s)",
                        url,
                        attempt,
                        max_attempts,
                    )
                    if attempt >= max_attempts:
                        resp.raise_for_status()
                    _sleep_for_429(resp, attempt, base_delay, max_delay)
                    continue
                if resp.status_code in (200, 206):
                    next(resp.iter_content(8192), None)
                    return
                if resp.status_code == 416:
                    # Some origins reject Range; try a tiny full GET
                    break
                if 500 <= resp.status_code < 600:
                    logging.warning(
                        "Server %s for %s (attempt %s/%s)",
                        resp.status_code,
                        url,
                        attempt,
                        max_attempts,
                    )
                    if attempt >= max_attempts:
                        resp.raise_for_status()
                    time.sleep(
                        min(base_delay * (2 ** (attempt - 1)), max_delay)
                        + random.uniform(0, 1.0)
                    )
                    continue
                resp.raise_for_status()
        except requests.RequestException as e:
            logging.warning("Request failed %s: %s (attempt %s/%s)", url, e, attempt, max_attempts)
            if attempt >= max_attempts:
                raise
            time.sleep(
                min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 1.0)
            )

    # Fallback: request without Range
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
                if resp.status_code == 429:
                    logging.warning(
                        "429 for %s (fallback GET, attempt %s/%s)", url, attempt, max_attempts
                    )
                    if attempt >= max_attempts:
                        resp.raise_for_status()
                    _sleep_for_429(resp, attempt, base_delay, max_delay)
                    continue
                if resp.status_code in (200, 206):
                    next(resp.iter_content(8192), None)
                    return
                if 500 <= resp.status_code < 600 and attempt < max_attempts:
                    time.sleep(
                        min(base_delay * (2 ** (attempt - 1)), max_delay)
                        + random.uniform(0, 1.0)
                    )
                    continue
                resp.raise_for_status()
        except requests.RequestException as e:
            if attempt >= max_attempts:
                raise
            logging.warning("Fallback GET failed %s: %s", url, e)
            time.sleep(
                min(base_delay * (2 ** (attempt - 1)), max_delay) + random.uniform(0, 1.0)
            )

    raise RuntimeError(f"Could not confirm URL after retries: {url}")


def verify_urls_with_progress(
    items: List[Tuple[str, str]],
    *,
    context: str,
    total_jobs_hint: Optional[int] = None,
    verify_slice_label: Optional[str] = None,
    no_progress: bool = False,
) -> None:
    """
    Run check_public_image_url_accessible for each (custom_id, url).
    On failure: log exception, IDs verified OK so far, failing ID/URL, and remaining IDs not yet OK.
    """
    n = len(items)
    if n == 0:
        return
    total_hint = total_jobs_hint if total_jobs_hint is not None else n
    if verify_slice_label:
        logging.info(
            "URL verification (%s): scope %s — checking %s URL(s) (total job list size %s)",
            context,
            verify_slice_label,
            n,
            total_hint,
        )
    done_ok: List[str] = []
    bar = tqdm(
        items,
        total=n,
        desc=f"Verify URLs ({context})",
        unit="url",
        disable=no_progress,
        dynamic_ncols=True,
        leave=True,
    )
    for i, (cid, u) in enumerate(bar, start=1):
        bar.set_postfix_str(cid[:42] + ("…" if len(cid) > 42 else ""), refresh=False)
        if no_progress:
            logging.info("Checking [%s/%s] %s — %s", i, n, cid, u)
        try:
            check_public_image_url_accessible(u)
        except Exception as e:
            bar.close()
            failed_and_remaining = [x[0] for x in items[i - 1 :]]
            not_yet_attempted = [x[0] for x in items[i:]]
            logging.error(
                "VERIFY FAILED (%s): %s: %s",
                context,
                type(e).__name__,
                e,
            )
            logging.error(
                "Failed at index %s/%s in this verification batch — custom_id=%r url=%s",
                i,
                n,
                cid,
                u,
            )
            logging.error(
                "Successfully verified before this failure (%s/%s in this batch): %s",
                len(done_ok),
                n,
                done_ok if len(done_ok) <= 30 else done_ok[:30] + [f"... (+{len(done_ok) - 30} more)"],
            )
            logging.error(
                "Not yet successfully verified — failed + remaining in this batch (%s IDs): %s",
                len(failed_and_remaining),
                failed_and_remaining
                if len(failed_and_remaining) <= 40
                else failed_and_remaining[:40] + [f"... (+{len(failed_and_remaining) - 40} more)"],
            )
            logging.error(
                "Remaining URLs not yet attempted after failure (%s IDs): %s",
                len(not_yet_attempted),
                not_yet_attempted
                if len(not_yet_attempted) <= 40
                else not_yet_attempted[:40] + [f"... (+{len(not_yet_attempted) - 40} more)"],
            )
            if total_jobs_hint is not None and total_jobs_hint > n:
                logging.error(
                    "Note: %s job(s) were not part of this verification run (e.g. --verify-max); "
                    "re-run without --verify-max or with a larger limit to cover them.",
                    total_jobs_hint - n,
                )
            append_failure_record(
                {
                    "stage": f"verify_urls:{context}",
                    "error": f"{type(e).__name__}: {e}",
                    "custom_id": cid,
                    "url": u,
                    "index_in_verify_batch": i,
                    "verify_batch_size": n,
                    "total_jobs_in_prepare_list": total_jobs_hint,
                    "verified_ok_before_failure": done_ok,
                    "failed_and_remaining_custom_ids": failed_and_remaining,
                    "remaining_not_yet_attempted": not_yet_attempted,
                }
            )
            raise SystemExit(1) from e
        done_ok.append(cid)
    bar.close()
    logging.info(
        "URL verification (%s): OK for all %s URL(s) in this batch.", context, n
    )


def append_failure_record(record: Dict[str, Any]) -> None:
    record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    with open(LOG_FAILURE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_image_jobs(
    csv_path: Optional[str],
    list_path: Optional[str],
    images: List[str],
) -> List[Dict[str, Any]]:
    """Each entry: custom_id, local_path (hint), r2_key, public_url placeholder."""
    rows: List[Dict[str, Any]] = []

    if csv_path:
        df = pd.read_csv(csv_path)
        if "target_filename" not in df.columns:
            raise SystemExit("CSV must contain column target_filename")
        for _, row in df.iterrows():
            fn = str(row["target_filename"]).strip()
            local = resolve_local_image_path(fn)
            key = infer_r2_key(local, fn)
            cid = os.path.basename(fn)
            rows.append(
                {
                    "custom_id": cid,
                    "local_path": local,
                    "r2_key": key,
                }
            )

    if list_path:
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if os.sep in line or "/" in line:
                    local = line.replace("/", os.sep)
                    key = infer_r2_key(local)
                    cid = os.path.basename(local)
                else:
                    local = resolve_local_image_path(line)
                    key = infer_r2_key(local, line)
                    cid = os.path.basename(line)
                rows.append(
                    {
                        "custom_id": cid,
                        "local_path": local,
                        "r2_key": key,
                    }
                )

    for img in images:
        img = img.strip()
        if not img:
            continue
        if os.sep in img or "/" in img:
            local = img.replace("/", os.sep)
            key = infer_r2_key(local)
            cid = os.path.basename(local)
        else:
            local = resolve_local_image_path(img)
            key = infer_r2_key(local, img)
            cid = os.path.basename(img)
        rows.append(
            {
                "custom_id": cid,
                "local_path": local,
                "r2_key": key,
            }
        )

    # De-dupe by custom_id (last wins)
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        by_id[r["custom_id"]] = r
    return list(by_id.values())


def cmd_prepare(args: argparse.Namespace) -> None:
    jobs = collect_image_jobs(args.csv, args.images_list, args.image or [])
    if not jobs:
        raise SystemExit("No images specified (use --csv, --images-list, and/or --image)")

    env_name: Optional[str] = None
    base_url: Optional[str] = None
    if not args.local_base64:
        env_name, base_url = get_public_base_url(args.public_base_env)
        if getattr(args, "verify_public_urls", False):
            lim = int(getattr(args, "verify_max", 0) or 0)
            check_jobs = jobs[:lim] if lim > 0 else jobs
            slice_label = (
                f"first {len(check_jobs)} of {len(jobs)}"
                if lim > 0
                else f"all {len(jobs)}"
            )
            verify_items = [
                (j["custom_id"], f"{base_url}/{j['r2_key']}") for j in check_jobs
            ]
            verify_urls_with_progress(
                verify_items,
                context="prepare",
                total_jobs_hint=len(jobs),
                verify_slice_label=slice_label,
                no_progress=getattr(args, "no_progress", False),
            )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = Path(args.staging_dir) / run_id
    staging.mkdir(parents=True, exist_ok=True)

    provider = args.provider
    if provider == "openai":
        jsonl_name = "openai_input.jsonl"
    else:
        jsonl_name = "gemini_input.jsonl"

    jsonl_path = staging / jsonl_name
    manifest_images: List[Dict[str, Any]] = []

    np = getattr(args, "no_progress", False)
    with open(jsonl_path, "w", encoding="utf-8") as out:
        for job in tqdm(
            jobs,
            desc="Writing batch JSONL",
            unit="img",
            disable=np,
            dynamic_ncols=True,
            leave=True,
        ):
            cid = job["custom_id"]
            key = job["r2_key"]
            local_path = job["local_path"]

            if args.local_base64:
                if not os.path.isfile(local_path):
                    raise SystemExit(f"Missing local file for --local-base64: {local_path}")
                b64 = encode_image_to_base64(local_path)
                if provider == "openai":
                    body = build_openai_chat_completions_body_for_base64_jpeg(b64)
                    line = {
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                else:
                    req = build_gemini_generate_content_request_for_base64_jpeg(b64)
                    line = {"key": cid, "request": req}
                public_url = None
            else:
                assert base_url is not None
                public_url = f"{base_url}/{key}"
                if provider == "openai":
                    body = build_openai_chat_completions_body_for_image_url(public_url)
                    line = {
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                else:
                    req = build_gemini_generate_content_request_for_image_url(public_url)
                    line = {"key": cid, "request": req}

            out.write(json.dumps(line, ensure_ascii=False) + "\n")
            manifest_images.append(
                {
                    "custom_id": cid,
                    "local_path": local_path,
                    "r2_key": key,
                    "public_url": public_url,
                }
            )

    manifest = {
        "provider": provider,
        "openai_model": LLM_MODELS["openai"],
        "gemini_model": LLM_MODELS["gemini"],
        "image_base_env": env_name,
        "image_base_url": base_url,
        "local_base64": bool(args.local_base64),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "staging_dir": str(staging.resolve()),
        "input_jsonl": str(jsonl_path.resolve()),
        "images": manifest_images,
        "openai_batch_id": None,
        "openai_input_file_id": None,
        "gemini_job_name": None,
        "gemini_src_file_name": None,
        "last_status": None,
        "output_saved_at": None,
    }

    manifest_path = staging / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logging.info("Wrote %s (%s lines)", jsonl_path, len(manifest_images))
    logging.info("Manifest: %s", manifest_path)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def cmd_submit(args: argparse.Namespace) -> None:
    manifest_path = os.path.abspath(args.manifest)
    m = load_manifest(manifest_path)
    provider = m["provider"]
    np = getattr(args, "no_progress", False)

    if provider == "openai":
        client = openai_client()
        input_jsonl = m["input_jsonl"]
        if m.get("openai_batch_id") and args.resume:
            logging.info("Skipping submit; manifest already has openai_batch_id=%s", m["openai_batch_id"])
            return

        if not np:
            tqdm.write("Uploading batch JSONL to OpenAI Files API…")
        with open(input_jsonl, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "valuation table batch"},
        )
        m["openai_input_file_id"] = batch_file.id
        m["openai_batch_id"] = batch.id
        m["last_status"] = batch.status
        save_manifest(manifest_path, m)
        logging.info("Created OpenAI batch id=%s status=%s", batch.id, batch.status)
    else:
        from google import genai
        from google.genai import types

        api_key = os.environ.get("gemini_api_key")
        if not api_key:
            raise SystemExit("Set gemini_api_key in environment")

        client = genai.Client(api_key=api_key)
        input_jsonl = m["input_jsonl"]

        if m.get("gemini_job_name") and args.resume:
            logging.info("Skipping submit; manifest already has gemini_job_name=%s", m["gemini_job_name"])
            return

        if not np:
            tqdm.write("Uploading batch JSONL to Gemini Files API…")
        uploaded = client.files.upload(
            file=input_jsonl,
            config=types.UploadFileConfig(
                display_name="valuation_batch_input",
                mime_type="jsonl",
            ),
        )
        batch_job = client.batches.create(
            model=m.get("gemini_model") or LLM_MODELS["gemini"],
            src=uploaded.name,
            config=types.CreateBatchJobConfig(display_name="valuation-batch"),
        )
        m["gemini_src_file_name"] = uploaded.name
        m["gemini_job_name"] = batch_job.name
        st = batch_job.state
        m["last_status"] = st.name if st is not None else None
        save_manifest(manifest_path, m)
        logging.info("Created Gemini batch job name=%s state=%s", batch_job.name, m["last_status"])


def _gemini_state_name(job: Any) -> str:
    st = getattr(job, "state", None)
    if st is None:
        return "UNKNOWN"
    return getattr(st, "name", None) or str(st)


def cmd_poll(args: argparse.Namespace) -> None:
    manifest_path = os.path.abspath(args.manifest)
    m = load_manifest(manifest_path)
    provider = m["provider"]
    np = getattr(args, "no_progress", False)

    if provider == "openai":
        client = openai_client()
        bid = m.get("openai_batch_id")
        if not bid:
            raise SystemExit("Manifest missing openai_batch_id; run submit first")

        pbar = tqdm(
            disable=np,
            dynamic_ncols=True,
            unit="check",
            bar_format="{desc} | {n_fmt} polls [{elapsed}] {postfix}",
        )
        pbar.set_description("OpenAI batch")
        try:
            while True:
                batch = client.batches.retrieve(bid)
                m["last_status"] = batch.status
                save_manifest(manifest_path, m)
                if np:
                    logging.info("OpenAI batch %s status=%s", bid, batch.status)
                else:
                    pbar.set_postfix_str(f"status={batch.status}", refresh=True)
                    pbar.update(1)
                if batch.status in _TERMINAL_OPENAI:
                    break
                time.sleep(args.interval)
        finally:
            pbar.close()
    else:
        from google import genai

        api_key = os.environ.get("gemini_api_key")
        if not api_key:
            raise SystemExit("Set gemini_api_key in environment")
        client = genai.Client(api_key=api_key)
        name = m.get("gemini_job_name")
        if not name:
            raise SystemExit("Manifest missing gemini_job_name; run submit first")

        pbar = tqdm(
            disable=np,
            dynamic_ncols=True,
            unit="check",
            bar_format="{desc} | {n_fmt} polls [{elapsed}] {postfix}",
        )
        pbar.set_description("Gemini batch")
        try:
            while True:
                job = client.batches.get(name=name)
                sn = _gemini_state_name(job)
                m["last_status"] = sn
                save_manifest(manifest_path, m)
                if np:
                    logging.info("Gemini batch %s state=%s", name, sn)
                else:
                    pbar.set_postfix_str(f"state={sn}", refresh=True)
                    pbar.update(1)
                if sn in _TERMINAL_GEMINI:
                    break
                time.sleep(args.interval)
        finally:
            pbar.close()


def _finalize_openai_one_line(
    obj: Dict[str, Any], provider: str
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """Returns (custom_id, assistant_text, error_dict_or_none)."""
    cid = obj.get("custom_id")
    err = obj.get("error")
    if err:
        return cid, None, err if isinstance(err, dict) else {"message": str(err)}
    resp = obj.get("response")
    if not resp:
        return cid, None, {"message": "no response field"}
    body = resp.get("body")
    if not body:
        return cid, None, {"message": "no response.body"}
    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        return cid, None, {"message": f"parse choices: {e}", "body_keys": list(body.keys())}
    return cid, text, None


def _finalize_gemini_one_line(
    obj: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    key = obj.get("key")
    err = obj.get("error")
    if err:
        return key, None, err if isinstance(err, dict) else {"message": str(err)}
    # Successful line: often {"key": "...", "response": { candidates... }}
    gresp = obj.get("response")
    if gresp is None:
        # Some formats nest differently
        if "generateContentResponse" in obj:
            gresp = obj["generateContentResponse"]
    if not gresp:
        return key, None, {"message": "no response"}
    try:
        parts = gresp["candidates"][0]["content"]["parts"]
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        text = "".join(texts)
    except (KeyError, IndexError, TypeError) as e:
        return key, None, {"message": f"parse gemini response: {e}"}
    return key, text or None, None


def _result_out_path(out_dir: Path, cid: str) -> Path:
    if cid.endswith(".jpg"):
        out_name = cid.replace(".jpg", ".jpg.json")
    else:
        out_name = f"{cid}.jpg.json"
    return out_dir / out_name


def finalize_write_results(
    provider: str,
    manifest_images: List[Dict[str, Any]],
    text_by_id: Dict[str, str],
    *,
    no_progress: bool = False,
    only_missing: bool = False,
) -> int:
    """Parse model output text -> parish JSON files. Returns success count."""
    custom_to_local = {x["custom_id"]: x.get("local_path", "") for x in manifest_images}
    out_dir = Path("results") / ("openai" if provider == "openai" else "gemini")
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped_existing = 0
    parse_failed = 0
    normalize_failed = 0
    items = list(text_by_id.items())
    bar = tqdm(
        items,
        desc="Writing result JSON files",
        unit="file",
        disable=no_progress,
        dynamic_ncols=True,
        leave=True,
    )
    for cid, content in bar:
        bar.set_postfix_str(cid[:36] + ("…" if len(cid) > 36 else ""), refresh=False)
        out_path = _result_out_path(out_dir, cid)
        if only_missing and out_path.is_file():
            skipped_existing += 1
            continue
        local_hint = custom_to_local.get(cid, cid)
        data, _rec = extract_json_from_content(content)
        if not data:
            append_failure_record(
                {
                    "stage": "finalize",
                    "provider": provider,
                    "custom_id": cid,
                    "error": "extract_json_from_content returned None",
                    "local_path": local_hint,
                }
            )
            logging.error("Could not parse JSON from model output for %s", cid)
            parse_failed += 1
            continue
        model_name = "openai" if provider == "openai" else "gemini"
        normalized = normalize_extraction_result(data, model_name, local_hint)
        if not normalized:
            append_failure_record(
                {
                    "stage": "finalize",
                    "provider": provider,
                    "custom_id": cid,
                    "error": "normalize_extraction_result failed",
                    "local_path": local_hint,
                }
            )
            normalize_failed += 1
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        if no_progress:
            logging.info("Wrote %s", out_path)
        ok += 1
    bar.close()
    logging.info(
        "finalize_write_results: written=%s skipped_existing=%s parse_failed=%s normalize_failed=%s (attempted=%s)",
        ok,
        skipped_existing,
        parse_failed,
        normalize_failed,
        len(items),
    )
    return ok


def load_text_by_id_from_openai_jsonl(
    jsonl_path: str, *, no_progress: bool = False
) -> Dict[str, str]:
    text_by_id: Dict[str, str] = {}
    out_lines = [ln for ln in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in tqdm(
        out_lines,
        desc="Parse batch responses",
        unit="line",
        disable=no_progress,
        dynamic_ncols=True,
        leave=False,
    ):
        obj = json.loads(line)
        cid, assistant_text, err = _finalize_openai_one_line(obj, "openai")
        if err:
            append_failure_record(
                {"stage": "refinalize_openai_line", "custom_id": cid, "error": err}
            )
            logging.error("Line error for %s: %s", cid, err)
            continue
        if cid and assistant_text:
            text_by_id[cid] = assistant_text
    return text_by_id


def load_text_by_id_from_gemini_jsonl(
    jsonl_path: str, *, no_progress: bool = False
) -> Dict[str, str]:
    text_by_id: Dict[str, str] = {}
    gem_lines = [ln for ln in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in tqdm(
        gem_lines,
        desc="Parse batch responses",
        unit="line",
        disable=no_progress,
        dynamic_ncols=True,
        leave=False,
    ):
        obj = json.loads(line)
        key, part_text, err = _finalize_gemini_one_line(obj)
        if err:
            append_failure_record(
                {"stage": "refinalize_gemini_line", "custom_id": key, "error": err}
            )
            logging.error("Line error for %s: %s", key, err)
            continue
        if key and part_text:
            text_by_id[key] = part_text
    return text_by_id


def default_batch_output_jsonl(manifest: Dict[str, Any]) -> Path:
    staging = Path(manifest["staging_dir"])
    if manifest.get("provider") == "openai":
        return staging / "openai_output.jsonl"
    return staging / "gemini_output.jsonl"


def cmd_refinalize(args: argparse.Namespace) -> None:
    manifest_path = os.path.abspath(args.manifest)
    m = load_manifest(manifest_path)
    provider = m["provider"]
    images = m.get("images") or []
    np = getattr(args, "no_progress", False)
    only_missing = not getattr(args, "overwrite", False)

    if args.jsonl:
        jsonl_path = Path(args.jsonl).resolve()
    else:
        jsonl_path = default_batch_output_jsonl(m)
    if not jsonl_path.is_file():
        raise SystemExit(
            f"Batch output JSONL not found: {jsonl_path}. "
            "Run finalize once to download/save it, or pass --jsonl."
        )

    logging.info("Refinalizing from local %s (only_missing=%s)", jsonl_path, only_missing)
    if provider == "openai":
        text_by_id = load_text_by_id_from_openai_jsonl(str(jsonl_path), no_progress=np)
    else:
        text_by_id = load_text_by_id_from_gemini_jsonl(str(jsonl_path), no_progress=np)

    n = finalize_write_results(
        provider, images, text_by_id, no_progress=np, only_missing=only_missing
    )
    m["output_saved_at"] = datetime.now().isoformat(timespec="seconds")
    save_manifest(manifest_path, m)
    logging.info("Refinalized %s successful JSON files", n)


def cmd_finalize(args: argparse.Namespace) -> None:
    manifest_path = os.path.abspath(args.manifest)
    m = load_manifest(manifest_path)
    provider = m["provider"]
    images = m.get("images") or []
    np = getattr(args, "no_progress", False)

    if provider == "openai":
        client = openai_client()
        bid = m.get("openai_batch_id")
        if not bid:
            raise SystemExit("Manifest missing openai_batch_id")
        batch = client.batches.retrieve(bid)
        ost = batch.status
        if ost != "completed":
            if ost == "failed":
                extra = ""
                errs = getattr(batch, "errors", None)
                if errs is not None:
                    extra = f" API batch.errors: {errs}"
                raise SystemExit(
                    "OpenAI batch status is 'failed' — input validation or batch processing failed. "
                    "Inspect this batch in the OpenAI dashboard (Batch API) and the batch errors "
                    f"field for id {bid}.{extra}"
                )
            if ost in ("expired", "cancelled"):
                raise SystemExit(
                    f"OpenAI batch status is '{ost}'; there is no successful output to download. "
                    f"Batch id: {bid}"
                )
            raise SystemExit(
                f"OpenAI batch is not finished yet (status={ost}). "
                "Run: python3 batch_llm_extract.py poll --manifest <your-manifest.json> "
                "until status becomes completed, then finalize again."
            )

        out_id = batch.output_file_id
        if not out_id:
            raise SystemExit("Batch completed but output_file_id is missing")

        if not np:
            tqdm.write("Downloading OpenAI batch output file…")
        content = client.files.content(out_id)
        text = content.text
        raw_path = Path(m["staging_dir"]) / "openai_output.jsonl"
        raw_path.write_text(text, encoding="utf-8")
        logging.info("Saved raw OpenAI output JSONL to %s", raw_path)
        text_by_id = load_text_by_id_from_openai_jsonl(str(raw_path), no_progress=np)

        err_id = batch.error_file_id
        if err_id:
            err_body = client.files.content(err_id).text
            err_path = Path(m["staging_dir"]) / "openai_error.jsonl"
            err_path.write_text(err_body, encoding="utf-8")
            logging.warning("Wrote OpenAI batch errors to %s", err_path)

        n = finalize_write_results(
            provider, images, text_by_id, no_progress=np
        )
        m["output_saved_at"] = datetime.now().isoformat(timespec="seconds")
        save_manifest(manifest_path, m)
        logging.info("Finalized %s successful JSON files", n)
    else:
        from google import genai

        api_key = os.environ.get("gemini_api_key")
        if not api_key:
            raise SystemExit("Set gemini_api_key in environment")
        client = genai.Client(api_key=api_key)
        name = m.get("gemini_job_name")
        if not name:
            raise SystemExit("Manifest missing gemini_job_name")

        job = client.batches.get(name=name)
        st = _gemini_state_name(job)
        if st not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
            if st == "JOB_STATE_FAILED":
                je = getattr(job, "error", None)
                extra = f" Job error: {je}" if je else ""
                raise SystemExit(
                    "Gemini batch job failed. Check the job in Google AI / Vertex "
                    f"and the job error details. job={name}.{extra}"
                )
            if st in ("JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                raise SystemExit(
                    f"Gemini batch job state is {st}; there is no successful output to download. "
                    f"job={name}"
                )
            raise SystemExit(
                f"Gemini job not finished yet (state={st}). "
                "Run: python3 batch_llm_extract.py poll --manifest <your-manifest.json> "
                "until the job succeeds, then finalize again."
            )

        dest = job.dest
        file_name = getattr(dest, "file_name", None) if dest else None
        if not file_name:
            raise SystemExit("Job dest.file_name missing; cannot download results")

        if not np:
            tqdm.write("Downloading Gemini batch output file…")
        raw = client.files.download(file=file_name)
        text = raw.decode("utf-8")
        raw_path = Path(m["staging_dir"]) / "gemini_output.jsonl"
        raw_path.write_text(text, encoding="utf-8")
        logging.info("Saved raw Gemini output JSONL to %s", raw_path)
        text_by_id = load_text_by_id_from_gemini_jsonl(str(raw_path), no_progress=np)

        n = finalize_write_results(
            provider, images, text_by_id, no_progress=np
        )
        m["output_saved_at"] = datetime.now().isoformat(timespec="seconds")
        save_manifest(manifest_path, m)
        logging.info("Finalized %s successful JSON files", n)


def cmd_verify_urls(args: argparse.Namespace) -> None:
    """Probe public HTTPS URLs (e.g. r2.dev) with 429 backoff."""
    urls: List[Tuple[str, str]] = []
    if args.manifest:
        m = load_manifest(os.path.abspath(args.manifest))
        for im in m.get("images") or []:
            cid = im.get("custom_id", "")
            u = im.get("public_url")
            if not u:
                logging.warning("No public_url in manifest for %s — skip", cid)
                continue
            urls.append((cid, u))
    elif args.csv or args.images_list or (args.image or []):
        jobs = collect_image_jobs(args.csv, args.images_list, args.image or [])
        if not jobs:
            raise SystemExit("No images from csv/list/image arguments.")
        _, base_url = get_public_base_url(args.public_base_env)
        for job in jobs:
            urls.append((job["custom_id"], f"{base_url}/{job['r2_key']}"))
    else:
        raise SystemExit(
            "Provide --manifest or image source (--csv / --images-list / --image)."
        )

    lim = int(getattr(args, "verify_max", 0) or 0)
    full_count = len(urls)
    if lim > 0:
        urls = urls[:lim]
    slice_label = (
        f"first {len(urls)} of {full_count}" if lim > 0 else f"all {full_count}"
    )
    verify_urls_with_progress(
        urls,
        context="verify-urls",
        total_jobs_hint=full_count,
        verify_slice_label=slice_label,
        no_progress=getattr(args, "no_progress", False),
    )


def cmd_run(args: argparse.Namespace) -> None:
    np = getattr(args, "no_progress", False)
    cmd_submit(
        argparse.Namespace(
            manifest=args.manifest,
            resume=getattr(args, "resume", False),
            no_progress=np,
        )
    )
    cmd_poll(
        argparse.Namespace(
            manifest=args.manifest,
            interval=getattr(args, "interval", 45),
            no_progress=np,
        )
    )
    cmd_finalize(argparse.Namespace(manifest=args.manifest, no_progress=np))


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("prepare", help="Build JSONL + manifest")
    pr.add_argument("--provider", choices=["openai", "gemini"], required=True)
    pr.add_argument("--csv", help="CSV with target_filename column")
    pr.add_argument("--images-list", dest="images_list", help="Newline-separated paths or basenames")
    pr.add_argument("--image", action="append", default=[], help="Single image basename or path (repeatable)")
    pr.add_argument("--staging-dir", default="batch_staging")
    pr.add_argument("--run-id", default=None)
    pr.add_argument(
        "--public-base-env",
        default=None,
        help="Which env var holds the HTTPS base for R2 keys (default: try standard names)",
    )
    pr.add_argument(
        "--local-base64",
        action="store_true",
        help="Embed JPEG as base64 (large JSONL); no public URL needed",
    )
    pr.add_argument(
        "--verify-public-urls",
        action="store_true",
        help="Before writing JSONL, GET each public URL with 429 retry (useful for r2.dev)",
    )
    pr.add_argument(
        "--verify-max",
        type=int,
        default=0,
        metavar="N",
        help="With --verify-public-urls, only check the first N URLs (0 = all)",
    )
    pr.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (plain logging only)",
    )
    pr.set_defaults(func=cmd_prepare)

    vu = sub.add_parser(
        "verify-urls",
        help="Check public image URLs (429 backoff); does not call LLM APIs",
    )
    vu.add_argument(
        "--manifest",
        help="manifest.json from prepare (uses images[].public_url)",
    )
    vu.add_argument("--csv", help="CSV with target_filename (same as prepare)")
    vu.add_argument(
        "--images-list",
        dest="images_list",
        help="Newline-separated paths or basenames",
    )
    vu.add_argument("--image", action="append", default=[], help="Repeatable basename/path")
    vu.add_argument(
        "--public-base-env",
        default=None,
        help="Env var for HTTPS base when using --csv / list / --image",
    )
    vu.add_argument(
        "--verify-max",
        type=int,
        default=0,
        metavar="N",
        help="Only check the first N URLs (0 = all)",
    )
    vu.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    vu.set_defaults(func=cmd_verify_urls)

    su = sub.add_parser("submit", help="Upload JSONL and create batch job")
    su.add_argument("--manifest", required=True, help="Path to manifest.json from prepare")
    su.add_argument(
        "--resume",
        action="store_true",
        help="Skip if batch id already in manifest",
    )
    su.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm status lines during upload",
    )
    su.set_defaults(func=cmd_submit)

    po = sub.add_parser("poll", help="Poll batch until terminal state")
    po.add_argument("--manifest", required=True)
    po.add_argument("--interval", type=int, default=45, help="Seconds between checks")
    po.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm polling bar (log each status line instead)",
    )
    po.set_defaults(func=cmd_poll)

    fi = sub.add_parser("finalize", help="Download batch output and write results/<provider>/*.jpg.json")
    fi.add_argument("--manifest", required=True)
    fi.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars while parsing/writing",
    )
    fi.set_defaults(func=cmd_finalize)

    rf = sub.add_parser(
        "refinalize",
        help="Parse saved batch output JSONL locally (no API download)",
    )
    rf.add_argument("--manifest", required=True)
    rf.add_argument(
        "--jsonl",
        default=None,
        help="Path to gemini_output.jsonl or openai_output.jsonl (default: staging_dir from manifest)",
    )
    rf.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite all result files (default: only write missing results/gemini/*.jpg.json)",
    )
    rf.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars while parsing/writing",
    )
    rf.set_defaults(func=cmd_refinalize)

    ru = sub.add_parser("run", help="submit + poll + finalize")
    ru.add_argument("--manifest", required=True)
    ru.add_argument("--resume", action="store_true")
    ru.add_argument("--interval", type=int, default=45)
    ru.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars for submit/poll/finalize steps",
    )
    ru.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
