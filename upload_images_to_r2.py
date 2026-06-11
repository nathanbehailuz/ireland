#!/usr/bin/env python3
"""Upload Griffith JPEGs under Nanonets/analysis/ to Cloudflare R2 (S3-compatible API)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEFAULT_ROOT = Path("Nanonets/analysis")
_SUFFIXES = {".jpg", ".jpeg"}


def build_s3_client():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
    if not endpoint:
        if not account_id:
            sys.exit(
                "Set CLOUDFLARE_ACCOUNT_ID or R2_ENDPOINT_URL for the R2 endpoint."
            )
        endpoint = f"https://{account_id}.r2.cloudflarestorage.com"
    key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    if not key_id or not secret:
        sys.exit("Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY.")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="auto",
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            signature_version="s3v4",
        ),
    )


def collect_images(root: Path, cwd: Path) -> list[tuple[Path, str]]:
    """Return (absolute_path, object_key) using POSIX keys relative to cwd."""
    root = root.resolve()
    cwd = cwd.resolve()
    if not root.is_dir():
        logging.warning("Root does not exist or is not a directory: %s", root)
        return []

    out: list[tuple[Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SUFFIXES:
            continue
        try:
            rel = path.relative_to(cwd)
        except ValueError:
            try:
                root_rel = root.relative_to(cwd)
            except ValueError:
                logging.warning(
                    "Skipping path outside project cwd (cannot build stable key): %s",
                    path,
                )
                continue
            rel_file = path.relative_to(root)
            key = (root_rel / rel_file).as_posix()
            out.append((path, key))
            continue
        out.append((path, rel.as_posix()))
    out.sort(key=lambda x: x[1])
    return out


def apply_prefix(key: str, prefix: str | None) -> str:
    if not prefix:
        return key
    p = prefix.strip().strip("/")
    return f"{p}/{key}" if p else key


def remote_matches_local(client, bucket: str, key: str, local_path: Path) -> bool:
    """True if object exists and ContentLength matches local file size."""
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        remote_size = head["ContentLength"]
        local_size = local_path.stat().st_size
        return remote_size == local_size
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in ("404", "NoSuchKey") or status == 404:
            return False
        raise


def upload_one(client, bucket: str, key: str, local_path: Path) -> None:
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ContentType="image/jpeg",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Directory tree to scan for JPEGs (default: Nanonets/analysis)",
    )
    p.add_argument(
        "--prefix",
        default="",
        help="Optional R2 key prefix prepended to each object key",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List keys only; do not call R2 (no credentials required)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Parallel uploads (default: 8)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Upload even when remote size matches local file",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (plain logging only)",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    cwd = Path.cwd()

    items = collect_images(args.root, cwd)
    prefixed = [(path, apply_prefix(key, args.prefix or None)) for path, key in items]

    logging.info("Found %s JPEG files under %s", len(prefixed), args.root.resolve())

    if args.dry_run:
        for _, key in prefixed[:10]:
            logging.info("[dry-run] %s", key)
        if len(prefixed) > 10:
            logging.info("[dry-run] ... and %s more", len(prefixed) - 10)
        return

    bucket = os.environ.get("R2_BUCKET", "").strip()
    if not bucket:
        sys.exit("Set R2_BUCKET.")

    client = build_s3_client()

    root_log = logging.getLogger()
    show_progress = not args.no_progress and sys.stderr.isatty()
    quiet = show_progress

    to_upload: list[tuple[Path, str]] = []
    skipped = 0
    if args.force:
        to_upload = prefixed
    else:
        check_it = tqdm(
            prefixed,
            desc="Checking remote",
            unit="file",
            disable=not show_progress,
        )
        prev_level = root_log.level
        if quiet:
            root_log.setLevel(logging.WARNING)
        try:
            for path, key in check_it:
                try:
                    if remote_matches_local(client, bucket, key, path):
                        skipped += 1
                        continue
                except ClientError:
                    logging.exception("HeadObject failed for %s", key)
                    raise
                to_upload.append((path, key))
        finally:
            if quiet:
                root_log.setLevel(prev_level)

    logging.info("Skipping %s already synced (size match)", skipped)
    logging.info("Uploading %s objects to bucket %s", len(to_upload), bucket)

    errors = 0
    done = 0

    def task(item: tuple[Path, str]) -> None:
        path, key = item
        upload_one(client, bucket, key, path)

    workers = max(1, args.workers)
    prev_level = root_log.level
    if quiet and to_upload:
        root_log.setLevel(logging.WARNING)
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(task, item): item for item in to_upload}
            pbar_ctx = (
                tqdm(
                    total=len(to_upload),
                    desc="Uploading",
                    unit="obj",
                    disable=not show_progress,
                )
                if to_upload
                else None
            )
            if pbar_ctx is not None:
                with pbar_ctx as pbar:
                    for fut in as_completed(futures):
                        path, key = futures[fut]
                        try:
                            fut.result()
                            done += 1
                        except Exception:
                            errors += 1
                            logging.exception("Failed upload: %s -> %s", path, key)
                        finally:
                            pbar.update(1)
            else:
                for fut in as_completed(futures):
                    path, key = futures[fut]
                    try:
                        fut.result()
                        done += 1
                        if done % 500 == 0 or done == len(to_upload):
                            logging.info("Uploaded %s / %s", done, len(to_upload))
                    except Exception:
                        errors += 1
                        logging.exception("Failed upload: %s -> %s", path, key)
    finally:
        if quiet and to_upload:
            root_log.setLevel(prev_level)

    logging.info("Finished: uploaded=%s errors=%s skipped=%s", done, errors, skipped)
    if errors:
        sys.exit(1)

    # Spot-check: count remote keys under the scanned tree prefix
    try:
        root_rel = args.root.resolve().relative_to(cwd.resolve())
    except ValueError:
        root_rel = None
    if root_rel is not None:
        tree_prefix = apply_prefix(root_rel.as_posix(), args.prefix or None).strip("/")
        list_prefix = f"{tree_prefix}/" if tree_prefix else ""
    else:
        list_prefix = f"{(args.prefix or '').strip().strip('/')}/" if args.prefix else ""

    paginator = client.get_paginator("list_objects_v2")
    total_remote = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        total_remote += len(page.get("Contents", []))
    logging.info(
        "ListObjectsV2 total keys under prefix=%r: %s",
        list_prefix,
        total_remote,
    )


if __name__ == "__main__":
    main()
