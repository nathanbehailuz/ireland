"""Central paths for results/log subdirectories."""

from pathlib import Path

RESULTS_LOG_ROOT = Path("results/log")


def results_log_file(subdir: str, filename: str) -> str:
    log_dir = RESULTS_LOG_ROOT / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir / filename)
