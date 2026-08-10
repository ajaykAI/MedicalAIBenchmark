"""Small, auditable I/O helpers used throughout the project."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


class DownloadError(RuntimeError):
    """Raised when a source file cannot be downloaded safely."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    if pretty:
        content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        content += "\n"
        atomic_write_bytes(path, content.encode("utf-8"))
    else:
        atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Malformed JSONL at %s:%d: %s" % (path, line_number, exc)
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    "Expected an object at %s:%d" % (path, line_number)
                )
            yield value


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(path, content)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _request_json(url: str, timeout: int = 60) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "medical-ai-benchmark/0.1 (research eval)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise DownloadError("Could not read %s: %s" % (url, exc)) from exc
    if not isinstance(value, dict):
        raise DownloadError("Expected JSON object from %s" % url)
    return value


def verify_huggingface_revision(dataset: str, expected_revision: str) -> None:
    metadata = _request_json(
        "https://huggingface.co/api/datasets/%s/revision/main" % dataset
    )
    actual = metadata.get("sha")
    if actual != expected_revision:
        raise DownloadError(
            "Dataset %s changed: configured revision %s, current revision %s. "
            "Review the upstream changes and update config intentionally."
            % (dataset, expected_revision, actual)
        )


def download_file(url: str, destination: Path, *, timeout: int = 120) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name, dir=str(destination.parent)
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "medical-ai-benchmark/0.1 (research eval)"},
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise DownloadError("Could not download %s: %s" % (url, exc)) from exc
    return destination


def fetch_huggingface_rows(
    dataset: str,
    config: str,
    split: str,
    destination: Path,
    *,
    page_size: int = 100,
) -> Path:
    """Fetch a small labeled split through the public Dataset Viewer API."""
    rows: List[Dict[str, Any]] = []
    offset = 0
    expected_total: Optional[int] = None
    while expected_total is None or offset < expected_total:
        url = (
            "https://datasets-server.huggingface.co/rows"
            "?dataset=%s&config=%s&split=%s&offset=%d&length=%d"
            % (
                urllib.parse.quote(dataset, safe=""),
                urllib.parse.quote(config, safe=""),
                urllib.parse.quote(split, safe=""),
                offset,
                page_size,
            )
        )
        payload = _request_json(url)
        if expected_total is None:
            expected_total = int(payload.get("num_rows_total", 0))
            if expected_total <= 0:
                raise DownloadError("No rows reported for %s/%s" % (dataset, split))
        page = payload.get("rows")
        if not isinstance(page, list) or not page:
            raise DownloadError(
                "Unexpected empty page for %s/%s at offset %d"
                % (dataset, split, offset)
            )
        for item in page:
            row = item.get("row") if isinstance(item, dict) else None
            if not isinstance(row, dict):
                raise DownloadError("Malformed row at offset %d" % offset)
            rows.append(row)
        offset += len(page)
    if len(rows) != expected_total:
        raise DownloadError(
            "Expected %d rows for %s/%s, received %d"
            % (expected_total, dataset, split, len(rows))
        )
    write_jsonl(destination, rows)
    return Path(destination)
