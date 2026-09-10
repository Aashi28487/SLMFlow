"""Shared HTTP + IO helpers for the repurposing downloaders.

Every public API used here is free and key-less. An NCBI key raises the PubMed
rate limit from 3 to 10 requests/second; set NCBI_API_KEY if you have one.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[4]      # -> implementation/
RAW = ROOT / "data" / "raw" / "repurposing"
PROCESSED = ROOT / "data" / "processed" / "repurposing"
PARTITIONS = ROOT / "data" / "partitions" / "repurposing"

USER_AGENT = "FedSLM-FYP-research/1.0 (academic prototype; contact via institution)"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "").strip()

_last_call: dict[str, float] = {}


def _throttle(bucket: str, min_interval: float) -> None:
    prev = _last_call.get(bucket)
    if prev is not None:
        wait = min_interval - (time.time() - prev)
        if wait > 0:
            time.sleep(wait)
    _last_call[bucket] = time.time()


def get(url: str, params: dict | None = None, *, bucket: str = "default",
        min_interval: float = 0.35, tries: int = 4, timeout: int = 60,
        empty_on: tuple[int, ...] = ()) -> requests.Response | None:
    """GET with polite throttling and exponential backoff.

    `empty_on` lists status codes that mean "no results" rather than an error
    (openFDA answers 404 for an empty search); those return None immediately
    instead of burning four retries.
    """
    last_err = None
    for attempt in range(tries):
        _throttle(bucket, min_interval)
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r
            if r.status_code in empty_on:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException as exc:            # network hiccup
            last_err = str(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {tries} tries: {url} ({last_err})")


def write_jsonl(path: pathlib.Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
