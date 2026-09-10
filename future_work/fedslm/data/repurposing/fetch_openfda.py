"""Download FDA-approved indication text for the seed drugs (openFDA, key-less).

This is the ground truth for what counts as an *approved* use. Everything the
literature proposes that is absent here is, by definition, beyond the label.

Writes data/raw/repurposing/openfda_labels.jsonl
"""

from __future__ import annotations

import re

from common import RAW, get, progress, write_jsonl
from drugs import SEED_DRUGS

API = "https://api.fda.gov/drug/label.json"
FDA_BUCKET = "openfda"

# Products whose generic name in the label differs from the common drug name.
ALIASES = {
    "botulinum toxin": "onabotulinumtoxina",
    "valproic acid": "divalproex sodium",
    "ursodeoxycholic acid": "ursodiol",
}


def clean(text: str, limit: int = 6000) -> str:
    text = (text or "").replace("�", " ").replace(" ", " ")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _score(res: dict, name: str) -> int:
    """Prefer a single-ingredient label over a combination product."""
    ofda = res.get("openfda", {}) or {}
    generic = " ".join(ofda.get("generic_name") or []).lower()
    substances = ofda.get("substance_name") or []
    s = 0
    if name.lower() in generic:
        s += 4
    if len(substances) == 1:
        s += 6                      # single active ingredient
    elif substances:
        s -= len(substances)        # combination product
    if " and " in generic or "," in generic:
        s -= 3
    return s


def query(term: str) -> list[dict]:
    params = {"search": term, "limit": 10}
    try:
        r = get(API, params, bucket=FDA_BUCKET, min_interval=0.3, timeout=60,
                empty_on=(404,))
    except RuntimeError:
        return []
    if r is None:                       # 404 = openFDA's "no matches found"
        return []
    return r.json().get("results", []) or []


def fetch_one(drug: str) -> dict | None:
    name = ALIASES.get(drug, drug)
    strategies = [
        f'openfda.generic_name:"{name}" AND _exists_:indications_and_usage',
        f'openfda.substance_name:"{name}" AND _exists_:indications_and_usage',
        f'openfda.brand_name:"{name}" AND _exists_:indications_and_usage',
        f'indications_and_usage:"{name}"',
    ]
    for strat in strategies:
        candidates = [r for r in query(strat)
                      if len(clean(" ".join(r.get("indications_and_usage", []) or []))) >= 40]
        if not candidates:
            continue
        for res in sorted(candidates, key=lambda r: -_score(r, name))[:1]:
            ind = clean(" ".join(res.get("indications_and_usage", []) or []))
            ofda = res.get("openfda", {}) or {}
            return {
                "source": "openfda",
                "drug": drug,
                "label_query": strat.split(" AND ")[0],
                "generic_name": (ofda.get("generic_name") or [name])[0],
                "brand_names": (ofda.get("brand_name") or [])[:5],
                "substance_names": (ofda.get("substance_name") or [])[:5],
                "route": (ofda.get("route") or [])[:3],
                "manufacturer": (ofda.get("manufacturer_name") or [""])[0],
                "application_number": (ofda.get("application_number") or [""])[0],
                "indications_and_usage": ind,
                "boxed_warning": clean(" ".join(res.get("boxed_warning", []) or []), 2000),
                "contraindications": clean(" ".join(res.get("contraindications", []) or []), 2000),
                "url": "https://open.fda.gov/apis/drug/label/",
                "citation": f"FDA prescribing information for "
                            f"{(ofda.get('generic_name') or [name])[0]} (openFDA label API).",
            }
    return None


def main() -> None:
    progress(f"[openfda] fetching labels for {len(SEED_DRUGS)} drugs")
    rows, missing = [], []
    for i, d in enumerate(SEED_DRUGS, 1):
        rec = fetch_one(d["drug"])
        if rec:
            rec["hospital"] = d["hospital"]
            rec["expected_approved_area"] = d["approved_for"]
            rows.append(rec)
            progress(f"  [{i}/{len(SEED_DRUGS)}] {d['drug']}: "
                     f"{len(rec['indications_and_usage'])} chars"
                     f"{' [boxed warning]' if rec['boxed_warning'] else ''}")
        else:
            missing.append(d["drug"])
            progress(f"  [{i}/{len(SEED_DRUGS)}] {d['drug']}: no label found")

    out = RAW / "openfda_labels.jsonl"
    n = write_jsonl(out, rows)
    progress(f"[openfda] wrote {n} labels -> {out}")
    if missing:
        progress(f"[openfda] no label for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
