"""Download ClinicalTrials.gov studies for the seed drugs (API v2, key-less).

Writes data/raw/repurposing/ctgov_studies.jsonl
"""

from __future__ import annotations

import argparse

from common import RAW, get, progress, write_jsonl
from drugs import SEED_DRUGS

API = "https://clinicaltrials.gov/api/v2/studies"
CTG_BUCKET = "ctgov"

FIELDS = ",".join([
    "NCTId", "BriefTitle", "OfficialTitle", "Condition", "InterventionName",
    "InterventionType", "Phase", "OverallStatus", "BriefSummary", "EnrollmentCount",
    "StartDate", "CompletionDate", "LeadSponsorName", "StudyType",
    "PrimaryOutcomeMeasure", "WhyStopped", "HasResults",
])

PHASE_RANK = {"EARLY_PHASE1": 0, "PHASE1": 1, "PHASE1|PHASE2": 1, "PHASE2": 2,
              "PHASE2|PHASE3": 2, "PHASE3": 3, "PHASE4": 4, "NA": -1}


def _mod(study: dict, name: str) -> dict:
    return study.get("protocolSection", {}).get(name, {}) or {}


def parse_study(study: dict) -> dict | None:
    ident = _mod(study, "identificationModule")
    nct = ident.get("nctId")
    if not nct:
        return None
    status = _mod(study, "statusModule")
    design = _mod(study, "designModule")
    desc = _mod(study, "descriptionModule")
    conds = _mod(study, "conditionsModule").get("conditions", []) or []
    arms = _mod(study, "armsInterventionsModule").get("interventions", []) or []
    outcomes = _mod(study, "outcomesModule").get("primaryOutcomes", []) or []

    phases = design.get("phases", []) or []
    phase = "|".join(phases) if phases else "NA"
    sponsor = (_mod(study, "sponsorCollaboratorsModule").get("leadSponsor", {}) or {}).get("name", "")

    return {
        "source": "clinicaltrials.gov",
        "nct_id": nct,
        "brief_title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "conditions": conds,
        "interventions": [{"name": i.get("name", ""), "type": i.get("type", "")} for i in arms],
        "drug_interventions": [i.get("name", "") for i in arms
                               if (i.get("type") or "").upper() in ("DRUG", "BIOLOGICAL")],
        "phase": phase,
        "phase_rank": PHASE_RANK.get(phase, -1),
        "study_type": design.get("studyType", ""),
        "overall_status": status.get("overallStatus", ""),
        "why_stopped": status.get("whyStopped", ""),
        "start_date": (status.get("startDateStruct", {}) or {}).get("date", ""),
        "completion_date": (status.get("completionDateStruct", {}) or {}).get("date", ""),
        "enrollment": (design.get("enrollmentInfo", {}) or {}).get("count"),
        "lead_sponsor": sponsor,
        "brief_summary": desc.get("briefSummary", ""),
        "primary_outcomes": [o.get("measure", "") for o in outcomes],
        "has_results": bool(study.get("hasResults")),
        "url": f"https://clinicaltrials.gov/study/{nct}",
        "citation": f"{ident.get('briefTitle', '')} ClinicalTrials.gov {nct} "
                    f"(phase {phase}, {status.get('overallStatus', '')}).",
    }


def search(params: dict, max_studies: int) -> list[dict]:
    out, token = [], None
    while len(out) < max_studies:
        p = dict(params)
        p.update({"format": "json", "fields": FIELDS,
                  "pageSize": min(100, max_studies - len(out))})
        if token:
            p["pageToken"] = token
        r = get(API, p, bucket=CTG_BUCKET, min_interval=0.4, timeout=90)
        payload = r.json()
        batch = payload.get("studies", []) or []
        if not batch:
            break
        out.extend(batch)
        token = payload.get("nextPageToken")
        if not token:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-drug", type=int, default=60)
    ap.add_argument("--per-pair", type=int, default=20)
    ap.add_argument("--general", type=int, default=300)
    args = ap.parse_args()

    tasks: list[tuple[str, dict, int]] = [
        ("_general", {"query.term": "drug repurposing OR drug repositioning OR off-label"},
         args.general)
    ]
    for d in SEED_DRUGS:
        tasks.append((d["drug"], {"query.intr": d["drug"]}, args.per_drug))
        for ind in d["investigated_indications"]:
            tasks.append((f"{d['drug']}|{ind}",
                          {"query.intr": d["drug"], "query.cond": ind}, args.per_pair))

    progress(f"[ctgov] {len(tasks)} searches")
    records: dict[str, dict] = {}
    tags: dict[str, set[str]] = {}

    for n, (tag, params, cap) in enumerate(tasks, 1):
        try:
            studies = search(params, cap)
        except RuntimeError as exc:
            progress(f"  [{n}/{len(tasks)}] {tag}: FAILED {exc}")
            continue
        added = 0
        for s in studies:
            rec = parse_study(s)
            if not rec:
                continue
            if rec["nct_id"] not in records:
                records[rec["nct_id"]] = rec
                added += 1
            tags.setdefault(rec["nct_id"], set()).add(tag)
        progress(f"  [{n}/{len(tasks)}] {tag}: {len(studies)} studies (+{added} new, "
                 f"total {len(records)})")

    rows = []
    for nct, rec in records.items():
        t = sorted(x for x in tags.get(nct, ()) if x != "_general")
        rec["matched_queries"] = t
        rec["query_drugs"] = sorted({x.split("|")[0] for x in t})
        rows.append(rec)

    out = RAW / "ctgov_studies.jsonl"
    n = write_jsonl(out, rows)
    late = sum(1 for r in rows if r["phase_rank"] >= 2)
    progress(f"[ctgov] wrote {n} studies -> {out}  ({late} phase 2+)")


if __name__ == "__main__":
    main()
