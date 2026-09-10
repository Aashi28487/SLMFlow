"""Assemble the raw PubMed / ClinicalTrials.gov / openFDA pulls into the
artefacts FedSLM actually trains and evaluates on.

Outputs (data/processed/repurposing/):
  passages.jsonl          retrieval corpus for the FAISS index (Evidence Agent)
  repurposing_pairs.jsonl drug x indication with a label derived from evidence
  qa_pairs.jsonl          instruction/response pairs, every answer cited
  stats.json              counts used in the dataset card

Outputs (data/partitions/repurposing/):
  noniid/hospital_{a,b,c}/{train,val}.jsonl
  iid/hospital_{a,b,c}/{train,val}.jsonl
  test.jsonl              held-out drugs, shared by every arm

Labels are *derived*, never asserted by hand:
  approved        the indication appears in the drug's FDA label text
  investigational not on the label, but >=1 registered trial studies it
  hypothesis      not on the label, no trial, but literature discusses it
  no_evidence     nothing found in this pull
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict

from common import PARTITIONS, PROCESSED, RAW, progress, read_jsonl, write_jsonl
from drugs import HOSPITAL_LABELS, SEED_DRUGS

SEED = 42
CHUNK, OVERLAP = 1200, 150
HELD_OUT_PER_HOSPITAL = 4
VAL_FRACTION = 0.12

# Broad seed terms rarely appear verbatim on a label; expand them so the
# "is this already approved?" check is not fooled by vocabulary.
SYNONYMS = {
    "cancer": ["cancer", "carcinoma", "neoplas", "tumor", "tumour", "malignan",
               "lymphoma", "leukemia", "sarcoma", "myeloma"],
    "obesity": ["obesity", "obese", "weight management", "weight reduction"],
    "aging": ["aging", "ageing", "longevity"],
    "heart failure": ["heart failure", "cardiac failure"],
    "chronic kidney disease": ["chronic kidney disease", "renal impairment",
                              "diabetic nephropathy", "ckd"],
    "migraine prophylaxis": ["migraine"],
    "breast cancer prevention": ["breast cancer", "breast carcinoma"],
    "smoking cessation": ["smoking cessation", "nicotine"],
    "androgenetic alopecia": ["alopecia", "hair loss", "hair growth"],
    "treatment-resistant depression": ["depression", "depressive"],
    "post-traumatic stress disorder nightmares": ["post-traumatic stress", "ptsd"],
    "autophagy inhibition": ["autophagy"],
    "neuroprotection": ["neuroprotect"],
}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())


def terms_for(indication: str) -> list[str]:
    return SYNONYMS.get(indication.lower(), [indication.lower()])


def mentions(haystack: str, indication: str) -> bool:
    h = norm(haystack)
    return any(norm(t).strip() in h for t in terms_for(indication) if t.strip())


# ----------------------------------------------------------------- passages

def chunk(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= CHUNK:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK, len(text))
        if end < len(text):                       # break on a sentence boundary
            dot = text.rfind(". ", start + CHUNK // 2, end)
            if dot != -1:
                end = dot + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return [c for c in out if c]


def build_passages(pubmed, ctgov, labels, drug_home) -> list[dict]:
    rows = []
    for rec in pubmed:
        body = f"{rec['title']} {rec['abstract']}"
        drugs = rec.get("query_drugs", [])
        for i, piece in enumerate(chunk(body)):
            rows.append({
                "passage_id": f"pm:{rec['pmid']}:{i}",
                "text": piece,
                "source": "pubmed",
                "doc_id": rec["pmid"],
                "title": rec["title"],
                "year": rec["year"],
                "evidence_type": rec["evidence_grade"],
                "drugs": drugs,
                "hospitals": sorted({drug_home[d] for d in drugs if d in drug_home}),
                "citation": rec["citation"],
                "url": rec["url"],
            })

    for rec in ctgov:
        body = " ".join(filter(None, [
            rec["brief_title"],
            f"Conditions studied: {', '.join(rec['conditions'])}." if rec["conditions"] else "",
            f"Interventions: {', '.join(i['name'] for i in rec['interventions'])}."
            if rec["interventions"] else "",
            f"Phase: {rec['phase']}. Status: {rec['overall_status']}."
            + (f" Enrollment: {rec['enrollment']}." if rec.get("enrollment") else ""),
            rec["brief_summary"],
        ]))
        drugs = rec.get("query_drugs", [])
        for i, piece in enumerate(chunk(body)):
            rows.append({
                "passage_id": f"ct:{rec['nct_id']}:{i}",
                "text": piece,
                "source": "clinicaltrials.gov",
                "doc_id": rec["nct_id"],
                "title": rec["brief_title"],
                "year": (rec.get("start_date") or "")[:4],
                "evidence_type": f"trial_{rec['phase'].lower()}",
                "drugs": drugs,
                "hospitals": sorted({drug_home[d] for d in drugs if d in drug_home}),
                "citation": rec["citation"],
                "url": rec["url"],
            })

    for rec in labels:
        body = (f"FDA-approved indications for {rec['generic_name']}: "
                f"{rec['indications_and_usage']}")
        for i, piece in enumerate(chunk(body)):
            rows.append({
                "passage_id": f"fda:{rec['drug'].replace(' ', '_')}:{i}",
                "text": piece,
                "source": "openfda",
                "doc_id": rec["drug"],
                "title": f"FDA label - {rec['generic_name']}",
                "year": "",
                "evidence_type": "regulatory_label",
                "drugs": [rec["drug"]],
                "hospitals": [rec["hospital"]],
                "citation": rec["citation"],
                "url": rec["url"],
            })
    return rows


# ------------------------------------------------------------------- pairs

def build_pairs(pubmed, ctgov, labels) -> list[dict]:
    label_by_drug = {r["drug"]: r for r in labels}
    pm_by_drug = defaultdict(list)
    for r in pubmed:
        for d in r.get("query_drugs", []):
            pm_by_drug[d].append(r)
    ct_by_drug = defaultdict(list)
    for r in ctgov:
        for d in r.get("query_drugs", []):
            ct_by_drug[d].append(r)

    pairs = []
    for spec in SEED_DRUGS:
        drug, hosp = spec["drug"], spec["hospital"]
        label_rec = label_by_drug.get(drug)
        label_text = label_rec["indications_and_usage"] if label_rec else ""

        for ind in spec["investigated_indications"]:
            trials = [t for t in ct_by_drug[drug]
                      if any(mentions(c, ind) for c in t["conditions"])
                      or mentions(t["brief_title"], ind)]
            papers = [p for p in pm_by_drug[drug]
                      if mentions(p["title"] + " " + p["abstract"], ind)]
            strong = [p for p in papers if p["evidence_grade"] == "strong"]

            on_label = bool(label_text) and mentions(label_text, ind)
            if on_label:
                status = "approved"
            elif trials:
                status = "investigational"
            elif papers:
                status = "hypothesis"
            else:
                status = "no_evidence"

            max_phase = max((t["phase_rank"] for t in trials), default=-1)
            best = sorted(trials, key=lambda t: -t["phase_rank"])[:8]
            top_papers = sorted(papers, key=lambda p: (p["evidence_grade"] != "strong",
                                                       -int(p["year"] or 0)))[:8]

            pairs.append({
                "pair_id": f"{drug.replace(' ', '_')}|{ind.replace(' ', '_')}",
                "drug": drug,
                "indication": ind,
                "hospital": hosp,
                "label": status,
                "on_fda_label": on_label,
                "approved_for": spec["approved_for"],
                "max_trial_phase": max_phase,
                "n_trials": len(trials),
                "n_publications": len(papers),
                "n_strong_publications": len(strong),
                "evidence_ncts": [t["nct_id"] for t in best],
                "evidence_trials": [{"nct_id": t["nct_id"], "title": t["brief_title"],
                                     "phase": t["phase"], "status": t["overall_status"],
                                     "enrollment": t.get("enrollment"),
                                     "url": t["url"]} for t in best],
                "evidence_pmids": [p["pmid"] for p in top_papers],
                "evidence_publications": [{"pmid": p["pmid"], "title": p["title"],
                                           "journal": p["journal"], "year": p["year"],
                                           "grade": p["evidence_grade"],
                                           "url": p["url"]} for p in top_papers],
                "fda_label_excerpt": label_text[:600],
            })
    return pairs


# ---------------------------------------------------------------- QA pairs

def _cite_list(pair, k=3) -> str:
    bits = [f"PMID:{p['pmid']}" for p in pair["evidence_publications"][:k]]
    bits += [f"{t['nct_id']} ({t['phase']})" for t in pair["evidence_trials"][:k]]
    return "; ".join(bits) if bits else "no citable record in this corpus"


def build_qa(pairs, labels) -> list[dict]:
    label_by_drug = {r["drug"]: r for r in labels}
    qa = []

    for p in pairs:
        drug, ind = p["drug"], p["indication"]
        cites = _cite_list(p)
        refs = ([f"https://pubmed.ncbi.nlm.nih.gov/{x}/" for x in p["evidence_pmids"][:3]]
                + [t["url"] for t in p["evidence_trials"][:3]])

        # Q1 - the core repurposing question
        if p["label"] == "approved":
            ans = (f"No - this is not off-label use. {drug.capitalize()} carries an "
                   f"FDA-labelled indication covering {ind}, so evidence in this area "
                   f"describes approved use rather than repurposing. "
                   f"Supporting records: {cites}.")
        elif p["label"] == "investigational":
            phase = max(t["phase"] for t in p["evidence_trials"]) if p["evidence_trials"] else "NA"
            ans = (f"Yes - there is registered clinical evidence beyond the approved label. "
                   f"{drug.capitalize()} is approved for {p['approved_for']}, and {ind} does "
                   f"not appear in its FDA label, yet {p['n_trials']} registered trial(s) "
                   f"study it for this indication (highest phase reached: {phase}), alongside "
                   f"{p['n_publications']} publication(s), {p['n_strong_publications']} of them "
                   f"trial- or review-grade. This is investigational, not approved use. "
                   f"Citations: {cites}.")
        elif p["label"] == "hypothesis":
            ans = (f"Only preliminary. {drug.capitalize()} is approved for "
                   f"{p['approved_for']}; {ind} is absent from its FDA label and this corpus "
                   f"contains no registered trial for that indication. "
                   f"{p['n_publications']} publication(s) discuss the link "
                   f"({p['n_strong_publications']} trial- or review-grade), so the claim is a "
                   f"literature hypothesis, not clinical evidence. Citations: {cites}.")
        else:
            ans = (f"No supporting evidence was retrieved. This corpus contains no FDA-label "
                   f"text, registered trial, or publication linking {drug} to {ind}. "
                   f"The correct answer is that the evidence base does not support a claim "
                   f"either way.")

        qa.append({
            "qa_id": f"{p['pair_id']}|q1",
            "question": f"Is there evidence that {drug} has an effect beyond its approved "
                        f"use, specifically in {ind}?",
            "answer": ans,
            "label": p["label"],
            "drug": drug,
            "indication": ind,
            "hospital": p["hospital"],
            "task": "repurposing_evidence",
            "citations": refs,
            "evidence_pmids": p["evidence_pmids"][:3],
            "evidence_ncts": p["evidence_ncts"][:3],
        })

        # Q2 - trial landscape, only where trials exist
        if p["evidence_trials"]:
            lines = "; ".join(f"{t['nct_id']} - {t['title'][:90]} (phase {t['phase']}, "
                              f"{t['status'].lower()})" for t in p["evidence_trials"][:4])
            qa.append({
                "qa_id": f"{p['pair_id']}|q2",
                "question": f"Which registered clinical trials study {drug} for {ind}?",
                "answer": f"{p['n_trials']} registered trial(s) in this corpus study {drug} "
                          f"for {ind}. The highest-phase records are: {lines}.",
                "label": p["label"],
                "drug": drug,
                "indication": ind,
                "hospital": p["hospital"],
                "task": "trial_lookup",
                "citations": [t["url"] for t in p["evidence_trials"][:4]],
                "evidence_pmids": [],
                "evidence_ncts": p["evidence_ncts"][:4],
            })

    # Q3 - what is the drug actually approved for (grounding / safety anchor)
    for spec in SEED_DRUGS:
        rec = label_by_drug.get(spec["drug"])
        if not rec:
            continue
        excerpt = rec["indications_and_usage"][:700]
        warn = (f" This product carries a boxed warning: "
                f"{rec['boxed_warning'][:280]}" if rec["boxed_warning"] else "")
        qa.append({
            "qa_id": f"{spec['drug'].replace(' ', '_')}|q3",
            "question": f"What is {spec['drug']} approved by the FDA to treat?",
            "answer": f"According to its FDA prescribing information, "
                      f"{rec['generic_name'].lower()} is indicated as follows: "
                      f"{excerpt}{warn}",
            "label": "approved",
            "drug": spec["drug"],
            "indication": spec["approved_for"],
            "hospital": spec["hospital"],
            "task": "label_lookup",
            "citations": [rec["url"]],
            "evidence_pmids": [],
            "evidence_ncts": [],
        })
    return qa


# -------------------------------------------------------------- partitions

def partition(qa, held_out_drugs) -> dict:
    rng = random.Random(SEED)
    test = [q for q in qa if q["drug"] in held_out_drugs]
    pool = [q for q in qa if q["drug"] not in held_out_drugs]

    noniid = defaultdict(list)
    for q in pool:
        noniid[q["hospital"]].append(q)

    shuffled = pool[:]
    rng.shuffle(shuffled)
    sizes = {h: len(v) for h, v in noniid.items()}
    iid, cursor = {}, 0
    for h in ("A", "B", "C"):
        n = sizes.get(h, 0)
        iid[h] = shuffled[cursor:cursor + n]
        cursor += n

    out = {"test": test, "noniid": {}, "iid": {}}
    for scheme, table in (("noniid", noniid), ("iid", iid)):
        for h in ("A", "B", "C"):
            rows = table.get(h, [])[:]
            rng.shuffle(rows)
            cut = max(1, int(len(rows) * VAL_FRACTION)) if rows else 0
            out[scheme][h] = {"val": rows[:cut], "train": rows[cut:]}
    return out


# -------------------------------------------------------------------- main

def main() -> None:
    pubmed = list(read_jsonl(RAW / "pubmed_abstracts.jsonl"))
    ctgov = list(read_jsonl(RAW / "ctgov_studies.jsonl"))
    labels = list(read_jsonl(RAW / "openfda_labels.jsonl"))
    progress(f"[build] raw: {len(pubmed)} abstracts, {len(ctgov)} trials, "
             f"{len(labels)} labels")

    drug_home = {d["drug"]: d["hospital"] for d in SEED_DRUGS}

    passages = build_passages(pubmed, ctgov, labels, drug_home)
    write_jsonl(PROCESSED / "passages.jsonl", passages)
    progress(f"[build] passages: {len(passages)}")

    pairs = build_pairs(pubmed, ctgov, labels)
    write_jsonl(PROCESSED / "repurposing_pairs.jsonl", pairs)
    dist = Counter(p["label"] for p in pairs)
    progress(f"[build] pairs: {len(pairs)}  {dict(dist)}")

    qa = build_qa(pairs, labels)
    write_jsonl(PROCESSED / "qa_pairs.jsonl", qa)
    progress(f"[build] qa: {len(qa)}  {dict(Counter(q['task'] for q in qa))}")

    rng = random.Random(SEED)
    held_out = []
    for h in ("A", "B", "C"):
        drugs_h = sorted(d["drug"] for d in SEED_DRUGS if d["hospital"] == h)
        held_out += rng.sample(drugs_h, min(HELD_OUT_PER_HOSPITAL, len(drugs_h)))
    held_out = set(held_out)

    parts = partition(qa, held_out)
    write_jsonl(PARTITIONS / "test.jsonl", parts["test"])
    for scheme in ("noniid", "iid"):
        for h in ("A", "B", "C"):
            base = PARTITIONS / scheme / f"hospital_{h.lower()}"
            write_jsonl(base / "train.jsonl", parts[scheme][h]["train"])
            write_jsonl(base / "val.jsonl", parts[scheme][h]["val"])
    progress(f"[build] test set: {len(parts['test'])} items from {len(held_out)} held-out drugs")

    stats = {
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "raw": {"pubmed_abstracts": len(pubmed), "ctgov_studies": len(ctgov),
                "openfda_labels": len(labels)},
        "processed": {"passages": len(passages), "pairs": len(pairs), "qa_pairs": len(qa)},
        "pair_label_distribution": dict(dist),
        "qa_task_distribution": dict(Counter(q["task"] for q in qa)),
        "pubmed_evidence_grade": dict(Counter(r["evidence_grade"] for r in pubmed)),
        "trial_phase_distribution": dict(Counter(r["phase"] for r in ctgov)),
        "held_out_test_drugs": sorted(held_out),
        "test_items": len(parts["test"]),
        "partitions": {
            scheme: {f"hospital_{h.lower()}": {
                "train": len(parts[scheme][h]["train"]),
                "val": len(parts[scheme][h]["val"]),
                "specialty": HOSPITAL_LABELS[h],
            } for h in ("A", "B", "C")} for scheme in ("noniid", "iid")
        },
    }
    (PROCESSED / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    progress(f"[build] stats -> {PROCESSED / 'stats.json'}")
    print(json.dumps(stats["processed"] | stats["pair_label_distribution"], indent=2))


if __name__ == "__main__":
    main()
