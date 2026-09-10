"""Generate DATASET_CARD.md from stats.json, so every number in the card is
traceable to a produced file (PRD section 2.3)."""

from __future__ import annotations

import json

from common import PROCESSED, progress
from drugs import HOSPITAL_LABELS, SEED_DRUGS

TEMPLATE = """# DRE-Corpus - Drug Repurposing Evidence Corpus

Retrieval and evaluation dataset for the FedSLM **drug repurposing literature
assistant**: mine the literature for evidence that a drug may have effects
beyond its approved use, with verification and citations.

Built {built} by `src/fedslm/data/repurposing/download_all.py`. Rerunning that
script reproduces every file below from the live APIs.

## 1. Sources

| Source | API | Key needed | Licence | Records pulled |
|---|---|---|---|---|
| PubMed | NCBI E-utilities | no (optional, raises rate limit) | Public domain metadata; abstracts under publisher copyright, used here for research | {pubmed} abstracts |
| ClinicalTrials.gov | API v2 | no | Public domain (NLM) | {ctgov} studies |
| openFDA drug labels | api.fda.gov | no | Public domain | {openfda} labels |

openFDA supplies the **approved** indication text. That is what makes "beyond
its approved use" a checkable claim rather than an assertion.

## 2. Seed drugs

{n_drugs} drugs with well-documented repurposing literature, partitioned by
therapeutic area to give a genuinely non-IID federated split:

| Hospital | Specialty | Drugs |
|---|---|---|
{drug_table}

## 3. Files

### `data/raw/repurposing/`
Untouched API responses, one JSON object per line.

- `pubmed_abstracts.jsonl` - pmid, title, abstract, journal, year, doi, MeSH
  terms, publication types, `evidence_grade` (`strong` = RCT / trial /
  meta-analysis / systematic review / observational study, else `background`),
  and a preformatted `citation`.
- `ctgov_studies.jsonl` - nct_id, title, conditions, interventions, phase,
  status, enrollment, sponsor, brief summary, primary outcomes, `citation`.
- `openfda_labels.jsonl` - generic and brand names, `indications_and_usage`,
  boxed warning, contraindications.

### `data/processed/repurposing/`

- **`passages.jsonl`** ({passages} passages) - the retrieval corpus for the
  Evidence Agent's FAISS index. Chunked to ~1200 characters with 150-character
  overlap, each passage carrying `citation`, `url`, `source`, `drugs[]` and
  `evidence_type`, so every retrieved span can be cited back to a real record.
- **`repurposing_pairs.jsonl`** ({pairs} pairs) - one row per drug x candidate
  indication, with a **derived** label:

| Label | Meaning | Count |
|---|---|---|
{label_table}

  Derivation rule: the indication is matched against the drug's FDA label text
  (with a synonym expansion for broad terms like *cancer*). On the label ->
  `approved`. Not on the label but backed by a registered trial ->
  `investigational` - this is the genuine repurposing signal. Literature only ->
  `hypothesis`. Nothing -> `no_evidence`. No label is hand-assigned.

- **`qa_pairs.jsonl`** ({qa} pairs) - instruction/response pairs for LoRA
  fine-tuning and evaluation. Every answer is templated from the structured
  evidence and carries citations; nothing is model-generated, so the corpus
  cannot inject hallucinations into training.

| Task | What it teaches | Count |
|---|---|---|
{task_table}

### `data/partitions/repurposing/`

- `test.jsonl` - {test_items} items covering {n_held} held-out drugs
  ({held_out}). Held out **by drug**, so no drug appears in both train and
  test; the split measures generalisation to an unseen drug rather than
  memorisation.
- `noniid/hospital_{{a,b,c}}/{{train,val}}.jsonl` - the realistic federated
  split: each hospital sees only its own therapeutic area.
- `iid/hospital_{{a,b,c}}/{{train,val}}.jsonl` - same items shuffled uniformly,
  same per-client sizes. This is the control for PRD experiment E3
  (IID vs non-IID).

| Scheme | Hospital A | Hospital B | Hospital C |
|---|---|---|---|
{partition_table}

## 4. How it feeds the four FedSLM components

| Component | Uses |
|---|---|
| Clinical Agent | `qa_pairs.jsonl` as the fine-tuning target |
| Evidence Agent | FAISS index over `passages.jsonl`; retrieved passages carry PMIDs / NCT IDs for the citation trail |
| Safety Agent | `boxed_warning` and `contraindications` from `openfda_labels.jsonl`; `no_evidence` pairs are the cases where the honest answer is refusal |
| Verifier | `label` in `repurposing_pairs.jsonl` is the ground truth for PASS / REVIEW / FAIL - claiming a `hypothesis` pair is established evidence should FAIL |

The `hypothesis` and `no_evidence` rows matter as much as the positives: they
are the cases where a confident-sounding answer is wrong, which is exactly what
the reliability layer is meant to catch.

## 5. Limitations

- Abstracts only, not full text. PubMed does not serve full text through
  E-utilities, and full text is mostly not openly licensed.
- Label matching is string-based with a small synonym map. It will miss
  indications phrased unusually on a label, which pushes some `approved` pairs
  into `investigational`. Spot-check before quoting per-pair figures.
- Trial registration is not evidence of efficacy. A phase 2 record means the
  question was asked, not answered. `has_results` and `why_stopped` are carried
  through so the verifier can distinguish the two.
- `evidence_grade` comes from PubMed publication types, which are assigned by
  indexers and are occasionally wrong or absent on very recent records.
- Coverage is bounded by the {n_drugs} seed drugs and the per-query caps. This
  is a demonstration corpus, not a systematic review.
- One seed drug (niclosamide) has no openFDA label record, so its pairs can
  only reach `investigational` or `hypothesis` - there is no label text to
  check them against. Any drug without a label is in the same position.
- **Not clinical advice.** Research prototype only.

## 6. Reproduce

```bash
cd implementation/src/fedslm/data/repurposing
export NCBI_API_KEY=...        # optional
python download_all.py
```

Counts will drift slightly on rerun as PubMed and ClinicalTrials.gov are
updated daily. `stats.json` records the counts for any given pull.
"""


def main() -> None:
    stats = json.loads((PROCESSED / "stats.json").read_text(encoding="utf-8"))

    by_h = {h: [d["drug"] for d in SEED_DRUGS if d["hospital"] == h] for h in "ABC"}
    drug_table = "\n".join(
        f"| {h} | {HOSPITAL_LABELS[h]} | {', '.join(by_h[h])} |" for h in "ABC")

    meanings = {
        "approved": "indication appears on the FDA label - not repurposing",
        "investigational": "off-label, but a registered trial studies it",
        "hypothesis": "off-label, literature only, no trial",
        "no_evidence": "nothing found in this pull",
    }
    dist = stats["pair_label_distribution"]
    label_table = "\n".join(
        f"| `{k}` | {meanings.get(k, '')} | {dist[k]} |"
        for k in ("approved", "investigational", "hypothesis", "no_evidence") if k in dist)

    task_meanings = {
        "repurposing_evidence": "answer the beyond-approved-use question with a grounded verdict",
        "trial_lookup": "name the actual trials, with phase and status",
        "label_lookup": "state the approved indication - the anchor for off-label reasoning",
    }
    tasks = stats["qa_task_distribution"]
    task_table = "\n".join(f"| `{k}` | {task_meanings.get(k, '')} | {v} |"
                           for k, v in tasks.items())

    parts = stats["partitions"]
    partition_table = "\n".join(
        f"| {scheme} | "
        + " | ".join(f"{parts[scheme][f'hospital_{h.lower()}']['train']} train / "
                     f"{parts[scheme][f'hospital_{h.lower()}']['val']} val" for h in "ABC")
        + " |"
        for scheme in ("noniid", "iid"))

    card = TEMPLATE.format(
        built=stats.get("built", "on the date recorded in stats.json"),
        pubmed=f"{stats['raw']['pubmed_abstracts']:,}",
        ctgov=f"{stats['raw']['ctgov_studies']:,}",
        openfda=stats["raw"]["openfda_labels"],
        n_drugs=len(SEED_DRUGS),
        drug_table=drug_table,
        passages=f"{stats['processed']['passages']:,}",
        pairs=f"{stats['processed']['pairs']:,}",
        qa=f"{stats['processed']['qa_pairs']:,}",
        label_table=label_table,
        task_table=task_table,
        test_items=stats["test_items"],
        n_held=len(stats["held_out_test_drugs"]),
        held_out=", ".join(stats["held_out_test_drugs"]),
        partition_table=partition_table,
    )

    out = PROCESSED / "DATASET_CARD.md"
    out.write_text(card, encoding="utf-8")
    progress(f"[card] wrote {out}")


if __name__ == "__main__":
    main()
