# DRE-Corpus - Drug Repurposing Evidence Corpus

Retrieval and evaluation dataset for the FedSLM **drug repurposing literature
assistant**: mine the literature for evidence that a drug may have effects
beyond its approved use, with verification and citations.

Built 2026-08-18 by `src/fedslm/data/repurposing/download_all.py`. Rerunning that
script reproduces every file below from the live APIs.

## 1. Sources

| Source | API | Key needed | Licence | Records pulled |
|---|---|---|---|---|
| PubMed | NCBI E-utilities | no (optional, raises rate limit) | Public domain metadata; abstracts under publisher copyright, used here for research | 6,493 abstracts |
| ClinicalTrials.gov | API v2 | no | Public domain (NLM) | 4,866 studies |
| openFDA drug labels | api.fda.gov | no | Public domain | 63 labels |

openFDA supplies the **approved** indication text. That is what makes "beyond
its approved use" a checkable claim rather than an assertion.

## 2. Seed drugs

64 drugs with well-documented repurposing literature, partitioned by
therapeutic area to give a genuinely non-IID federated split:

| Hospital | Specialty | Drugs |
|---|---|---|
| A | general medicine / cardiometabolic | metformin, aspirin, colchicine, allopurinol, simvastatin, atorvastatin, dapagliflozin, empagliflozin, liraglutide, semaglutide, spironolactone, losartan, verapamil, fenofibrate, pentoxifylline, ursodeoxycholic acid, doxycycline, cimetidine, digoxin, hydroxychloroquine |
| B | pharmacology / oncology / immunology | thalidomide, methotrexate, rituximab, sirolimus, raloxifene, tamoxifen, celecoxib, disulfiram, mebendazole, niclosamide, itraconazole, auranofin, artesunate, chloroquine, leflunomide, zoledronic acid, baricitinib, tocilizumab, anakinra, dexamethasone, ivermectin, bevacizumab, infliximab, zidovudine |
| C | neurology / psychiatry / paediatrics | sildenafil, minoxidil, propranolol, bupropion, ketamine, naltrexone, amantadine, lithium, valproic acid, topiramate, gabapentin, duloxetine, amitriptyline, fluoxetine, prazosin, memantine, bumetanide, finasteride, dimethyl fumarate, botulinum toxin |

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

- **`passages.jsonl`** (19,378 passages) - the retrieval corpus for the
  Evidence Agent's FAISS index. Chunked to ~1200 characters with 150-character
  overlap, each passage carrying `citation`, `url`, `source`, `drugs[]` and
  `evidence_type`, so every retrieved span can be cited back to a real record.
- **`repurposing_pairs.jsonl`** (140 pairs) - one row per drug x candidate
  indication, with a **derived** label:

| Label | Meaning | Count |
|---|---|---|
| `approved` | indication appears on the FDA label - not repurposing | 24 |
| `investigational` | off-label, but a registered trial studies it | 106 |
| `hypothesis` | off-label, literature only, no trial | 10 |

  Derivation rule: the indication is matched against the drug's FDA label text
  (with a synonym expansion for broad terms like *cancer*). On the label ->
  `approved`. Not on the label but backed by a registered trial ->
  `investigational` - this is the genuine repurposing signal. Literature only ->
  `hypothesis`. Nothing -> `no_evidence`. No label is hand-assigned.

- **`qa_pairs.jsonl`** (333 pairs) - instruction/response pairs for LoRA
  fine-tuning and evaluation. Every answer is templated from the structured
  evidence and carries citations; nothing is model-generated, so the corpus
  cannot inject hallucinations into training.

| Task | What it teaches | Count |
|---|---|---|
| `repurposing_evidence` | answer the beyond-approved-use question with a grounded verdict | 140 |
| `trial_lookup` | name the actual trials, with phase and status | 130 |
| `label_lookup` | state the approved indication - the anchor for off-label reasoning | 63 |

### `data/partitions/repurposing/`

- `test.jsonl` - 64 items covering 12 held-out drugs
  (allopurinol, amitriptyline, baricitinib, bevacizumab, botulinum toxin, cimetidine, dexamethasone, doxycycline, empagliflozin, minoxidil, sildenafil, zoledronic acid). Held out **by drug**, so no drug appears in both train and
  test; the split measures generalisation to an unseen drug rather than
  memorisation.
- `noniid/hospital_{a,b,c}/{train,val}.jsonl` - the realistic federated
  split: each hospital sees only its own therapeutic area.
- `iid/hospital_{a,b,c}/{train,val}.jsonl` - same items shuffled uniformly,
  same per-client sizes. This is the control for PRD experiment E3
  (IID vs non-IID).

| Scheme | Hospital A | Hospital B | Hospital C |
|---|---|---|---|
| noniid | 73 train / 9 val | 93 train / 12 val | 73 train / 9 val |
| iid | 73 train / 9 val | 93 train / 12 val | 73 train / 9 val |

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
- Coverage is bounded by the 64 seed drugs and the per-query caps. This
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
