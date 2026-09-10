"""Download PubMed abstracts about drug repurposing / off-label use.

Uses NCBI E-utilities (esearch + efetch). Free, no key required.
Writes data/raw/repurposing/pubmed_abstracts.jsonl
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET

from common import NCBI_API_KEY, RAW, get, progress, write_jsonl
from drugs import SEED_DRUGS

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_BUCKET = "ncbi"
NCBI_INTERVAL = 0.11 if NCBI_API_KEY else 0.36     # 10/s with key, 3/s without

# Evidence-bearing publication types get priority in the corpus.
STRONG_TYPES = {
    "Randomized Controlled Trial", "Clinical Trial", "Clinical Trial, Phase I",
    "Clinical Trial, Phase II", "Clinical Trial, Phase III", "Clinical Trial, Phase IV",
    "Meta-Analysis", "Systematic Review", "Observational Study", "Multicenter Study",
}

REPURPOSE_HOOK = (
    '("drug repositioning"[MeSH Terms] OR "off-label use"[MeSH Terms] '
    'OR repurpos*[Title/Abstract] OR repositio*[Title/Abstract] '
    'OR "off-label"[Title/Abstract] OR "new indication"[Title/Abstract] '
    'OR "novel indication"[Title/Abstract] OR "beyond its approved"[Title/Abstract])'
)
FILTERS = 'AND hasabstract AND english[Language] AND humans[MeSH Terms]'


def _key():
    return {"api_key": NCBI_API_KEY} if NCBI_API_KEY else {}


def esearch(term: str, retmax: int) -> list[str]:
    params = {"db": "pubmed", "term": term, "retmax": retmax,
              "retmode": "json", "sort": "relevance", **_key()}
    r = get(f"{EUTILS}/esearch.fcgi", params, bucket=NCBI_BUCKET, min_interval=NCBI_INTERVAL)
    return r.json().get("esearchresult", {}).get("idlist", [])


def _text(node) -> str:
    return "".join(node.itertext()).strip() if node is not None else ""


def parse_article(art) -> dict | None:
    cit = art.find("MedlineCitation")
    if cit is None:
        return None
    pmid = _text(cit.find("PMID"))
    a = cit.find("Article")
    if a is None or not pmid:
        return None

    title = _text(a.find("ArticleTitle"))

    parts = []
    for seg in a.findall("./Abstract/AbstractText"):
        label = seg.get("Label")
        body = _text(seg)
        if not body:
            continue
        parts.append(f"{label.title()}: {body}" if label else body)
    abstract = " ".join(parts)
    if not abstract:
        return None

    journal = _text(a.find("./Journal/Title"))
    year = (_text(a.find("./Journal/JournalIssue/PubDate/Year"))
            or _text(a.find("./Journal/JournalIssue/PubDate/MedlineDate"))[:4])

    doi = ""
    for eid in a.findall("ELocationID"):
        if eid.get("EIdType") == "doi":
            doi = _text(eid)
            break

    pubtypes = [_text(p) for p in a.findall("./PublicationTypeList/PublicationType")]
    mesh = [_text(d) for d in cit.findall("./MeshHeadingList/MeshHeading/DescriptorName")]
    keywords = [_text(k) for k in cit.findall("./KeywordList/Keyword")]

    authors = []
    for au in a.findall("./AuthorList/Author")[:3]:
        last, init = _text(au.find("LastName")), _text(au.find("Initials"))
        if last:
            authors.append(f"{last} {init}".strip())

    return {
        "source": "pubmed",
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "doi": doi,
        "publication_types": pubtypes,
        "mesh_terms": mesh,
        "keywords": keywords,
        "authors": authors,
        "evidence_grade": "strong" if set(pubtypes) & STRONG_TYPES else "background",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        "citation": f"{'; '.join(authors) or 'Anon'}. {title} {journal}. {year}. PMID:{pmid}",
    }


def efetch(pmids: list[str], batch: int = 200):
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i + batch]
        params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml", **_key()}
        r = get(f"{EUTILS}/efetch.fcgi", params, bucket=NCBI_BUCKET,
                min_interval=NCBI_INTERVAL, timeout=120)
        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as exc:
            progress(f"    ! XML parse error on batch {i}: {exc}")
            continue
        got = 0
        for art in root.findall("PubmedArticle"):
            rec = parse_article(art)
            if rec:
                got += 1
                yield rec
        progress(f"    efetch {i + len(chunk)}/{len(pmids)} -> +{got} with abstracts")


def build_queries(per_drug: int, per_pair: int) -> list[tuple[str, str, int]]:
    """(tag, query, retmax) triples."""
    q = [("_general", f'{REPURPOSE_HOOK} {FILTERS}', 1200)]
    for d in SEED_DRUGS:
        name = d["drug"]
        q.append((name, f'"{name}"[Title/Abstract] AND {REPURPOSE_HOOK} {FILTERS}', per_drug))
        for ind in d["investigated_indications"]:
            q.append((f"{name}|{ind}",
                      f'"{name}"[Title/Abstract] AND "{ind}"[Title/Abstract] {FILTERS}',
                      per_pair))
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-drug", type=int, default=60)
    ap.add_argument("--per-pair", type=int, default=25)
    args = ap.parse_args()

    queries = build_queries(args.per_drug, args.per_pair)
    progress(f"[pubmed] {len(queries)} search queries "
             f"(api key: {'yes' if NCBI_API_KEY else 'no'})")

    seen: dict[str, set[str]] = {}
    order: list[str] = []
    for n, (tag, term, retmax) in enumerate(queries, 1):
        try:
            ids = esearch(term, retmax)
        except RuntimeError as exc:
            progress(f"  [{n}/{len(queries)}] {tag}: FAILED {exc}")
            continue
        for pid in ids:
            if pid not in seen:
                seen[pid] = set()
                order.append(pid)
            seen[pid].add(tag)
        progress(f"  [{n}/{len(queries)}] {tag}: {len(ids)} hits "
                 f"(unique so far {len(order)})")

    progress(f"[pubmed] fetching {len(order)} unique abstracts")
    rows = []
    for rec in efetch(order):
        tags = sorted(t for t in seen.get(rec["pmid"], ()) if t != "_general")
        rec["matched_queries"] = tags
        rec["query_drugs"] = sorted({t.split("|")[0] for t in tags})
        rows.append(rec)

    out = RAW / "pubmed_abstracts.jsonl"
    n = write_jsonl(out, rows)
    strong = sum(1 for r in rows if r["evidence_grade"] == "strong")
    progress(f"[pubmed] wrote {n} records -> {out}  ({strong} trial/review-grade)")


if __name__ == "__main__":
    main()
