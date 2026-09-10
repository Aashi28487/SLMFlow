"""Phase 9: local retrieval over the DRE-Corpus, fully offline.

Indexes ``data/processed/repurposing/passages.jsonl`` -- the real 19,378-passage
corpus already in this repository. It is not regenerated and no corpus is
invented (``BUILD_BRIEF.md`` Phase 9).

Two retrievers behind one interface:

* :class:`BM25Retriever` -- pure Python, zero dependencies, works on T0. BM25 is
  a genuinely strong lexical baseline on this corpus because the queries carry
  exact drug and indication names, which is precisely where lexical matching
  beats dense retrieval.
* :class:`DenseRetriever` -- ``slmflow[rag]``, sentence-transformers plus FAISS,
  lazily imported.

Retrieval quality is reported **separately** from answer quality (recall@k here,
answer accuracy elsewhere) so a failure is attributable to retrieval or to
generation rather than to "the pipeline".
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from slmflow.errors import ConfigError
from slmflow.telemetry.trace import get_logger

__all__ = [
    "Passage",
    "Retriever",
    "BM25Retriever",
    "DenseRetriever",
    "load_passages",
    "recall_at_k",
    "RetrievalReport",
]

_LOG = get_logger("rag")

DEFAULT_CORPUS = Path("data/processed/repurposing/passages.jsonl")
_TOKEN = re.compile(r"[a-z0-9]+")

#: English function words. Small list on purpose: aggressive stop-listing hurts
#: on this corpus, where "beyond its approved use" carries real signal.
_STOP = frozenset(
    """a an and are as at be been by for from has have in is it its of on or that the
    to was were what which who with does do did there their this these those""".split()
)


def _tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stop-words removed."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass(frozen=True, slots=True)
class Passage:
    """One retrievable passage, carrying the citation it must be traced to."""

    passage_id: str
    text: str
    citation: str = ""
    url: str = ""
    source: str = ""
    doc_id: str = ""
    evidence_type: str = ""
    drugs: tuple[str, ...] = ()

    def cite(self) -> str:
        """A human-readable citation string for the answer."""
        return f"{self.citation} <{self.url}>" if self.url else self.citation


def load_passages(
    path: str | Path = DEFAULT_CORPUS, *, limit: int | None = None
) -> list[Passage]:
    """Load passages from the corpus JSONL.

    Args:
        path: Corpus file.
        limit: Read at most this many passages. Use a small limit while
            iterating: the full corpus is 19k passages and indexing it costs
            real seconds on T1.

    Raises:
        ConfigError: When the corpus file is absent, with the unzip hint.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"Corpus not found at {p}. It ships with this repository under "
            f"data/processed/repurposing/; if data/ is empty, unzip data.zip first.",
            path=str(p),
        )

    passages: list[Passage] = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            drugs = record.get("drugs")
            if isinstance(drugs, str):
                try:
                    drugs = json.loads(drugs.replace("'", '"'))
                except json.JSONDecodeError:
                    drugs = []
            passages.append(
                Passage(
                    passage_id=str(record.get("passage_id", "")),
                    text=str(record.get("text", "")),
                    citation=str(record.get("citation", "")),
                    url=str(record.get("url", "")),
                    source=str(record.get("source", "")),
                    doc_id=str(record.get("doc_id", "")),
                    evidence_type=str(record.get("evidence_type", "")),
                    drugs=tuple(str(d) for d in (drugs or [])),
                )
            )
            if limit and len(passages) >= limit:
                break
    return passages


class Retriever(Protocol):
    """A retriever over a passage corpus."""

    name: str

    def search(self, query: str, k: int = 5) -> list[tuple[Passage, float]]:
        """Return the top ``k`` passages with their scores."""
        ...


@dataclass(slots=True)
class BM25Retriever:
    """Okapi BM25, pure Python. The dependency-free default.

    Args:
        passages: Corpus to index.
        k1: Term-frequency saturation.
        b: Length normalisation.

    Example:
        >>> r = BM25Retriever([Passage("p1", "metformin in cancer prevention")])
        >>> r.search("metformin cancer")[0][0].passage_id
        'p1'
    """

    passages: Sequence[Passage]
    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"
    _df: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _tf: list[Counter[str]] = field(default_factory=list, init=False, repr=False)
    _lengths: list[int] = field(default_factory=list, init=False, repr=False)
    _avg_length: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the inverted-index statistics."""
        for passage in self.passages:
            tokens = _tokenise(passage.text)
            counts = Counter(tokens)
            self._tf.append(counts)
            self._lengths.append(len(tokens))
            self._df.update(counts.keys())
        self._avg_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        _LOG.info(
            "bm25 index built",
            extra={
                "event": "index_built",
                "slmflow": {
                    "passages": len(self.passages),
                    "vocab": len(self._df),
                    "avg_len": round(self._avg_length, 1),
                },
            },
        )

    def _idf(self, term: str) -> float:
        """Robertson-Sparck-Jones IDF, floored at zero."""
        n = len(self.passages)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, k: int = 5) -> list[tuple[Passage, float]]:
        """Score every passage against the query and return the top ``k``."""
        terms = _tokenise(query)
        if not terms or not self.passages:
            return []
        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self._tf):
            length = self._lengths[index] or 1
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_length or 1.0)
                )
                score += self._idf(term) * tf * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((score, index))
        scored.sort(reverse=True)
        return [(self.passages[i], s) for s, i in scored[:k]]


@dataclass(slots=True)
class DenseRetriever:
    """Sentence-transformers plus FAISS. Requires ``slmflow[rag]``.

    Kept behind a lazy import so the core stays installable on T0. The
    embedding model is itself a resident model: on a 6 GB VRAM machine it
    competes with the generator for memory, so index-then-release is the right
    pattern and :meth:`release` exists for exactly that.
    """

    passages: Sequence[Passage]
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "dense"
    _index: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)

    def build(self) -> None:
        """Embed the corpus and build the FAISS index.

        Raises:
            BackendUnavailable: When the ``rag`` extra is not installed.
        """
        from slmflow.errors import BackendUnavailable

        try:
            import faiss  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailable(
                "Dense retrieval needs sentence-transformers, faiss and numpy",
                backend="dense_retriever",
                install_hint="pip install 'slmflow[rag]'",
            ) from exc

        self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            [p.text for p in self.passages],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype="float32")
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def search(self, query: str, k: int = 5) -> list[tuple[Passage, float]]:
        """Embed the query and return the top ``k`` passages."""
        if self._index is None:
            self.build()
        import numpy as np  # noqa: PLC0415

        vector = self._model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.asarray(vector, dtype="float32"), k)
        return [
            (self.passages[int(i)], float(s))
            for s, i in zip(scores[0], indices[0], strict=True)
            if 0 <= int(i) < len(self.passages)
        ]

    def release(self) -> None:
        """Drop the embedding model and index to reclaim memory."""
        self._model = None
        self._index = None


# --------------------------------------------------------------------------- #
# Retrieval quality, reported separately from answer quality
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Measured retrieval quality at one ``k``."""

    k: int
    recall: float
    n_queries: int
    mean_hits: float
    retriever: str

    def as_row(self) -> dict[str, object]:
        """Flat dict for ``docs/RESULTS.md``."""
        return {
            "retriever": self.retriever,
            "k": self.k,
            "recall_at_k": round(self.recall, 4),
            "mean_relevant_retrieved": round(self.mean_hits, 3),
            "n_queries": self.n_queries,
        }


def recall_at_k(
    retriever: Retriever,
    queries: Sequence[tuple[str, Sequence[str]]],
    k: int = 5,
) -> RetrievalReport:
    """Measure recall@k against gold document ids.

    A query counts as a hit when *any* gold document appears in the top ``k``,
    which is the right definition here: ``qa_pairs.jsonl`` lists several
    supporting PMIDs and one is enough to ground an answer.

    Args:
        retriever: The retriever under test.
        queries: ``(query_text, gold_doc_ids)`` pairs.
        k: Cutoff.

    Returns:
        A :class:`RetrievalReport`.

    Raises:
        ValueError: When ``queries`` is empty, since recall would be undefined.
    """
    if not queries:
        raise ValueError("recall_at_k needs at least one query.")
    hits = 0
    total_relevant = 0
    for query, gold in queries:
        results = retriever.search(query, k)
        retrieved_docs = {p.doc_id for p, _ in results}
        matched = retrieved_docs & {str(g) for g in gold}
        if matched:
            hits += 1
        total_relevant += len(matched)
    return RetrievalReport(
        k=k,
        recall=hits / len(queries),
        n_queries=len(queries),
        mean_hits=total_relevant / len(queries),
        retriever=getattr(retriever, "name", type(retriever).__name__),
    )


def iter_qa_pairs(
    path: str | Path = "data/processed/repurposing/qa_pairs.jsonl",
) -> Iterator[dict[str, Any]]:
    """Stream the QA pairs, parsing the stringified list fields.

    ``evidence_pmids`` and ``citations`` are stored as Python-repr strings in
    the corpus, so they are normalised to real lists here rather than at every
    call site.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"QA pairs not found at {p}.", path=str(p))
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("citations", "evidence_pmids", "evidence_ncts"):
                value = record.get(key)
                if isinstance(value, str):
                    try:
                        record[key] = json.loads(value.replace("'", '"'))
                    except json.JSONDecodeError:
                        record[key] = []
            yield record
