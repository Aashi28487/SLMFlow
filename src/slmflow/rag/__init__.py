"""Phase 9: local retrieval over the DRE-Corpus. Optional extra ``slmflow[rag]``.

``BM25Retriever`` is pure Python and always available; ``DenseRetriever`` needs
the extra and is imported lazily.
"""

from slmflow.rag.store import (
    BM25Retriever,
    Passage,
    RetrievalReport,
    load_passages,
    recall_at_k,
)

__all__ = [
    "BM25Retriever",
    "Passage",
    "RetrievalReport",
    "load_passages",
    "recall_at_k",
]
