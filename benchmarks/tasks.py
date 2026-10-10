"""Benchmark task families, built from the DRE-Corpus already in this repo.

Three families, chosen because each has an **objective** metric (Ground Rule 10:
prefer objective metrics over LLM-as-judge):

``repurposing_label``
    Classify a drug x indication pair as ``approved`` / ``investigational`` /
    ``hypothesis``. Exact match against the derived label in
    ``repurposing_pairs.jsonl``. 140 pairs, 3 classes, and the class balance is
    skewed (106/24/10) so the majority-class baseline must be reported or an
    accuracy of 0.76 looks like competence.

``trial_lookup``
    Extract structured fields (max trial phase, trial count) for a pair.
    Schema-validity rate is measurable independently of correctness, which is
    what Phase 10's before/after grammar comparison needs.

``grounded_qa``
    Answer a question from ``qa_pairs.jsonl`` using retrieved passages, scored
    for groundedness against the gold PMIDs. Retrieval quality is reported
    separately (recall@k) so failures are attributable.

The held-out split is honoured: ``data/partitions/repurposing/test.jsonl``
covers 12 drugs absent from training, so a model cannot have memorised them
from the training partition.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slmflow.agents.schema import enum_schema, object_schema
from slmflow.types import TaskType

__all__ = [
    "BenchmarkItem",
    "TaskFamily",
    "load_family",
    "FAMILIES",
    "REPURPOSING_LABELS",
    "majority_class_baseline",
]

DATA = Path("data")
PROCESSED = DATA / "processed" / "repurposing"
PARTITIONS = DATA / "partitions" / "repurposing"

REPURPOSING_LABELS = ("approved", "investigational", "hypothesis")


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    """One evaluated item: inputs, gold answer, and provenance."""

    item_id: str
    inputs: dict[str, Any]
    gold: str
    task_type: TaskType
    evidence_doc_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskFamily:
    """A named set of items with the prompt and schema they are run under."""

    name: str
    task_type: TaskType
    items: tuple[BenchmarkItem, ...]
    prompt_template: str
    output_schema: dict[str, Any] | None
    metric: str = "exact_match"
    max_output_tokens: int = 32
    needs_retrieval: bool = False

    def __len__(self) -> int:
        """Number of items."""
        return len(self.items)

    def label_distribution(self) -> dict[str, int]:
        """Gold label counts. Print this before believing any accuracy number."""
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.gold] = counts.get(item.gold, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def majority_baseline(self) -> float:
        """Accuracy of always predicting the most common label."""
        return majority_class_baseline(self.items)

    def as_step_config(self) -> dict[str, Any]:
        """The pipeline step config this family should be run under."""
        step: dict[str, Any] = {
            "name": self.name,
            "task_type": str(self.task_type),
            "prompt": self.prompt_template,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.output_schema:
            step["output_schema"] = self.output_schema
        return step


def majority_class_baseline(items: Sequence[BenchmarkItem]) -> float:
    """Accuracy of the trivial majority-class predictor.

    Reported alongside every classification result. On this corpus the
    majority class is 76 percent of the pairs, so a model scoring 0.7 is worse
    than a constant, and omitting this number would hide that.
    """
    if not items:
        return 0.0
    counts: dict[str, int] = {}
    for item in items:
        counts[item.gold] = counts.get(item.gold, 0) + 1
    return max(counts.values()) / len(items)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file, skipping unparseable lines."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The DRE-Corpus ships with this repository under "
            f"data/processed/repurposing/; if data/ is empty, unzip data.zip."
        )
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _as_list(value: Any) -> list[str]:
    """Normalise the corpus's stringified Python lists into real lists."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return []
        return [str(v) for v in parsed] if isinstance(parsed, list) else []
    return []


_LABEL_PROMPT = (
    "You classify drug repurposing evidence.\n"
    "\n"
    "Drug: {drug}\n"
    "Candidate indication: {indication}\n"
    "\n"
    "Choose exactly one label:\n"
    "- approved: the indication is on the FDA label for this drug\n"
    "- investigational: off-label, but a registered clinical trial studies it\n"
    "- hypothesis: off-label, published literature only, no registered trial\n"
)

_TRIAL_PROMPT = (
    "Extract the trial evidence for this drug and indication.\n"
    "\n"
    "Drug: {drug}\n"
    "Candidate indication: {indication}\n"
    "\n"
    "Report the highest clinical trial phase reached (0 if none) and how many "
    "registered trials exist.\n"
)

_QA_PROMPT = (
    "Answer the question using only the evidence above. "
    "If the evidence does not support an answer, say so.\n"
    "\n"
    "Question: {question}\n"
)


def load_repurposing_label(*, split: str = "train", limit: int | None = None) -> TaskFamily:
    """The 3-class classification family.

    Args:
        split: ``train`` uses ``repurposing_pairs.jsonl``; ``test`` uses the
            held-out partition, whose 12 drugs never appear in training.
        limit: Cap the item count while iterating.
    """
    items: list[BenchmarkItem] = []
    if split == "test":
        for record in _read_jsonl(PARTITIONS / "test.jsonl"):
            if record.get("task") != "repurposing_evidence":
                continue
            items.append(
                BenchmarkItem(
                    item_id=str(record.get("qa_id", "")),
                    inputs={
                        "drug": str(record.get("drug", "")),
                        "indication": str(record.get("indication", "")),
                    },
                    gold=str(record.get("label", "")),
                    task_type=TaskType.CLASSIFY,
                    evidence_doc_ids=tuple(_as_list(record.get("evidence_pmids"))),
                    metadata={"hospital": record.get("hospital"), "split": "test"},
                )
            )
    else:
        for record in _read_jsonl(PROCESSED / "repurposing_pairs.jsonl"):
            items.append(
                BenchmarkItem(
                    item_id=str(record.get("pair_id", "")),
                    inputs={
                        "drug": str(record.get("drug", "")),
                        "indication": str(record.get("indication", "")),
                    },
                    gold=str(record.get("label", "")),
                    task_type=TaskType.CLASSIFY,
                    evidence_doc_ids=tuple(_as_list(record.get("evidence_pmids"))),
                    metadata={
                        "hospital": record.get("hospital"),
                        "max_trial_phase": record.get("max_trial_phase"),
                        "n_trials": record.get("n_trials"),
                        "split": "train",
                    },
                )
            )
    if limit:
        items = items[:limit]

    return TaskFamily(
        name="repurposing_label",
        task_type=TaskType.CLASSIFY,
        items=tuple(items),
        prompt_template=_LABEL_PROMPT,
        output_schema=enum_schema(list(REPURPOSING_LABELS), name="label"),
        metric="exact_match",
        max_output_tokens=32,
    )


def load_trial_lookup(*, limit: int | None = None) -> TaskFamily:
    """The structured-extraction family, for schema-validity measurement."""
    items: list[BenchmarkItem] = []
    for record in _read_jsonl(PROCESSED / "repurposing_pairs.jsonl"):
        phase = record.get("max_trial_phase")
        if phase in (None, ""):
            continue
        items.append(
            BenchmarkItem(
                item_id=f"{record.get('pair_id', '')}|trial",
                inputs={
                    "drug": str(record.get("drug", "")),
                    "indication": str(record.get("indication", "")),
                },
                gold=json.dumps(
                    {
                        "max_phase": int(float(phase)),
                        "n_trials": int(record.get("n_trials", 0) or 0),
                    },
                    sort_keys=True,
                ),
                task_type=TaskType.EXTRACT,
                metadata={"split": "train"},
            )
        )
    if limit:
        items = items[:limit]

    return TaskFamily(
        name="trial_lookup",
        task_type=TaskType.EXTRACT,
        items=tuple(items),
        prompt_template=_TRIAL_PROMPT,
        output_schema=object_schema(
            {
                "max_phase": {"type": "integer", "minimum": 0, "maximum": 4},
                "n_trials": {"type": "integer", "minimum": 0},
            },
            required=["max_phase", "n_trials"],
            title="trial_lookup",
        ),
        metric="json_exact_match",
        max_output_tokens=96,
    )


def load_grounded_qa(*, split: str = "train", limit: int | None = None) -> TaskFamily:
    """The RAG family, scored for groundedness against gold PMIDs."""
    source = PARTITIONS / "test.jsonl" if split == "test" else PROCESSED / "qa_pairs.jsonl"
    items: list[BenchmarkItem] = []
    for record in _read_jsonl(source):
        question = str(record.get("question", ""))
        if not question:
            continue
        items.append(
            BenchmarkItem(
                item_id=str(record.get("qa_id", "")),
                inputs={
                    "question": question,
                    "drug": str(record.get("drug", "")),
                    "indication": str(record.get("indication", "")),
                },
                gold=str(record.get("answer", "")),
                task_type=TaskType.QA,
                evidence_doc_ids=tuple(_as_list(record.get("evidence_pmids"))),
                metadata={"task": record.get("task"), "split": split},
            )
        )
    if limit:
        items = items[:limit]

    return TaskFamily(
        name="grounded_qa",
        task_type=TaskType.QA,
        items=tuple(items),
        prompt_template=_QA_PROMPT,
        output_schema=None,
        metric="groundedness",
        max_output_tokens=192,
        needs_retrieval=True,
    )


#: Registry of loaders, so the harness can take a family name on the CLI.
FAMILIES = {
    "repurposing_label": load_repurposing_label,
    "trial_lookup": load_trial_lookup,
    "grounded_qa": load_grounded_qa,
}


def load_family(name: str, **kwargs: Any) -> TaskFamily:
    """Load one family by name.

    Raises:
        KeyError: With the valid names listed.
    """
    try:
        loader = FAMILIES[name]
    except KeyError:
        raise KeyError(
            f"Unknown task family {name!r}. Valid: {', '.join(sorted(FAMILIES))}."
        ) from None
    return loader(**kwargs)
