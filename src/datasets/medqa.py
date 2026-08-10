"""MedQA USMLE four-option test-set loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.io_utils import download_file, iter_jsonl
from src.schemas import BenchmarkCase, SchemaError


def source_url(source: str, revision: str, filename: str) -> str:
    return "https://huggingface.co/datasets/%s/resolve/%s/%s?download=true" % (
        source,
        revision,
        filename,
    )


def acquire(config: Dict[str, Any], destination: Path) -> Path:
    if Path(destination).exists():
        return Path(destination)
    return download_file(
        source_url(config["source"], config["revision"], config["file"]),
        destination,
    )


def load(path: Path, split: str) -> List[BenchmarkCase]:
    cases = []
    for index, row in enumerate(iter_jsonl(path)):
        options = row.get("options")
        answer = str(row.get("answer_idx", "")).upper()
        if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
            continue
        source_id = row.get("id") or "medqa-row-%d" % index
        question = str(row.get("question", "")).strip()
        if not question or answer not in {"A", "B", "C", "D"}:
            continue
        case = BenchmarkCase(
            id="unassigned",
            source_id=str(source_id),
            dataset="MedQA",
            split=split,
            category=str(row.get("meta_info") or "medical_knowledge"),
            question=question,
            choices={key: str(options[key]).strip() for key in "ABCD"},
            correct_answer=answer,
            source_metadata={
                "meta_info": row.get("meta_info"),
                "correct_answer_text": row.get("answer"),
            },
        )
        try:
            case.validate()
        except SchemaError:
            continue
        cases.append(case)
    return cases

