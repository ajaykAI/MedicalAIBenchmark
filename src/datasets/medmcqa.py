"""MedMCQA labeled validation-set loader.

The official public test split intentionally withholds answer labels. Using it
would make automatic scoring impossible, so config pins the labeled validation
split and records that decision in benchmark metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.io_utils import fetch_huggingface_rows, iter_jsonl, verify_huggingface_revision
from src.schemas import BenchmarkCase, SchemaError


def acquire(config: Dict[str, Any], destination: Path) -> Path:
    if Path(destination).exists():
        return Path(destination)
    verify_huggingface_revision(config["source"], config["revision"])
    return fetch_huggingface_rows(
        config["source"], config.get("config", "default"), config["split"], destination
    )


def load(path: Path, split: str) -> List[BenchmarkCase]:
    labels = {0: "A", 1: "B", 2: "C", 3: "D", "0": "A", "1": "B", "2": "C", "3": "D"}
    cases = []
    for index, row in enumerate(iter_jsonl(path)):
        answer = labels.get(row.get("cop"))
        if answer is None:
            continue
        question = str(row.get("question", "")).strip()
        choices = {
            "A": str(row.get("opa", "")).strip(),
            "B": str(row.get("opb", "")).strip(),
            "C": str(row.get("opc", "")).strip(),
            "D": str(row.get("opd", "")).strip(),
        }
        if not question or any(not value for value in choices.values()):
            continue
        case = BenchmarkCase(
            id="unassigned",
            source_id=str(row.get("id") or "medmcqa-row-%d" % index),
            dataset="MedMCQA",
            split=split,
            category=str(row.get("subject_name") or "medical_knowledge"),
            question=question,
            choices=choices,
            correct_answer=answer,
            source_metadata={
                "subject_name": row.get("subject_name"),
                "topic_name": row.get("topic_name"),
                "choice_type": row.get("choice_type"),
            },
        )
        try:
            case.validate()
        except SchemaError:
            continue
        cases.append(case)
    return cases

