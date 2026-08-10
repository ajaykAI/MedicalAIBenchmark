"""OpenAI HealthBench loader that preserves conversations and rubrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.datasets.medqa import source_url
from src.io_utils import download_file, iter_jsonl
from src.schemas import BenchmarkCase, RubricCriterion, SchemaError


def acquire(config: Dict[str, Any], destination: Path) -> Path:
    if Path(destination).exists():
        return Path(destination)
    return download_file(
        source_url(config["source"], config["revision"], config["file"]),
        destination,
        timeout=300,
    )


def _category(tags: Any) -> str:
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("theme:"):
                return tag.split(":", 1)[1]
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("physician_agreed_category:"):
                return tag.split(":", 1)[1]
    return "healthcare_advice"


def load(path: Path, split: str) -> List[BenchmarkCase]:
    cases = []
    for index, row in enumerate(iter_jsonl(path)):
        prompt = row.get("prompt")
        rubrics_value = row.get("rubrics")
        if not isinstance(prompt, list) or not prompt or not isinstance(rubrics_value, list):
            continue
        try:
            rubrics = [RubricCriterion.from_dict(item) for item in rubrics_value]
        except (SchemaError, TypeError):
            continue
        conversation = []
        for message in prompt:
            if not isinstance(message, dict):
                conversation = []
                break
            conversation.append(
                {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
            )
        if not conversation:
            continue
        last_user = next(
            (item["content"] for item in reversed(conversation) if item["role"] == "user"),
            "",
        )
        case = BenchmarkCase(
            id="unassigned",
            source_id=str(row.get("prompt_id") or "healthbench-row-%d" % index),
            dataset="HealthBench",
            split=split,
            category=_category(row.get("example_tags")),
            question=last_user,
            conversation=conversation,
            rubrics=rubrics,
            source_metadata={
                "example_tags": row.get("example_tags") or [],
                "canary": row.get("canary"),
            },
        )
        try:
            case.validate()
        except SchemaError:
            continue
        cases.append(case)
    return cases

