"""Canonical benchmark and result schemas with dependency-free validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA_VERSION = "1.0"
OBJECTIVE_DATASETS = {"MedQA", "MedMCQA"}
ALLOWED_ANSWERS = {"A", "B", "C", "D"}


class SchemaError(ValueError):
    """Raised when a benchmark case or model response is invalid."""


def _nonempty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError("%s must be a non-empty string" % field_name)
    return value.strip()


@dataclass(frozen=True)
class RubricCriterion:
    criterion: str
    points: int
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RubricCriterion":
        criterion = _nonempty_string(value.get("criterion"), "rubric.criterion")
        points = value.get("points")
        if isinstance(points, bool) or not isinstance(points, int):
            raise SchemaError("rubric.points must be an integer")
        tags = value.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
            raise SchemaError("rubric.tags must be a list of strings")
        return cls(criterion=criterion, points=points, tags=list(tags))


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    source_id: str
    dataset: str
    split: str
    category: str
    question: str
    choices: Optional[Dict[str, str]] = None
    correct_answer: Optional[str] = None
    conversation: Optional[List[Dict[str, str]]] = None
    rubrics: Optional[List[RubricCriterion]] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_objective(self) -> bool:
        return self.dataset in OBJECTIVE_DATASETS

    def validate(self) -> None:
        _nonempty_string(self.id, "id")
        _nonempty_string(self.source_id, "source_id")
        _nonempty_string(self.dataset, "dataset")
        _nonempty_string(self.split, "split")
        _nonempty_string(self.category, "category")
        _nonempty_string(self.question, "question")
        if self.is_objective:
            if not isinstance(self.choices, dict) or set(self.choices) != ALLOWED_ANSWERS:
                raise SchemaError("%s must have exactly choices A-D" % self.id)
            for label, text in self.choices.items():
                _nonempty_string(text, "%s.choices.%s" % (self.id, label))
            if self.correct_answer not in ALLOWED_ANSWERS:
                raise SchemaError("%s has an invalid correct_answer" % self.id)
            if self.conversation is not None or self.rubrics is not None:
                raise SchemaError("%s mixes objective and HealthBench fields" % self.id)
        elif self.dataset == "HealthBench":
            if self.choices is not None or self.correct_answer is not None:
                raise SchemaError("%s must not contain objective answer fields" % self.id)
            if not self.conversation:
                raise SchemaError("%s must preserve its conversation" % self.id)
            for message in self.conversation:
                if not isinstance(message, dict):
                    raise SchemaError("%s has a malformed conversation" % self.id)
                if message.get("role") not in {"user", "assistant", "system"}:
                    raise SchemaError("%s has an invalid conversation role" % self.id)
                _nonempty_string(message.get("content"), "%s conversation content" % self.id)
            if not self.rubrics:
                raise SchemaError("%s must preserve physician rubrics" % self.id)
        else:
            raise SchemaError("Unknown dataset for %s: %s" % (self.id, self.dataset))

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        return {key: item for key, item in value.items() if item is not None}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkCase":
        rubrics_value = value.get("rubrics")
        rubrics = None
        if rubrics_value is not None:
            if not isinstance(rubrics_value, list):
                raise SchemaError("rubrics must be a list")
            rubrics = [RubricCriterion.from_dict(item) for item in rubrics_value]
        choices_value = value.get("choices")
        choices = None
        if choices_value is not None:
            if not isinstance(choices_value, dict):
                raise SchemaError("choices must be an object")
            choices = {str(key): str(item) for key, item in choices_value.items()}
        conversation_value = value.get("conversation")
        conversation = None
        if conversation_value is not None:
            if not isinstance(conversation_value, list):
                raise SchemaError("conversation must be a list")
            conversation = [dict(item) for item in conversation_value]
        case = cls(
            id=str(value.get("id", "")),
            source_id=str(value.get("source_id", "")),
            dataset=str(value.get("dataset", "")),
            split=str(value.get("split", "")),
            category=str(value.get("category", "")),
            question=str(value.get("question", "")),
            choices=choices,
            correct_answer=value.get("correct_answer"),
            conversation=conversation,
            rubrics=rubrics,
            source_metadata=dict(value.get("source_metadata") or {}),
        )
        case.validate()
        return case


def validate_benchmark_cases(cases: Sequence[BenchmarkCase]) -> None:
    if not cases:
        raise SchemaError("Benchmark has no cases")
    ids = set()
    source_ids = set()
    normalized_questions = set()
    for case in cases:
        case.validate()
        if case.id in ids:
            raise SchemaError("Duplicate benchmark id: %s" % case.id)
        source_key = (case.dataset, case.source_id)
        if source_key in source_ids:
            raise SchemaError("Duplicate source id: %s/%s" % source_key)
        question_key = " ".join(case.question.casefold().split())
        if question_key in normalized_questions:
            raise SchemaError("Duplicate normalized question: %s" % case.id)
        ids.add(case.id)
        source_ids.add(source_key)
        normalized_questions.add(question_key)
