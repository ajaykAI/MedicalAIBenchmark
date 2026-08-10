"""Loading and integrity verification for the frozen benchmark artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.io_utils import canonical_json_bytes, read_json, sha256_bytes
from src.schemas import BenchmarkCase, SchemaError, validate_benchmark_cases


class BenchmarkIntegrityError(RuntimeError):
    """Raised when the frozen benchmark no longer matches its recorded ID."""


def benchmark_identity_payload(artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": artifact.get("schema_version"),
        "random_seed": artifact.get("random_seed"),
        "sources": artifact.get("sources"),
        "cases": artifact.get("cases"),
    }


def compute_benchmark_id(artifact: Dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(benchmark_identity_payload(artifact)))


def validate_evaluation_rules(rules: Any) -> None:
    if not isinstance(rules, list) or not rules:
        raise BenchmarkIntegrityError("evaluation_rules must be a non-empty list")
    seen_ids = set()
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise BenchmarkIntegrityError(
                "evaluation_rules[%d] must be an object" % index
            )
        rule_id = rule.get("id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise BenchmarkIntegrityError(
                "evaluation_rules[%d].id must be a non-empty string" % index
            )
        if rule_id in seen_ids:
            raise BenchmarkIntegrityError("Duplicate evaluation rule id: %s" % rule_id)
        seen_ids.add(rule_id)
        if not isinstance(rule.get("version"), int) or rule["version"] < 1:
            raise BenchmarkIntegrityError(
                "evaluation_rules[%d].version must be a positive integer" % index
            )
        for field in ("title", "requirement", "enforcement"):
            if not isinstance(rule.get(field), str) or not rule[field].strip():
                raise BenchmarkIntegrityError(
                    "evaluation_rules[%d].%s must be a non-empty string"
                    % (index, field)
                )
        prohibited = rule.get("prohibited")
        if not isinstance(prohibited, list) or not prohibited or not all(
            isinstance(item, str) and item.strip() for item in prohibited
        ):
            raise BenchmarkIntegrityError(
                "evaluation_rules[%d].prohibited must be a non-empty string list"
                % index
            )


def compute_protocol_id(artifact: Dict[str, Any]) -> str:
    rules = artifact.get("evaluation_rules")
    validate_evaluation_rules(rules)
    return sha256_bytes(
        canonical_json_bytes(
            {"benchmark_id": artifact.get("benchmark_id"), "evaluation_rules": rules}
        )
    )


def load_benchmark(path: Path) -> Tuple[Dict[str, Any], List[BenchmarkCase]]:
    artifact = read_json(path)
    if not isinstance(artifact, dict):
        raise BenchmarkIntegrityError("Benchmark root must be an object")
    raw_cases = artifact.get("cases")
    if not isinstance(raw_cases, list):
        raise BenchmarkIntegrityError("Benchmark cases must be a list")
    try:
        cases = [BenchmarkCase.from_dict(value) for value in raw_cases]
        validate_benchmark_cases(cases)
    except (SchemaError, TypeError) as exc:
        raise BenchmarkIntegrityError("Benchmark schema is invalid: %s" % exc) from exc
    expected = artifact.get("benchmark_id")
    actual = compute_benchmark_id(artifact)
    if expected != actual:
        raise BenchmarkIntegrityError(
            "Benchmark ID mismatch: recorded %s, computed %s" % (expected, actual)
        )
    recorded_counts = artifact.get("case_counts") or {}
    actual_counts = {}
    for case in cases:
        actual_counts[case.dataset] = actual_counts.get(case.dataset, 0) + 1
    if recorded_counts != actual_counts:
        raise BenchmarkIntegrityError(
            "Case counts do not match: recorded %r, actual %r"
            % (recorded_counts, actual_counts)
        )
    if "evaluation_rules" in artifact:
        actual_protocol_id = compute_protocol_id(artifact)
        if artifact.get("protocol_id") != actual_protocol_id:
            raise BenchmarkIntegrityError(
                "Protocol ID mismatch: recorded %s, computed %s"
                % (artifact.get("protocol_id"), actual_protocol_id)
            )
    return artifact, cases
