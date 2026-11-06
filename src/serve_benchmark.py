"""Serve the frozen benchmark through a local, read-only web viewer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlparse

from src.benchmark import load_benchmark
from src.config import PROJECT_ROOT
from src.io_utils import read_json
from src.models import CURRENT_MODELS


LOGGER = logging.getLogger("medical_benchmark.viewer")
VIEWER_ROOT = PROJECT_ROOT / "viewer"
DEFAULT_ROUND_SLUG = "combined"
DEFAULT_RESULTS_SLUG = "grok-4.5-fast"
HISTORICAL_RESULT_SPECS = (
    {
        "slug": "gpt-5.6-sol",
        "directory": "gpt-5.6-sol",
        "display_name": "GPT-5.6 Sol",
        "reasoning_label": "High reasoning",
        "collection_method": "automated",
        "audit_label": "2/2 audits passed",
    },
    {
        "slug": "claude-fable-5",
        "directory": "claude-fable-5",
        "display_name": "Claude Fable 5",
        "reasoning_label": "High reasoning",
        "collection_method": "automated",
        "audit_label": "2/2 audits passed",
    },
    {
        "slug": "gemini-3.6-flash",
        "model": "gemini-3.6-flash-high",
        "directory": "gemini-3.6-flash",
        "display_name": "Gemini 3.6 Flash",
        "reasoning_label": "High reasoning",
        "collection_method": "automated",
        "audit_label": "2/2 audits passed",
    },
    {
        "slug": "grok-4.5-fast",
        "directory": "grok-4.5-fast-manual",
        "display_name": "Grok 4.5",
        "reasoning_label": "Fast mode",
        "collection_method": "manual",
        "audit_label": "Manual entry",
    },
)
RESULT_SPECS = CURRENT_MODELS + HISTORICAL_RESULT_SPECS
RESPONSE_COLUMNS = {
    "gpt-5.6-sol": ("GPT Pass 1", "GPT Pass 2"),
    "claude-fable-5": ("Fable Pass 1", "Fable Pass 2"),
    "gemini-3.6-flash": ("Gemini Pass 1", "Gemini Pass 2"),
    "grok-4.5-fast": ("Grok Pass 1", "Grok Pass 2"),
}
ROUND_SPECS = {
    "round-1": {
        "slug": "round-1",
        "display_name": "Round 1",
        "subtitle": "Original benchmark · 40 objective questions + 20 HealthBench cases",
        "benchmark_path": PROJECT_ROOT / "data" / "benchmark.json",
        "results_root": PROJECT_ROOT / "results",
        "response_matrix_path": (
            PROJECT_ROOT / "results" / "medical_ai_benchmark_responses.csv"
        ),
        "default_score_view": "adjusted",
        "limitations": {
            "medmcqa_005": {
                "qid": 35,
                "title": "Missing source image",
                "reason": (
                    "The prompt asks about an arrow-marked mandibular muscle, but "
                    "the frozen source record contains no image."
                ),
            }
        },
    },
    "round-2": {
        "slug": "round-2",
        "display_name": "Round 2",
        "subtitle": "New objective sample · 30 MedQA + 10 MedMCQA questions",
        "benchmark_path": PROJECT_ROOT / "data" / "round-2" / "benchmark.json",
        "results_root": PROJECT_ROOT / "results" / "round-2",
        "response_matrix_path": None,
        "default_score_view": "adjusted",
        "limitations": {
            "medmcqa_004": {
                "qid": 34,
                "title": "Ambiguous oxygen-delivery item",
                "reason": (
                    "The keyed answer depends on an unspecified bag-mask setup, "
                    "oxygen flow, reservoir, seal, and patient context; more than "
                    "one option can deliver the highest concentration."
                ),
            },
            "medmcqa_008": {
                "qid": 38,
                "title": "Contradictory hemoptysis item",
                "reason": (
                    "The source explanation says all four interventions may be "
                    "used, so the question has no unique 'not done' answer."
                ),
            },
        },
    },
}
COMBINED_ROUND_SPEC = {
    "slug": "combined",
    "display_name": "Combined",
    "subtitle": "Primary reporting view · both frozen 40-question objective rounds",
    "default_score_view": "adjusted",
}
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/questions": ("question-bank.html", "text/html; charset=utf-8"),
    "/question-bank.html": ("question-bank.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/question-bank.js": ("question-bank.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/vendor/plotly-basic-3.7.0.min.js": (
        "vendor/plotly-basic-3.7.0.min.js",
        "text/javascript; charset=utf-8",
    ),
}


def _optional_int(value: Any) -> Any:
    if value in (None, ""):
        return None
    parsed = int(value)
    if not 0 <= parsed <= 100:
        raise ValueError("Confidence must be between 0 and 100")
    return parsed


def _metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    if not count:
        raise ValueError("Cannot score an empty result set")
    run_1_correct = sum(bool(row["run_1_correct"]) for row in rows)
    run_2_correct = sum(bool(row["run_2_correct"]) for row in rows)
    consistent = sum(bool(row["consistent"]) for row in rows)
    stable = sum(bool(row["correct_and_consistent"]) for row in rows)
    return {
        "question_count": count,
        "run_1_correct": run_1_correct,
        "run_1_accuracy": run_1_correct / count,
        "run_2_correct": run_2_correct,
        "run_2_accuracy": run_2_correct / count,
        "mean_accuracy": (run_1_correct + run_2_correct) / (2 * count),
        "consistent_answers": consistent,
        "answer_consistency": consistent / count,
        "correct_and_consistent": stable,
        "correct_and_consistent_rate": stable / count,
    }


def _score_view(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    return {
        "objective": _metrics(rows),
        "by_dataset": {
            dataset: _metrics(
                [row for row in rows if row["dataset"] == dataset]
            )
            for dataset in datasets
        },
    }


def _same_number(first: Any, second: Any) -> bool:
    if isinstance(first, bool) or isinstance(second, bool):
        return first == second
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(float(first), float(second), rel_tol=1e-12, abs_tol=1e-12)
    return first == second


def _validate_summary_metrics(
    summary_metrics: Mapping[str, Any], computed: Mapping[str, Any], slug: str
) -> None:
    for key, value in computed.items():
        if not _same_number(summary_metrics.get(key), value):
            raise ValueError("%s summary metric does not match CSV: %s" % (slug, key))


def _load_response_matrix(
    path: Path, objective_cases: Sequence[Any]
) -> Dict[str, Any]:
    """Load the user-supplied response matrix and verify it against the frozen key."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"qID", "Correct Response", "Notes"}
        for columns in RESPONSE_COLUMNS.values():
            required.update(columns)
        missing = required - fieldnames
        if missing:
            raise ValueError(
                "Response matrix is missing required columns: %s"
                % ", ".join(sorted(missing))
            )
        source_rows = list(reader)

    rows_by_qid: Dict[int, Dict[str, str]] = {}
    for row in source_rows:
        qid_text = (row.get("qID") or "").strip()
        if not qid_text.isdigit():
            continue
        qid = int(qid_text)
        if qid in rows_by_qid:
            raise ValueError("Response matrix has a duplicate qID: %d" % qid)
        rows_by_qid[qid] = row

    expected_qids = set(range(1, len(objective_cases) + 1))
    if set(rows_by_qid) != expected_qids:
        raise ValueError("Response matrix qIDs must be consecutive from 1 to 40")

    model_answers: Dict[str, Dict[str, Dict[str, Any]]] = {
        slug: {} for slug in RESPONSE_COLUMNS
    }
    for qid, case in enumerate(objective_cases, start=1):
        row = rows_by_qid[qid]
        gold = (row.get("Correct Response") or "").strip().upper()
        if gold != case.correct_answer:
            raise ValueError(
                "Response matrix gold answer does not match frozen benchmark at qID %d"
                % qid
            )
        note = (row.get("Notes") or "").strip() or None
        for slug, (first_column, second_column) in RESPONSE_COLUMNS.items():
            first = (row.get(first_column) or "").strip().upper()
            second = (row.get(second_column) or "").strip().upper()
            if first not in {"A", "B", "C", "D"} or second not in {
                "A",
                "B",
                "C",
                "D",
            }:
                raise ValueError(
                    "%s responses at qID %d must be A-D" % (slug, qid)
                )
            model_answers[slug][case.id] = {
                "run_1_answer": first,
                "run_2_answer": second,
                "note": note,
            }

    return {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "question_count": len(objective_cases),
        "models": model_answers,
    }


def _load_result_rows(
    path: Path,
    cases_by_id: Mapping[str, Any],
    matrix_answers: Optional[Mapping[str, Mapping[str, Any]]],
    limitations: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if {row.get("id") for row in source_rows} != set(cases_by_id):
        raise ValueError("Objective result IDs do not match the benchmark")
    source_by_id = {str(row["id"]): row for row in source_rows}

    rows: List[Dict[str, Any]] = []
    for case_id, case in cases_by_id.items():
        row = source_by_id[case_id]
        matrix_row = matrix_answers.get(case_id) if matrix_answers is not None else None
        if matrix_answers is not None and matrix_row is None:
            raise ValueError("Response matrix is missing result ID: %s" % case_id)
        if matrix_row is None:
            first = row["run_1_answer"]
            second = row["run_2_answer"]
            response_note = None
        else:
            first = str(matrix_row["run_1_answer"])
            second = str(matrix_row["run_2_answer"])
            response_note = matrix_row.get("note")
            if row["run_1_answer"] != first or row["run_2_answer"] != second:
                raise ValueError(
                    "Archived model result does not match response matrix for %s"
                    % case_id
                )
        if first not in {"A", "B", "C", "D"} or second not in {"A", "B", "C", "D"}:
            raise ValueError("Objective responses must be A-D")
        if row["dataset"] != case.dataset:
            raise ValueError("Objective result dataset does not match benchmark")
        typed = {
            "id": case_id,
            "dataset": row["dataset"],
            "run_1_answer": first,
            "run_1_confidence": _optional_int(row.get("run_1_confidence")),
            "run_1_correct": row["run_1_correct"].lower() == "true",
            "run_2_answer": second,
            "run_2_confidence": _optional_int(row.get("run_2_confidence")),
            "run_2_correct": row["run_2_correct"].lower() == "true",
            "consistent": row["consistent"].lower() == "true",
            "correct_and_consistent": (
                row["correct_and_consistent"].lower() == "true"
            ),
            "excluded_from_adjusted": case_id in limitations,
            "limitation": limitations.get(case_id),
            "response_note": response_note,
        }
        expected = {
            "run_1_correct": first == case.correct_answer,
            "run_2_correct": second == case.correct_answer,
            "consistent": first == second,
            "correct_and_consistent": first == second == case.correct_answer,
        }
        if any(typed[key] != value for key, value in expected.items()):
            raise ValueError("Objective result flags are invalid for %s" % case_id)
        rows.append(typed)
    return rows


def _round_spec(round_slug: str) -> Mapping[str, Any]:
    spec = ROUND_SPECS.get(round_slug)
    if spec is None:
        raise KeyError("Unknown benchmark round: %s" % round_slug)
    return spec


def _combined_digest(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_benchmark_payload(round_slug: str = DEFAULT_ROUND_SLUG) -> Dict[str, Any]:
    """Load a physical round or build the virtual 80-question combined view."""
    if round_slug != COMBINED_ROUND_SPEC["slug"]:
        artifact, _ = load_benchmark(_round_spec(round_slug)["benchmark_path"])
        return artifact

    artifacts = [
        (slug, load_benchmark(spec["benchmark_path"])[0])
        for slug, spec in ROUND_SPECS.items()
    ]
    cases: List[Dict[str, Any]] = []
    for source_round, artifact in artifacts:
        local_qid = 0
        for source_case in artifact["cases"]:
            if not source_case.get("choices"):
                continue
            local_qid += 1
            case = dict(source_case)
            case["id"] = "%s:%s" % (source_round, source_case["id"])
            case["source_round"] = source_round
            case["source_qid"] = local_qid
            cases.append(case)

    benchmark_ids = [artifact["benchmark_id"] for _, artifact in artifacts]
    protocol_ids = [artifact["protocol_id"] for _, artifact in artifacts]
    return {
        "schema_version": 1,
        "round_id": "combined",
        "benchmark_id": _combined_digest("combined-objective", *benchmark_ids),
        "protocol_id": _combined_digest("combined-protocol", *protocol_ids),
        "component_benchmark_ids": benchmark_ids,
        "case_counts": {"MedQA": 60, "MedMCQA": 20, "HealthBench": 0},
        "cases": cases,
        "evaluation_rules": artifacts[0][1].get("evaluation_rules", []),
        "sources": [
            {
                "round": source_round,
                "benchmark_id": artifact["benchmark_id"],
                "sources": artifact.get("sources", []),
            }
            for source_round, artifact in artifacts
        ],
    }


def load_rounds_payload() -> Dict[str, Any]:
    specs = [COMBINED_ROUND_SPEC, *ROUND_SPECS.values()]
    return {
        "default_round_slug": DEFAULT_ROUND_SLUG,
        "rounds": [
            {
                "slug": spec["slug"],
                "display_name": spec["display_name"],
                "subtitle": spec["subtitle"],
                "default_score_view": spec["default_score_view"],
            }
            for spec in specs
        ],
    }


def _load_combined_results_payload() -> Dict[str, Any]:
    round_payloads = [load_results_payload(slug) for slug in ROUND_SPECS]
    models_by_round = [
        {model["slug"]: model for model in payload["models"]}
        for payload in round_payloads
    ]
    models = []
    for spec in RESULT_SPECS:
        # A combined score requires this exact model in both rounds.
        if not all(spec["slug"] in indexed for indexed in models_by_round):
            continue
        combined_rows: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {}
        offset = 0
        for payload, indexed_models in zip(round_payloads, models_by_round):
            source_round = payload["round"]["slug"]
            source_model = indexed_models[spec["slug"]]
            summary = dict(source_model["summary"])
            summary.pop("healthbench", None)
            for source_qid, source_row in enumerate(
                source_model["objective_results"], start=1
            ):
                combined_qid = offset + source_qid
                row = dict(source_row)
                row["id"] = "%s:%s" % (source_round, source_row["id"])
                row["source_round"] = source_round
                row["source_qid"] = source_qid
                row["combined_qid"] = combined_qid
                if source_row.get("limitation"):
                    row["limitation"] = {
                        **source_row["limitation"],
                        "qid": combined_qid,
                        "source_round": source_round,
                        "source_qid": source_qid,
                    }
                combined_rows.append(row)
            offset += len(source_model["objective_results"])

        adjusted_rows = [
            row for row in combined_rows if not row["excluded_from_adjusted"]
        ]
        combined_spec = dict(spec)
        combined_spec["historical"] = spec in HISTORICAL_RESULT_SPECS
        if combined_spec["collection_method"] == "automated":
            combined_spec["audit_label"] = "4/4 audits passed"
        models.append(
            {
                **combined_spec,
                "summary": summary,
                "raw": _score_view(combined_rows),
                "adjusted": _score_view(adjusted_rows),
                "objective_results": combined_rows,
            }
        )

    known_limitations = []
    offset = 0
    for payload in round_payloads:
        source_round = payload["round"]["slug"]
        round_count = payload["response_source"]["question_count"]
        for limitation in payload["known_limitations"]:
            source_qid = int(limitation["qid"])
            known_limitations.append(
                {
                    **limitation,
                    "id": "%s:%s" % (source_round, limitation["id"]),
                    "qid": offset + source_qid,
                    "source_round": source_round,
                    "source_qid": source_qid,
                }
            )
        offset += round_count

    source_hashes = [
        payload["response_source"]["sha256"] for payload in round_payloads
    ]
    return {
        "round": {
            key: COMBINED_ROUND_SPEC[key]
            for key in ("slug", "display_name", "subtitle")
        },
        "default_score_view": COMBINED_ROUND_SPEC["default_score_view"],
        "default_model_slug": _default_result_slug(models),
        "current_models": _current_model_status(round_payloads),
        "response_source": {
            "kind": "combined validated result sources",
            "source_file": "Round 1 validated sources + Round 2 result archives",
            "sha256": _combined_digest(*source_hashes),
            "question_count": sum(
                payload["response_source"]["question_count"] for payload in round_payloads
            ),
        },
        "models": models,
        "known_limitations": known_limitations,
    }


def load_results_payload(round_slug: str = DEFAULT_ROUND_SLUG) -> Dict[str, Any]:
    if round_slug == COMBINED_ROUND_SPEC["slug"]:
        return _load_combined_results_payload()
    round_spec = _round_spec(round_slug)
    artifact, cases = load_benchmark(round_spec["benchmark_path"])
    objective_sequence = [case for case in cases if case.is_objective]
    objective_cases = {case.id: case for case in objective_sequence}
    limitations = round_spec["limitations"]
    matrix_path = round_spec.get("response_matrix_path")
    matrix = (
        _load_response_matrix(matrix_path, objective_sequence)
        if isinstance(matrix_path, Path)
        else None
    )
    models = []
    archive_digest = hashlib.sha256()
    for spec in RESULT_SPECS:
        root = round_spec["results_root"] / spec["directory"]
        result_path = root / "objective_results.csv"
        summary_path = root / "summary.json"
        if spec in CURRENT_MODELS and not (
            result_path.exists() and summary_path.exists()
        ):
            continue
        archive_digest.update(spec["slug"].encode("utf-8"))
        archive_digest.update(b"\0")
        archive_digest.update(result_path.read_bytes())
        summary = read_json(summary_path)
        expected_model = spec.get("model", spec["slug"])
        if summary.get("model") != expected_model:
            raise ValueError("%s model ID mismatch" % spec["slug"])
        if spec in CURRENT_MODELS:
            _validate_current_summary(summary, spec)
        if summary.get("benchmark_id") != artifact.get("benchmark_id"):
            raise ValueError("%s benchmark ID mismatch" % spec["slug"])
        if summary.get("protocol_id") != artifact.get("protocol_id"):
            raise ValueError("%s protocol ID mismatch" % spec["slug"])
        rows = _load_result_rows(
            result_path,
            objective_cases,
            matrix["models"].get(spec["slug"]) if matrix is not None else None,
            limitations,
        )
        raw = _score_view(rows)
        _validate_summary_metrics(summary.get("objective") or {}, raw["objective"], spec["slug"])
        adjusted_rows = [row for row in rows if not row["excluded_from_adjusted"]]
        adjusted = _score_view(adjusted_rows)
        public_summary = {
            key: summary.get(key)
            for key in (
                "model",
                "model_display_name",
                "reasoning_effort",
                "reasoning_verification",
                "authentication",
                "collection_method",
                "run_count",
                "healthbench",
            )
            if summary.get(key) is not None
        }
        models.append(
            {
                **spec,
                "historical": spec in HISTORICAL_RESULT_SPECS,
                "summary": public_summary,
                "raw": raw,
                "adjusted": adjusted,
                "objective_results": rows,
            }
        )
    current_slugs = {spec["slug"] for spec in CURRENT_MODELS}
    has_current_results = any(model["slug"] in current_slugs for model in models)
    if matrix is not None and not has_current_results:
        response_source = {
            "kind": "response matrix",
            "source_file": matrix["source_file"],
            "sha256": matrix["sha256"],
            "question_count": matrix["question_count"],
        }
    else:
        response_source = {
            "kind": "per-model result archives",
            "source_file": (
                "results/*/objective_results.csv + historical response matrix"
                if matrix is not None else "results/%s/*/objective_results.csv" % round_slug
            ),
            "sha256": (
                _combined_digest(matrix["sha256"], archive_digest.hexdigest())
                if matrix is not None else archive_digest.hexdigest()
            ),
            "question_count": len(objective_sequence),
        }
    return {
        "round": {
            "slug": round_spec["slug"],
            "display_name": round_spec["display_name"],
            "subtitle": round_spec["subtitle"],
        },
        "default_score_view": round_spec["default_score_view"],
        "default_model_slug": _default_result_slug(models),
        "current_models": _current_model_status([{"models": models}]),
        "response_source": response_source,
        "models": models,
        "known_limitations": [
            {"id": case_id, **details}
            for case_id, details in sorted(
                limitations.items(), key=lambda item: int(item[1]["qid"])
            )
        ],
    }


def _default_result_slug(models: Sequence[Mapping[str, Any]]) -> Optional[str]:
    available = {model["slug"] for model in models}
    fallback = DEFAULT_RESULTS_SLUG if DEFAULT_RESULTS_SLUG in available else next(iter(available), None)
    return next(
        (spec["slug"] for spec in CURRENT_MODELS if spec["slug"] in available),
        fallback,
    )


def _current_model_status(round_payloads: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep pending targets visible without manufacturing scores or audit claims."""
    available = [{model["slug"] for model in payload["models"]} for payload in round_payloads]
    targets = []
    for spec in CURRENT_MODELS:
        completed = sum(spec["slug"] in slugs for slugs in available)
        status = "completed" if completed == len(available) else "partial" if completed else "pending"
        targets.append({
            **{key: value for key, value in spec.items() if key != "audit_label"},
            "status": status,
            "completed_rounds": completed,
            "required_rounds": len(available),
        })
    return targets


def _validate_current_summary(summary: Mapping[str, Any], spec: Mapping[str, str]) -> None:
    if summary.get("run_count") != 2:
        raise ValueError("%s requires two completed passes" % spec["slug"])
    if spec["collection_method"] == "automated":
        audits = summary.get("closed_book_audits")
        if not isinstance(audits, list) or len(audits) != 2 or not all(
            isinstance(audit, dict) and audit.get("closed_book_valid") is True
            and audit.get("forbidden_tool_events") == 0 for audit in audits
        ):
            raise ValueError("%s requires two valid closed-book audits" % spec["slug"])
        if summary.get("reasoning_effort") != "high":
            raise ValueError("%s reasoning effort mismatch" % spec["slug"])


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "MedicalBenchmarkViewer/1.0"

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        path = parsed.path
        round_slug = parse_qs(parsed.query).get("round", [DEFAULT_ROUND_SLUG])[0]
        if path == "/api/rounds":
            self._serve_bytes(
                json.dumps(load_rounds_payload(), ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if path == "/api/benchmark":
            self._serve_benchmark(round_slug)
            return
        if path == "/api/results":
            self._serve_results(round_slug)
            return
        static = STATIC_FILES.get(path)
        if static is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        filename, content_type = static
        self._serve_bytes((VIEWER_ROOT / filename).read_bytes(), content_type)

    def _serve_benchmark(self, round_slug: str) -> None:
        try:
            artifact = load_benchmark_payload(round_slug)
        except KeyError as exc:
            self._serve_bytes(
                json.dumps({"error": str(exc)}).encode("utf-8"),
                "application/json; charset=utf-8",
                HTTPStatus.NOT_FOUND,
            )
            return
        except Exception as exc:
            LOGGER.exception("Could not load frozen benchmark")
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            self._serve_bytes(
                payload, "application/json; charset=utf-8", HTTPStatus.INTERNAL_SERVER_ERROR
            )
            return
        payload = json.dumps(artifact, ensure_ascii=False).encode("utf-8")
        self._serve_bytes(payload, "application/json; charset=utf-8")

    def _serve_results(self, round_slug: str) -> None:
        try:
            payload = load_results_payload(round_slug)
        except KeyError as exc:
            self._serve_bytes(
                json.dumps({"error": str(exc)}).encode("utf-8"),
                "application/json; charset=utf-8",
                HTTPStatus.NOT_FOUND,
            )
            return
        except FileNotFoundError:
            self._serve_bytes(
                json.dumps({"error": "No model results are available"}).encode("utf-8"),
                "application/json; charset=utf-8",
                HTTPStatus.NOT_FOUND,
            )
            return
        except Exception as exc:
            LOGGER.exception("Could not load model results")
            self._serve_bytes(
                json.dumps({"error": str(exc)}).encode("utf-8"),
                "application/json; charset=utf-8",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return
        self._serve_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _serve_bytes(
        self,
        content: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    for round_spec in ROUND_SPECS.values():
        benchmark_path = round_spec["benchmark_path"]
        if not benchmark_path.exists():
            raise SystemExit("Frozen benchmark not found: %s" % benchmark_path)
        load_benchmark(benchmark_path)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    LOGGER.info("Benchmark viewer running at http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
