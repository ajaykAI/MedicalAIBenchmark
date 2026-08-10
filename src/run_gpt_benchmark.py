"""Run the frozen benchmark twice with GPT-5.6 Sol under a closed-book protocol.

Gold answers and HealthBench rubrics are used only after generation by the local
scorer. They are never serialized into the model prompt or its isolated working
directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.benchmark import load_benchmark
from src.io_utils import read_json, sha256_bytes, write_csv, write_json
from src.schemas import ALLOWED_ANSWERS, BenchmarkCase


MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "high"
RUN_COUNT = 2
OBJECTIVE_FIELDS = [
    "id",
    "dataset",
    "run_1_answer",
    "run_1_confidence",
    "run_1_correct",
    "run_2_answer",
    "run_2_confidence",
    "run_2_correct",
    "consistent",
    "correct_and_consistent",
]


class RunnerError(RuntimeError):
    """Raised when a run violates the protocol or produces invalid output."""


def build_blind_payload(cases: Sequence[BenchmarkCase]) -> Dict[str, Any]:
    """Create model input without gold labels, rubrics, or source metadata."""
    objective = []
    healthbench = []
    for case in cases:
        if case.is_objective:
            objective.append(
                {
                    "id": case.id,
                    "dataset": case.dataset,
                    "question": case.question,
                    "choices": case.choices,
                }
            )
        else:
            healthbench.append(
                {
                    "id": case.id,
                    "dataset": case.dataset,
                    "conversation": case.conversation,
                }
            )
    return {"objective_questions": objective, "healthbench_cases": healthbench}


def build_prompt(blind_payload: Mapping[str, Any]) -> str:
    instructions = """You are taking a closed-book medical benchmark.

Mandatory rules:
- Use only your own internal knowledge and reasoning.
- Do not use or invoke any tools, shell commands, files, web search, browsing,
  retrieval, links, external references, other agents, or outside research.
- All information you may use is contained in this prompt.
- Answer every case independently. Do not omit any case.
- For each objective question, return exactly one option label: A, B, C, or D.
- For each HealthBench case, write the helpful response you would give after the
  supplied conversation. Do not discuss this evaluation protocol.
- Confidence is an integer from 0 to 100.

Return only the structured response required by the supplied output schema.

BENCHMARK CASES (contains no answer key or scoring rubric):
"""
    return instructions + json.dumps(
        blind_payload, ensure_ascii=False, separators=(",", ":")
    )


def response_schema() -> Dict[str, Any]:
    answer_item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["id", "answer", "confidence"],
        "additionalProperties": False,
    }
    health_item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "response": {"type": "string"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["id", "response", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "objective_answers": {"type": "array", "items": answer_item},
            "healthbench_responses": {"type": "array", "items": health_item},
        },
        "required": ["objective_answers", "healthbench_responses"],
        "additionalProperties": False,
    }


def codex_command(codex: str, schema_path: Path) -> List[str]:
    """Build an isolated, tool-disabled, subscription-authenticated CLI call."""
    return [
        codex,
        "exec",
        "--model",
        MODEL,
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--json",
        "--output-schema",
        str(schema_path),
        "--config",
        'forced_login_method="chatgpt"',
        "--config",
        'model_reasoning_effort="high"',
        "--config",
        'approval_policy="never"',
        "--config",
        'web_search="disabled"',
        "--config",
        "tools.web_search=false",
        "--config",
        "tools.view_image=false",
        "--config",
        "agents.enabled=false",
        "--config",
        "features.shell_tool=false",
        "--config",
        "features.unified_exec=false",
        "--config",
        "features.skill_mcp_dependency_install=false",
        "--config",
        "features.apps=false",
        "--config",
        "features.computer_use=false",
        "--config",
        "features.browser_use=false",
        "--config",
        "features.browser_use_external=false",
        "--config",
        "features.in_app_browser=false",
        "-",
    ]


def require_chatgpt_auth(codex: str) -> None:
    completed = subprocess.run(
        [codex, "login", "status"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    combined = "%s\n%s" % (completed.stdout, completed.stderr)
    if completed.returncode != 0 or "Logged in using ChatGPT" not in combined:
        raise RunnerError(
            "Codex CLI must be logged in using ChatGPT subscription access; "
            "API-key authentication is intentionally rejected."
        )


def parse_event_stream(stdout: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    events = []
    item_types: Counter[str] = Counter()
    final_text = None
    transport_warnings = []
    turn_completed = False
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunnerError("Malformed Codex event at line %d" % line_number) from exc
        if not isinstance(event, dict):
            raise RunnerError("Codex event %d is not an object" % line_number)
        events.append(event)
        event_type = str(event.get("type", ""))
        if event_type == "turn.failed":
            raise RunnerError("Codex reported a failed run: %s" % event)
        if event_type == "error":
            transport_warnings.append(str(event.get("message", "unknown error")))
        if event_type == "turn.completed":
            turn_completed = True
        item = event.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type", "unknown"))
            item_types[item_type] += 1
            if item_type not in {"reasoning", "agent_message"}:
                raise RunnerError(
                    "Closed-book protocol violation: Codex emitted %s" % item_type
                )
            if event_type == "item.completed" and item_type == "agent_message":
                final_text = item.get("text")
    if not turn_completed:
        raise RunnerError(
            "Codex did not complete the turn%s"
            % (
                ": %s" % "; ".join(transport_warnings)
                if transport_warnings
                else ""
            )
        )
    if not isinstance(final_text, str) or not final_text.strip():
        raise RunnerError("Codex did not emit a final agent message")
    try:
        response = json.loads(final_text)
    except json.JSONDecodeError as exc:
        raise RunnerError("Codex final message is not valid JSON") from exc
    if not isinstance(response, dict):
        raise RunnerError("Codex final response must be an object")
    usage = next(
        (
            event.get("usage")
            for event in reversed(events)
            if event.get("type") == "turn.completed"
        ),
        None,
    )
    audit = {
        "event_count": len(events),
        "item_types": dict(sorted(item_types.items())),
        "forbidden_tool_events": 0,
        "closed_book_valid": True,
        "transport_warnings": transport_warnings,
        "usage": usage,
    }
    return response, audit


def load_saved_run(
    response_path: Path,
    events_path: Path,
    cases: Sequence[BenchmarkCase],
    *,
    benchmark_id: str,
    protocol_id: Any,
    prompt_sha256: str,
    run_number: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resume a validated response or recover one from a completed event stream."""
    if response_path.exists():
        saved = read_json(response_path)
        expected = {
            "benchmark_id": benchmark_id,
            "protocol_id": protocol_id,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "run_number": run_number,
            "prompt_sha256": prompt_sha256,
        }
        if not isinstance(saved, dict) or any(
            saved.get(key) != value for key, value in expected.items()
        ):
            raise RunnerError("Saved run %d metadata does not match" % run_number)
        audit = saved.get("closed_book_audit")
        response = saved.get("responses")
        if not isinstance(audit, dict) or not audit.get("closed_book_valid"):
            raise RunnerError("Saved run %d failed its closed-book audit" % run_number)
        if not isinstance(response, dict):
            raise RunnerError("Saved run %d has no response object" % run_number)
        validate_response(response, cases)
        return response, audit

    if not events_path.exists():
        raise FileNotFoundError(str(events_path))
    response, audit = parse_event_stream(events_path.read_text(encoding="utf-8"))
    validate_response(response, cases)
    write_json(
        response_path,
        {
            "benchmark_id": benchmark_id,
            "protocol_id": protocol_id,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "run_number": run_number,
            "prompt_sha256": prompt_sha256,
            "closed_book_audit": audit,
            "responses": response,
        },
    )
    return response, audit


def _unique_rows(
    rows: Any, expected_ids: Iterable[str], kind: str
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rows, list):
        raise RunnerError("%s must be an array" % kind)
    expected = set(expected_ids)
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise RunnerError("%s contains a malformed row" % kind)
        case_id = row["id"]
        if case_id in indexed:
            raise RunnerError("%s contains duplicate id %s" % (kind, case_id))
        indexed[case_id] = row
    actual = set(indexed)
    if actual != expected:
        raise RunnerError(
            "%s case ids mismatch (missing=%s, unexpected=%s)"
            % (kind, sorted(expected - actual), sorted(actual - expected))
        )
    return indexed


def validate_response(
    response: Mapping[str, Any], cases: Sequence[BenchmarkCase]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    objective_ids = [case.id for case in cases if case.is_objective]
    health_ids = [case.id for case in cases if not case.is_objective]
    objective = _unique_rows(
        response.get("objective_answers"), objective_ids, "objective_answers"
    )
    health = _unique_rows(
        response.get("healthbench_responses"), health_ids, "healthbench_responses"
    )
    for case_id, row in objective.items():
        if row.get("answer") not in ALLOWED_ANSWERS:
            raise RunnerError("Invalid objective answer for %s" % case_id)
        confidence = row.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, int):
            raise RunnerError("Invalid confidence for %s" % case_id)
        if not 0 <= confidence <= 100:
            raise RunnerError("Out-of-range confidence for %s" % case_id)
    for case_id, row in health.items():
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            raise RunnerError("Empty HealthBench response for %s" % case_id)
        confidence = row.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, int):
            raise RunnerError("Invalid confidence for %s" % case_id)
        if not 0 <= confidence <= 100:
            raise RunnerError("Out-of-range confidence for %s" % case_id)
    return {"objective": objective, "healthbench": health}


def run_once(
    codex: str,
    prompt: str,
    cases: Sequence[BenchmarkCase],
    raw_events_path: Path,
    *,
    timeout_seconds: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="medical-benchmark-blind-") as name:
        isolated = Path(name)
        schema_path = isolated / "response_schema.json"
        write_json(schema_path, response_schema())
        completed = subprocess.run(
            codex_command(codex, schema_path),
            input=prompt,
            capture_output=True,
            text=True,
            cwd=isolated,
            check=False,
            timeout=timeout_seconds,
        )
    raw_events_path.parent.mkdir(parents=True, exist_ok=True)
    raw_events_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        error_tail = completed.stderr[-4000:].strip()
        raise RunnerError(
            "Codex exited with status %d: %s" % (completed.returncode, error_tail)
        )
    response, audit = parse_event_stream(completed.stdout)
    validate_response(response, cases)
    return response, audit


def score_runs(
    cases: Sequence[BenchmarkCase], runs: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(runs) != RUN_COUNT:
        raise RunnerError("Exactly two completed runs are required for consistency")
    indexed = [validate_response(run, cases) for run in runs]
    rows = []
    for case in cases:
        if not case.is_objective:
            continue
        first = indexed[0]["objective"][case.id]
        second = indexed[1]["objective"][case.id]
        first_correct = first["answer"] == case.correct_answer
        second_correct = second["answer"] == case.correct_answer
        consistent = first["answer"] == second["answer"]
        rows.append(
            {
                "id": case.id,
                "dataset": case.dataset,
                "run_1_answer": first["answer"],
                "run_1_confidence": first["confidence"],
                "run_1_correct": first_correct,
                "run_2_answer": second["answer"],
                "run_2_confidence": second["confidence"],
                "run_2_correct": second_correct,
                "consistent": consistent,
                "correct_and_consistent": consistent and first_correct,
            }
        )

    def metrics(subset: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        count = len(subset)
        if not count:
            return {"question_count": 0}
        run_1_correct = sum(bool(row["run_1_correct"]) for row in subset)
        run_2_correct = sum(bool(row["run_2_correct"]) for row in subset)
        consistent = sum(bool(row["consistent"]) for row in subset)
        correct_consistent = sum(
            bool(row["correct_and_consistent"]) for row in subset
        )
        return {
            "question_count": count,
            "run_1_correct": run_1_correct,
            "run_1_accuracy": run_1_correct / count,
            "run_2_correct": run_2_correct,
            "run_2_accuracy": run_2_correct / count,
            "mean_accuracy": (run_1_correct + run_2_correct) / (2 * count),
            "consistent_answers": consistent,
            "answer_consistency": consistent / count,
            "correct_and_consistent": correct_consistent,
            "correct_and_consistent_rate": correct_consistent / count,
        }

    datasets = sorted({str(row["dataset"]) for row in rows})
    summary = {
        "objective": metrics(rows),
        "by_dataset": {
            dataset: metrics([row for row in rows if row["dataset"] == dataset])
            for dataset in datasets
        },
        "healthbench": {
            "case_count": sum(not case.is_objective for case in cases),
            "runs_completed": RUN_COUNT,
            "accuracy_scored": False,
            "consistency_scored": False,
            "reason": "HealthBench cases are open-ended and require rubric grading.",
        },
    }
    return summary, rows


def run_benchmark(
    benchmark_path: Path,
    output_dir: Path,
    *,
    codex: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    artifact, cases = load_benchmark(benchmark_path)
    require_chatgpt_auth(codex)
    blind_payload = build_blind_payload(cases)
    prompt = build_prompt(blind_payload)
    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    audits = []
    for run_number in range(1, RUN_COUNT + 1):
        response_path = output_dir / ("run_%d_responses.json" % run_number)
        events_path = output_dir / ("run_%d_events.jsonl" % run_number)
        try:
            response, audit = load_saved_run(
                response_path,
                events_path,
                cases,
                benchmark_id=artifact["benchmark_id"],
                protocol_id=artifact.get("protocol_id"),
                prompt_sha256=prompt_sha256,
                run_number=run_number,
            )
        except FileNotFoundError:
            response, audit = run_once(
                codex,
                prompt,
                cases,
                events_path,
                timeout_seconds=timeout_seconds,
            )
        runs.append(response)
        audits.append(audit)
        write_json(
            response_path,
            {
                "benchmark_id": artifact["benchmark_id"],
                "protocol_id": artifact.get("protocol_id"),
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "run_number": run_number,
                "prompt_sha256": prompt_sha256,
                "closed_book_audit": audit,
                "responses": response,
            },
        )

    score_summary, objective_rows = score_runs(cases, runs)
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": artifact["benchmark_id"],
        "protocol_id": artifact.get("protocol_id"),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "authentication": "ChatGPT subscription",
        "run_count": RUN_COUNT,
        "prompt_sha256": prompt_sha256,
        "blind_prompt": {
            "gold_answers_included": False,
            "healthbench_rubrics_included": False,
            "external_tools_enabled": False,
        },
        "closed_book_audits": audits,
        **score_summary,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(
        output_dir / "objective_results.csv", objective_rows, OBJECTIVE_FIELDS
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/gpt-5.6-sol")
    )
    parser.add_argument("--codex", default=shutil.which("codex") or "codex")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(
        args.benchmark,
        args.output_dir,
        codex=args.codex,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
