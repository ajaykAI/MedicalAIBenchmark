"""Run the 40 objective cases twice with Claude Fable 5, closed book."""

from __future__ import annotations

import argparse
import json
import os
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


MODEL = "claude-fable-5"
REASONING_EFFORT = "high"
RUN_COUNT = 2
RESULT_FIELDS = [
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


class ClaudeRunnerError(RuntimeError):
    """Raised when Claude output is invalid or violates the protocol."""


def objective_cases(cases: Sequence[BenchmarkCase]) -> List[BenchmarkCase]:
    return [case for case in cases if case.is_objective]


def build_blind_payload(cases: Sequence[BenchmarkCase]) -> Dict[str, Any]:
    return {
        "objective_questions": [
            {
                "id": case.id,
                "dataset": case.dataset,
                "question": case.question,
                "choices": case.choices,
            }
            for case in objective_cases(cases)
        ]
    }


def build_prompt(payload: Mapping[str, Any]) -> str:
    instructions = """Complete this closed-book medical benchmark.

Mandatory rules:
- Use only your own internal knowledge and reasoning.
- Do not use tools, files, shell commands, web search, browsing, links,
  retrieval, outside references, other agents, or external research.
- All usable information is contained in this prompt.
- Answer every question independently and exactly once.
- Return exactly one option label (A, B, C, or D) for each question.
- Confidence must be an integer from 0 through 100.
- Return only the structured response required by the JSON schema.

OBJECTIVE QUESTIONS (contains no answer key or source metadata):
"""
    return instructions + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "objective_answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"],
                        },
                        "confidence": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 100,
                        },
                    },
                    "required": ["id", "answer", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["objective_answers"],
        "additionalProperties": False,
    }


def claude_environment() -> Dict[str, str]:
    environment = dict(os.environ)
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_BUG_COMMAND": "1",
        }
    )
    return environment


def require_subscription_auth(claude: str) -> Dict[str, Any]:
    completed = subprocess.run(
        [claude, "auth", "status"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=claude_environment(),
    )
    try:
        status = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeRunnerError("Could not parse Claude authentication status") from exc
    if (
        completed.returncode != 0
        or not status.get("loggedIn")
        or status.get("authMethod") != "claude.ai"
        or not status.get("subscriptionType")
    ):
        raise ClaudeRunnerError(
            "Claude must use a Claude.ai subscription login; API-billed auth is rejected."
        )
    return status


def claude_command(claude: str) -> List[str]:
    system_prompt = (
        "You are a closed-book benchmark respondent. Use internal reasoning only. "
        "Never request or invoke tools or external information."
    )
    return [
        claude,
        "--print",
        "--model",
        MODEL,
        "--effort",
        REASONING_EFFORT,
        "--tools",
        "",
        "--safe-mode",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--no-chrome",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--max-turns",
        "1",
        "--prompt-suggestions",
        "false",
        "--system-prompt",
        system_prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        json.dumps(response_schema(), separators=(",", ":")),
    ]


def parse_event_stream(stdout: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    event_types: Counter[str] = Counter()
    initialized_tools: Optional[List[Any]] = None
    reported_model: Optional[str] = None
    response: Optional[Dict[str, Any]] = None
    result_event: Optional[Dict[str, Any]] = None
    formatting_tool_events = 0
    structured_output_ids = set()
    forbidden_content_types = {
        "tool_use",
        "server_tool_use",
        "web_search_tool_result",
        "tool_result",
        "mcp_tool_use",
    }

    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClaudeRunnerError(
                "Malformed Claude event at line %d" % line_number
            ) from exc
        if not isinstance(event, dict):
            raise ClaudeRunnerError("Claude event %d is not an object" % line_number)
        events.append(event)
        event_type = str(event.get("type", "unknown"))
        event_types[event_type] += 1

        if event_type == "system" and event.get("subtype") == "init":
            tools = event.get("tools")
            initialized_tools = list(tools) if isinstance(tools, list) else None
            reported_model = str(event.get("model", "")) or None

        message = event.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "tool_use" and block.get("name") == "StructuredOutput":
                        formatting_tool_events += 1
                        if isinstance(block.get("id"), str):
                            structured_output_ids.add(block["id"])
                        continue
                    if (
                        block_type == "tool_result"
                        and block.get("tool_use_id") in structured_output_ids
                    ):
                        continue
                    if block_type in forbidden_content_types:
                        raise ClaudeRunnerError(
                            "Closed-book protocol violation: Claude emitted %s (%s)"
                            % (block_type, block.get("name", "unnamed"))
                        )

        if event_type == "result":
            result_event = event
            if event.get("is_error") or event.get("subtype") not in {"success", None}:
                raise ClaudeRunnerError("Claude reported a failed result")
            structured = event.get("structured_output")
            if isinstance(structured, dict):
                response = structured
            elif isinstance(event.get("result"), str):
                try:
                    parsed = json.loads(event["result"])
                except json.JSONDecodeError as exc:
                    raise ClaudeRunnerError("Claude result is not valid JSON") from exc
                if isinstance(parsed, dict):
                    response = parsed

    if initialized_tools is None:
        raise ClaudeRunnerError("Claude stream did not contain an initialization audit")
    external_tools = [tool for tool in initialized_tools if tool != "StructuredOutput"]
    if external_tools:
        raise ClaudeRunnerError(
            "Closed-book protocol violation: external tools were available (%s)"
            % external_tools
        )
    if result_event is None or response is None:
        raise ClaudeRunnerError("Claude stream did not contain a completed response")
    if reported_model and reported_model != MODEL:
        raise ClaudeRunnerError(
            "Claude reported model %s instead of %s" % (reported_model, MODEL)
        )

    audit = {
        "closed_book_valid": True,
        "tools_available": initialized_tools,
        "external_tools_available": external_tools,
        "structured_output_events": formatting_tool_events,
        "forbidden_tool_events": 0,
        "event_count": len(events),
        "event_types": dict(sorted(event_types.items())),
        "reported_model": reported_model,
        "usage": result_event.get("usage"),
        "total_cost_usd": result_event.get("total_cost_usd"),
    }
    return response, audit


def validate_response(
    response: Mapping[str, Any], cases: Sequence[BenchmarkCase]
) -> Dict[str, Dict[str, Any]]:
    rows = response.get("objective_answers")
    if not isinstance(rows, list):
        raise ClaudeRunnerError("objective_answers must be an array")
    expected = {case.id for case in objective_cases(cases)}
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ClaudeRunnerError("Malformed objective answer")
        case_id = row["id"]
        if case_id in indexed:
            raise ClaudeRunnerError("Duplicate objective answer: %s" % case_id)
        if row.get("answer") not in ALLOWED_ANSWERS:
            raise ClaudeRunnerError("Invalid answer for %s" % case_id)
        confidence = row.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int)
            or not 0 <= confidence <= 100
        ):
            raise ClaudeRunnerError("Invalid confidence for %s" % case_id)
        indexed[case_id] = dict(row)
    actual = set(indexed)
    if actual != expected:
        raise ClaudeRunnerError(
            "Objective case IDs mismatch (missing=%s, unexpected=%s)"
            % (sorted(expected - actual), sorted(actual - expected))
        )
    return indexed


def score_runs(
    cases: Sequence[BenchmarkCase], runs: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(runs) != RUN_COUNT:
        raise ClaudeRunnerError("Exactly two runs are required")
    indexed = [validate_response(run, cases) for run in runs]
    rows = []
    for case in objective_cases(cases):
        first = indexed[0][case.id]
        second = indexed[1][case.id]
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
        run_1_correct = sum(bool(row["run_1_correct"]) for row in subset)
        run_2_correct = sum(bool(row["run_2_correct"]) for row in subset)
        consistent = sum(bool(row["consistent"]) for row in subset)
        stable_correct = sum(bool(row["correct_and_consistent"]) for row in subset)
        return {
            "question_count": count,
            "run_1_correct": run_1_correct,
            "run_1_accuracy": run_1_correct / count,
            "run_2_correct": run_2_correct,
            "run_2_accuracy": run_2_correct / count,
            "mean_accuracy": (run_1_correct + run_2_correct) / (2 * count),
            "consistent_answers": consistent,
            "answer_consistency": consistent / count,
            "correct_and_consistent": stable_correct,
            "correct_and_consistent_rate": stable_correct / count,
        }

    datasets = sorted({str(row["dataset"]) for row in rows})
    summary = {
        "objective": metrics(rows),
        "by_dataset": {
            dataset: metrics([row for row in rows if row["dataset"] == dataset])
            for dataset in datasets
        },
    }
    return summary, rows


def run_once(
    claude: str,
    prompt: str,
    cases: Sequence[BenchmarkCase],
    events_path: Path,
    *,
    timeout_seconds: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="medical-benchmark-claude-blind-") as name:
        completed = subprocess.run(
            claude_command(claude),
            input=prompt,
            capture_output=True,
            text=True,
            cwd=Path(name),
            check=False,
            timeout=timeout_seconds,
            env=claude_environment(),
        )
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise ClaudeRunnerError(
            "Claude exited with status %d: %s"
            % (completed.returncode, completed.stderr[-4000:].strip())
        )
    response, audit = parse_event_stream(completed.stdout)
    validate_response(response, cases)
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
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    expected = {
        "benchmark_id": benchmark_id,
        "protocol_id": protocol_id,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "run_number": run_number,
        "prompt_sha256": prompt_sha256,
    }
    if response_path.exists():
        saved = read_json(response_path)
        if not isinstance(saved, dict) or any(
            saved.get(key) != value for key, value in expected.items()
        ):
            raise ClaudeRunnerError("Saved run %d metadata does not match" % run_number)
        response = saved.get("responses")
        audit = saved.get("closed_book_audit")
        if not isinstance(response, dict) or not isinstance(audit, dict):
            raise ClaudeRunnerError("Saved run %d is malformed" % run_number)
        if not audit.get("closed_book_valid") or audit.get("forbidden_tool_events") != 0:
            raise ClaudeRunnerError("Saved run %d failed its closed-book audit" % run_number)
        validate_response(response, cases)
        return response, audit
    if events_path.exists():
        response, audit = parse_event_stream(events_path.read_text(encoding="utf-8"))
        validate_response(response, cases)
        return response, audit
    return None


def run_benchmark(
    benchmark_path: Path,
    output_dir: Path,
    *,
    claude: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    artifact, cases = load_benchmark(benchmark_path)
    auth = require_subscription_auth(claude)
    prompt = build_prompt(build_blind_payload(cases))
    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    audits = []
    for run_number in range(1, RUN_COUNT + 1):
        response_path = output_dir / ("run_%d_responses.json" % run_number)
        events_path = output_dir / ("run_%d_events.jsonl" % run_number)
        saved = load_saved_run(
            response_path,
            events_path,
            cases,
            benchmark_id=artifact["benchmark_id"],
            protocol_id=artifact.get("protocol_id"),
            prompt_sha256=prompt_sha256,
            run_number=run_number,
        )
        if saved is None:
            response, audit = run_once(
                claude,
                prompt,
                cases,
                events_path,
                timeout_seconds=timeout_seconds,
            )
        else:
            response, audit = saved
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

    score_summary, rows = score_runs(cases, runs)
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": artifact["benchmark_id"],
        "protocol_id": artifact.get("protocol_id"),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "authentication": "Claude.ai %s subscription" % auth["subscriptionType"],
        "run_count": RUN_COUNT,
        "prompt_sha256": prompt_sha256,
        "blind_prompt": {
            "gold_answers_included": False,
            "source_metadata_included": False,
            "external_tools_enabled": False,
        },
        "closed_book_audits": audits,
        **score_summary,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "objective_results.csv", rows, RESULT_FIELDS)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/claude-fable-5")
    )
    parser.add_argument("--claude", default=shutil.which("claude") or "claude")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(
        args.benchmark,
        args.output_dir,
        claude=args.claude,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
