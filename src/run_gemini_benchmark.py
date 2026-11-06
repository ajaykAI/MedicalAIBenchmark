"""Run the 40 objective cases twice with Gemini 4 Argon (High), closed book.

The model receives only question IDs, question text, and A-D choices. Gold
answers are introduced only after both independent generations finish.
"""

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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.benchmark import load_benchmark
from src.io_utils import read_json, sha256_bytes, write_csv, write_json
from src.models import GEMINI
from src.schemas import ALLOWED_ANSWERS, BenchmarkCase


MODEL = GEMINI["model"]
MODEL_DISPLAY_NAME = GEMINI["runner_display_name"]
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

TOOL_GATE_SCRIPT = """#!/usr/bin/env python3
import json
import os
import sys

payload = json.load(sys.stdin)
audit_path = os.environ.get("MEDICAL_BENCHMARK_TOOL_AUDIT")
if "toolCall" not in payload:
    if audit_path:
        with open(audit_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "hook": "PreInvocation",
                "model": payload.get("modelName"),
            }) + "\\n")
    print(json.dumps({"injectSteps": []}))
    raise SystemExit(0)

name = str(payload.get("toolCall", {}).get("name", "unknown"))
decision = "allow" if name == "finish" else "deny"
if audit_path:
    with open(audit_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": name, "decision": decision}) + "\\n")
print(json.dumps({
    "decision": decision,
    "reason": (
        "Internal structured-output completion is allowed."
        if decision == "allow"
        else "Closed-book benchmark: every external tool is disabled."
    ),
}))
"""


class GeminiRunnerError(RuntimeError):
    """Raised when Gemini output is invalid or violates the protocol."""


def objective_cases(cases: Sequence[BenchmarkCase]) -> List[BenchmarkCase]:
    return [case for case in cases if case.is_objective]


def build_blind_payload(cases: Sequence[BenchmarkCase]) -> Dict[str, Any]:
    """Serialize objective questions without gold labels or source metadata."""
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
    questions = payload.get("objective_questions")
    question_count = len(questions) if isinstance(questions, list) else 0
    instructions = """Complete this closed-book medical benchmark.

Mandatory rules:
- Use only your own internal knowledge and reasoning.
- Do not use tools, files, shell commands, web search, browsing, links,
  retrieval, outside references, other agents, or external research.
- All usable information is contained in this prompt.
- The payload contains exactly %d questions. Answer all %d independently and
  exactly once; do not stop early or omit later IDs.
- Return exactly one option label (A, B, C, or D) for each question.
- Confidence must be an integer from 0 through 100.
- Return only the structured response required by the JSON schema.

OBJECTIVE QUESTIONS (contains no answer key or source metadata):
""" % (question_count, question_count)
    return instructions + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def response_schema(expected_count: Optional[int] = None) -> Dict[str, Any]:
    answers: Dict[str, Any] = {
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
    if expected_count is not None:
        answers.update({"minItems": expected_count, "maxItems": expected_count})
    return {
        "type": "object",
        "properties": {
            "objective_answers": answers
        },
        "required": ["objective_answers"],
        "additionalProperties": False,
    }


def gemini_environment() -> Dict[str, str]:
    """Remove API/Vertex billing routes so Antigravity must use saved OAuth."""
    environment = dict(os.environ)
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_QUOTA_PROJECT",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "VERTEXAI_PROJECT",
        "VERTEXAI_LOCATION",
        "CLOUDSDK_CORE_PROJECT",
    ):
        environment.pop(key, None)
    environment.update({"NO_COLOR": "1"})
    return environment


def _oauth_state() -> Dict[str, Any]:
    root = Path.home() / ".gemini"
    onboarding_path = root / "antigravity-cli" / "cache" / "onboarding.json"
    settings_path = root / "antigravity-cli" / "settings.json"
    try:
        onboarding = read_json(onboarding_path)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise GeminiRunnerError(
            "Antigravity consumer OAuth onboarding is incomplete"
        ) from exc
    if (
        not isinstance(onboarding, dict)
        or not onboarding.get("consumerOnboardingComplete")
        or onboarding.get("enterpriseOnboardingComplete")
    ):
        raise GeminiRunnerError(
            "Antigravity must use a consumer Google OAuth login, not a Cloud project"
        )

    settings: Dict[str, Any] = {}
    if settings_path.exists():
        value = read_json(settings_path)
        if not isinstance(value, dict):
            raise GeminiRunnerError("Antigravity settings are malformed")
        settings = value
    if settings.get("enableTelemetry") is not False:
        raise GeminiRunnerError(
            "Antigravity interaction telemetry must be explicitly disabled"
        )
    if settings.get("useG1Credits") is True:
        raise GeminiRunnerError(
            "Antigravity G1 credit fallback must be disabled for this run"
        )
    return {
        "consumer_oauth": True,
        "enterprise_cloud_project": False,
        "telemetry_enabled": False,
        "g1_credit_fallback_enabled": False,
    }


def require_model_access(agy: str) -> Dict[str, Any]:
    auth = _oauth_state()
    completed = subprocess.run(
        [agy, "models"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=gemini_environment(),
    )
    if completed.returncode != 0:
        raise GeminiRunnerError(
            "Could not verify Antigravity model access: %s"
            % completed.stderr[-2000:].strip()
        )
    models = {}
    for line in completed.stdout.splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            models[parts[0]] = parts[1]
    if models.get(MODEL) != MODEL_DISPLAY_NAME:
        raise GeminiRunnerError(
            "%s is not available on the signed-in account" % MODEL_DISPLAY_NAME
        )
    return {**auth, "verified_model": MODEL, "model_display_name": models[MODEL]}


def prepare_isolated_workspace(root: Path) -> None:
    """Create a gold-free workspace with a hard PreToolUse deny gate."""
    agents_dir = root / ".agents"
    scripts_dir = agents_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    gate_path = scripts_dir / "closed_book_tool_gate.py"
    gate_path.write_text(TOOL_GATE_SCRIPT, encoding="utf-8")
    gate_path.chmod(0o700)
    hooks = {
        "closed-book-deny-external-tools": {
            "enabled": True,
            "PreInvocation": [
                {
                    "type": "command",
                    "command": str(gate_path),
                    "timeout": 5,
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": str(gate_path),
                            "timeout": 5,
                        }
                    ],
                }
            ],
        }
    }
    (agents_dir / "hooks.json").write_text(
        json.dumps(hooks, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    (agents_dir / "mcp_config.json").write_text(
        '{"mcpServers":{}}\n', encoding="utf-8"
    )


def gemini_command(
    agy: str,
    prompt: str,
    root: Path,
    timeout_seconds: int,
    expected_count: int,
) -> List[str]:
    return [
        agy,
        "--model",
        MODEL,
        "--effort",
        REASONING_EFFORT,
        "--mode",
        "plan",
        "--sandbox",
        "--new-project",
        "--disable-slash-commands",
        "--log-file",
        str(root / "agy.log"),
        "--output-format",
        "stream-json",
        "--json-schema",
        json.dumps(response_schema(expected_count), separators=(",", ":")),
        "--print-timeout",
        "%ds" % timeout_seconds,
        "--print",
        prompt,
    ]


def parse_event_stream(stdout: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    event_types: Counter[str] = Counter()
    step_types: Counter[str] = Counter()
    initialized_tools: Optional[List[Any]] = None
    reported_model: Optional[str] = None
    permission_mode: Optional[str] = None
    conversation_id: Optional[str] = None
    result_event: Optional[Dict[str, Any]] = None
    response: Optional[Dict[str, Any]] = None
    forbidden_fragments = (
        "tool",
        "browser",
        "command",
        "file",
        "url",
        "web",
        "search",
        "mcp",
        "subagent",
        "image",
        "notebook",
        "resource",
    )

    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GeminiRunnerError(
                "Malformed Gemini event at line %d" % line_number
            ) from exc
        if not isinstance(event, dict):
            raise GeminiRunnerError("Gemini event %d is not an object" % line_number)
        events.append(event)
        event_type = str(event.get("event", "unknown"))
        event_types[event_type] += 1

        if event_type == "init":
            init = event.get("init")
            if not isinstance(init, dict):
                raise GeminiRunnerError("Gemini init event is malformed")
            tools = init.get("tools")
            initialized_tools = list(tools) if isinstance(tools, list) else None
            reported_model = str(init.get("model", "")) or None
            permission_mode = str(init.get("permission_mode", "")) or None
            conversation_id = str(event.get("conversation_id", "")) or None

        if event_type == "step_update":
            update = event.get("step_update")
            if not isinstance(update, dict):
                raise GeminiRunnerError("Gemini step update is malformed")
            step_type = str(update.get("step_type", "unknown")).lower()
            step_types[step_type] += 1
            if any(fragment in step_type for fragment in forbidden_fragments):
                raise GeminiRunnerError(
                    "Closed-book protocol violation: Gemini emitted %s" % step_type
                )
            if any(
                key in update
                for key in (
                    "tool",
                    "tool_call",
                    "tool_name",
                    "tool_input",
                    "subagent",
                )
            ):
                raise GeminiRunnerError(
                    "Closed-book protocol violation: Gemini emitted a tool payload"
                )

        if event_type == "result":
            result = event.get("result")
            if not isinstance(result, dict):
                raise GeminiRunnerError("Gemini result event is malformed")
            result_event = result
            if result.get("status") != "SUCCESS":
                raise GeminiRunnerError("Gemini reported a failed result")
            structured = result.get("structured_output")
            if isinstance(structured, dict):
                response = structured

    if initialized_tools is None:
        raise GeminiRunnerError("Gemini stream did not contain an initialization audit")
    if reported_model != MODEL:
        raise GeminiRunnerError(
            "Gemini reported model %s instead of %s" % (reported_model, MODEL)
        )
    if result_event is None or response is None:
        raise GeminiRunnerError("Gemini stream did not contain a completed response")

    audit = {
        "closed_book_valid": True,
        "tools_advertised_by_harness": initialized_tools,
        "tool_gate_enabled": False,
        "internal_finish_events": [],
        "external_tool_attempts": [],
        "forbidden_tool_events": 0,
        "event_count": len(events),
        "event_types": dict(sorted(event_types.items())),
        "step_types": dict(sorted(step_types.items())),
        "reported_model": reported_model,
        "reported_permission_mode": permission_mode,
        "reported_conversation_id": conversation_id,
        "reasoning_effort": REASONING_EFFORT,
        "usage": result_event.get("usage"),
        "turn_count": result_event.get("num_turns"),
    }
    return response, audit


def validate_response(
    response: Mapping[str, Any], cases: Sequence[BenchmarkCase]
) -> Dict[str, Dict[str, Any]]:
    rows = response.get("objective_answers")
    if not isinstance(rows, list):
        raise GeminiRunnerError("objective_answers must be an array")
    expected = {case.id for case in objective_cases(cases)}
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise GeminiRunnerError("Malformed objective answer")
        case_id = row["id"]
        if case_id in indexed:
            raise GeminiRunnerError("Duplicate objective answer: %s" % case_id)
        if row.get("answer") not in ALLOWED_ANSWERS:
            raise GeminiRunnerError("Invalid answer for %s" % case_id)
        confidence = row.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, int)
            or not 0 <= confidence <= 100
        ):
            raise GeminiRunnerError("Invalid confidence for %s" % case_id)
        indexed[case_id] = dict(row)
    actual = set(indexed)
    if actual != expected:
        raise GeminiRunnerError(
            "Objective case IDs mismatch (missing=%s, unexpected=%s)"
            % (sorted(expected - actual), sorted(actual - expected))
        )
    return indexed


def score_runs(
    cases: Sequence[BenchmarkCase], runs: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(runs) != RUN_COUNT:
        raise GeminiRunnerError("Exactly two runs are required")
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
    return (
        {
            "objective": metrics(rows),
            "by_dataset": {
                dataset: metrics(
                    [row for row in rows if row["dataset"] == dataset]
                )
                for dataset in datasets
            },
        },
        rows,
    )


def run_once(
    agy: str,
    prompt: str,
    cases: Sequence[BenchmarkCase],
    events_path: Path,
    *,
    timeout_seconds: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    gate_events: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="medical-benchmark-gemini-blind-") as name:
        root = Path(name)
        prepare_isolated_workspace(root)
        gate_audit_path = root / "tool_gate_audit.jsonl"
        environment = gemini_environment()
        environment["MEDICAL_BENCHMARK_TOOL_AUDIT"] = str(gate_audit_path)
        completed = subprocess.run(
            gemini_command(
                agy,
                prompt,
                root,
                timeout_seconds,
                len(objective_cases(cases)),
            ),
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
            timeout=timeout_seconds + 60,
            env=environment,
        )
        if gate_audit_path.exists():
            for line_number, line in enumerate(
                gate_audit_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GeminiRunnerError(
                        "Malformed tool-gate audit at line %d" % line_number
                    ) from exc
                if not isinstance(event, dict):
                    raise GeminiRunnerError("Malformed tool-gate audit event")
                gate_events.append(event)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise GeminiRunnerError(
            "Gemini exited with status %d: %s"
            % (completed.returncode, completed.stderr[-4000:].strip())
        )
    response, audit = parse_event_stream(completed.stdout)
    activation_events = [
        event for event in gate_events if event.get("hook") == "PreInvocation"
    ]
    if not activation_events:
        raise GeminiRunnerError("The closed-book PreToolUse gate was not activated")
    if any(event.get("model") != MODEL for event in activation_events):
        raise GeminiRunnerError("The tool gate observed an unexpected model")
    finish_events = [event for event in gate_events if event.get("tool") == "finish"]
    external_attempts = [
        event
        for event in gate_events
        if "tool" in event and event.get("tool") != "finish"
    ]
    if any(event.get("decision") != "allow" for event in finish_events):
        raise GeminiRunnerError("Internal completion was blocked by the tool gate")
    if any(event.get("decision") != "deny" for event in external_attempts):
        raise GeminiRunnerError("An external tool escaped the closed-book gate")
    if external_attempts:
        raise GeminiRunnerError(
            "Closed-book protocol violation: Gemini attempted external tools (%s)"
            % [event.get("tool") for event in external_attempts]
        )
    audit.update(
        {
            "tool_gate_enabled": True,
            "tool_gate_activation_count": len(activation_events),
            "internal_finish_events": finish_events,
            "external_tool_attempts": external_attempts,
        }
    )
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
            raise GeminiRunnerError("Saved run %d metadata does not match" % run_number)
        response = saved.get("responses")
        audit = saved.get("closed_book_audit")
        if not isinstance(response, dict) or not isinstance(audit, dict):
            raise GeminiRunnerError("Saved run %d is malformed" % run_number)
        if (
            not audit.get("closed_book_valid")
            or audit.get("forbidden_tool_events") != 0
            or not audit.get("tool_gate_enabled")
            or audit.get("external_tool_attempts") != []
            or audit.get("reported_model") != MODEL
        ):
            raise GeminiRunnerError("Saved run %d failed its closed-book audit" % run_number)
        validate_response(response, cases)
        return response, audit
    # An event stream alone cannot prove the temporary PreToolUse audit was clean,
    # so a partial run is repeated instead of being trusted.
    return None


def run_benchmark(
    benchmark_path: Path,
    output_dir: Path,
    *,
    agy: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    artifact, cases = load_benchmark(benchmark_path)
    auth = require_model_access(agy)
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
                agy,
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
                "model_display_name": MODEL_DISPLAY_NAME,
                "reasoning_effort": REASONING_EFFORT,
                "run_number": run_number,
                "prompt_sha256": prompt_sha256,
                "closed_book_audit": audit,
                "responses": response,
            },
        )

    conversation_ids = [audit.get("reported_conversation_id") for audit in audits]
    if len(set(conversation_ids)) != RUN_COUNT or None in conversation_ids:
        raise GeminiRunnerError("The two runs did not use independent conversations")

    score_summary, rows = score_runs(cases, runs)
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": artifact["benchmark_id"],
        "protocol_id": artifact.get("protocol_id"),
        "model": MODEL,
        "model_display_name": MODEL_DISPLAY_NAME,
        "reasoning_effort": REASONING_EFFORT,
        "authentication": "Google consumer OAuth via Antigravity",
        "authentication_audit": auth,
        "run_count": RUN_COUNT,
        "prompt_sha256": prompt_sha256,
        "blind_prompt": {
            "gold_answers_included": False,
            "source_metadata_included": False,
            "external_tool_execution_enabled": False,
            "external_tools_advertised_by_harness": True,
            "external_tools_hard_blocked_by_pretool_hook": True,
            "slash_commands_enabled": False,
            "workspace_mcp_servers": [],
        },
        "closed_book_audits": audits,
        **score_summary,
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "objective_results.csv", rows, RESULT_FIELDS)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.json"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results") / GEMINI["directory"]
    )
    parser.add_argument("--agy", default=shutil.which("agy") or "agy")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_benchmark(
        args.benchmark,
        args.output_dir,
        agy=args.agy,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
