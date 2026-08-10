from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.run_gemini_benchmark import (
    MODEL,
    TOOL_GATE_SCRIPT,
    GeminiRunnerError,
    build_blind_payload,
    gemini_command,
    parse_event_stream,
    prepare_isolated_workspace,
    score_runs,
    validate_response,
)
from src.schemas import BenchmarkCase


def cases():
    return [
        BenchmarkCase(
            id="medqa_001",
            source_id="secret-1",
            dataset="MedQA",
            split="test",
            category="fixture",
            question="Fixture one?",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
            source_metadata={"secret": "gold"},
        ),
        BenchmarkCase(
            id="medmcqa_001",
            source_id="secret-2",
            dataset="MedMCQA",
            split="validation",
            category="fixture",
            question="Fixture two?",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="B",
        ),
    ]


def response(first="A", second="B"):
    return {
        "objective_answers": [
            {"id": "medqa_001", "answer": first, "confidence": 90},
            {"id": "medmcqa_001", "answer": second, "confidence": 80},
        ]
    }


def event_stream(value=None, tools=None, model=MODEL, step_type="agent_response"):
    value = value or response()
    tools = ["search_web", "view_file", "finish"] if tools is None else tools
    return "\n".join(
        [
            json.dumps(
                {
                    "event": "init",
                    "conversation_id": "fixture-conversation",
                    "init": {
                        "model": model,
                        "tools": tools,
                        "permission_mode": "request-review",
                    },
                }
            ),
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "conversation_id": "fixture-conversation",
                        "step_index": 1,
                        "state": "DONE",
                        "step_type": step_type,
                        "text_delta": "done",
                    },
                }
            ),
            json.dumps(
                {
                    "event": "result",
                    "result": {
                        "conversation_id": "fixture-conversation",
                        "status": "SUCCESS",
                        "structured_output": value,
                        "num_turns": 1,
                        "usage": {"thinking_tokens": 10},
                    },
                }
            ),
        ]
    )


class GeminiRunnerTest(unittest.TestCase):
    def test_blind_payload_omits_gold_and_source_fields(self):
        serialized = json.dumps(build_blind_payload(cases()))
        for forbidden in ("correct_answer", "source_metadata", "secret-1", "gold"):
            self.assertNotIn(forbidden, serialized)

    def test_hard_tool_gate_and_isolated_workspace(self):
        self.assertIn('decision = "allow" if name == "finish" else "deny"', TOOL_GATE_SCRIPT)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            prepare_isolated_workspace(root)
            self.assertEqual(
                (root / ".agents" / "mcp_config.json").read_text(),
                '{"mcpServers":{}}\n',
            )
            files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            self.assertEqual(
                files,
                [
                    ".agents",
                    ".agents/hooks.json",
                    ".agents/mcp_config.json",
                    ".agents/scripts",
                    ".agents/scripts/closed_book_tool_gate.py",
                ],
            )
            hooks = json.loads((root / ".agents" / "hooks.json").read_text())
            definition = hooks["closed-book-deny-external-tools"]
            self.assertEqual(definition["PreToolUse"][0]["matcher"], "*")
            self.assertIn("PreInvocation", definition)

    def test_command_pins_exact_model_and_high_effort(self):
        command = gemini_command(
            Path("agy").as_posix(), "fixture", Path("/tmp/x"), 60, 40
        )
        self.assertEqual(command[command.index("--model") + 1], MODEL)
        self.assertEqual(command[command.index("--effort") + 1], "high")
        self.assertIn("--sandbox", command)
        self.assertIn("--disable-slash-commands", command)
        self.assertNotIn("--dangerously-skip-permissions", command)
        schema = json.loads(command[command.index("--json-schema") + 1])
        answers = schema["properties"]["objective_answers"]
        self.assertEqual(answers["minItems"], 40)
        self.assertEqual(answers["maxItems"], 40)

    def test_response_requires_exact_coverage(self):
        value = response()
        value["objective_answers"].pop()
        with self.assertRaises(GeminiRunnerError):
            validate_response(value, cases())

    def test_event_audit_accepts_advertised_tools_without_tool_steps(self):
        value = response()
        parsed, audit = parse_event_stream(event_stream(value))
        self.assertEqual(parsed, value)
        self.assertTrue(audit["closed_book_valid"])
        self.assertIn("search_web", audit["tools_advertised_by_harness"])
        self.assertEqual(audit["reported_model"], MODEL)
        self.assertEqual(audit["usage"]["thinking_tokens"], 10)

    def test_event_audit_rejects_tool_step(self):
        with self.assertRaises(GeminiRunnerError):
            parse_event_stream(event_stream(step_type="tool_use"))

    def test_event_audit_rejects_model_substitution(self):
        with self.assertRaises(GeminiRunnerError):
            parse_event_stream(event_stream(model="gemini-3.5-flash-high"))

    def test_scores_two_runs(self):
        summary, rows = score_runs(cases(), [response(), response("B", "B")])
        self.assertEqual(summary["objective"]["run_1_accuracy"], 1.0)
        self.assertEqual(summary["objective"]["run_2_accuracy"], 0.5)
        self.assertEqual(summary["objective"]["answer_consistency"], 0.5)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
