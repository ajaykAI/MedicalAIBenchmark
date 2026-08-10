from __future__ import annotations

import json
import unittest

from src.run_claude_benchmark import (
    ClaudeRunnerError,
    build_blind_payload,
    parse_event_stream,
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


class ClaudeRunnerTest(unittest.TestCase):
    def test_blind_payload_omits_gold_and_source_fields(self):
        serialized = json.dumps(build_blind_payload(cases()))
        for forbidden in ("correct_answer", "source_metadata", "secret-1", "gold"):
            self.assertNotIn(forbidden, serialized)

    def test_response_requires_exact_coverage(self):
        value = response()
        value["objective_answers"].pop()
        with self.assertRaises(ClaudeRunnerError):
            validate_response(value, cases())

    def test_event_audit_accepts_empty_tools_and_structured_result(self):
        value = response()
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "tools": [],
                        "model": "claude-fable-5",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "done"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "structured_output": value,
                        "usage": {"input_tokens": 10},
                    }
                ),
            ]
        )
        parsed, audit = parse_event_stream(stream)
        self.assertEqual(parsed, value)
        self.assertTrue(audit["closed_book_valid"])
        self.assertEqual(audit["tools_available"], [])

    def test_event_audit_allows_internal_structured_output_formatter(self):
        value = response()
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "tools": ["StructuredOutput"],
                        "model": "claude-fable-5",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "StructuredOutput",
                                    "id": "toolu_fixture",
                                    "input": value,
                                }
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_fixture",
                                }
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "structured_output": value,
                    }
                ),
            ]
        )
        parsed, audit = parse_event_stream(stream)
        self.assertEqual(parsed, value)
        self.assertEqual(audit["external_tools_available"], [])
        self.assertEqual(audit["structured_output_events"], 1)
        self.assertEqual(audit["forbidden_tool_events"], 0)

    def test_event_audit_rejects_tool_use(self):
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "tools": [],
                        "model": "claude-fable-5",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "tool_use", "name": "WebSearch"}]
                        },
                    }
                ),
            ]
        )
        with self.assertRaises(ClaudeRunnerError):
            parse_event_stream(stream)

    def test_scores_two_runs(self):
        summary, rows = score_runs(cases(), [response(), response("B", "B")])
        self.assertEqual(summary["objective"]["run_1_accuracy"], 1.0)
        self.assertEqual(summary["objective"]["run_2_accuracy"], 0.5)
        self.assertEqual(summary["objective"]["answer_consistency"], 0.5)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
