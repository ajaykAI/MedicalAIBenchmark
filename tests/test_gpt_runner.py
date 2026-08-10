from __future__ import annotations

import json
import unittest

from src.run_gpt_benchmark import (
    RunnerError,
    build_blind_payload,
    parse_event_stream,
    score_runs,
    validate_response,
)
from src.schemas import BenchmarkCase, RubricCriterion


def cases():
    return [
        BenchmarkCase(
            id="medqa_001",
            source_id="secret-source-1",
            dataset="MedQA",
            split="test",
            category="fixture",
            question="Fixture objective one?",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
            source_metadata={"correct_answer_text": "a"},
        ),
        BenchmarkCase(
            id="medmcqa_001",
            source_id="secret-source-2",
            dataset="MedMCQA",
            split="validation",
            category="fixture",
            question="Fixture objective two?",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="B",
        ),
        BenchmarkCase(
            id="healthbench_001",
            source_id="secret-health",
            dataset="HealthBench",
            split="test",
            category="fixture",
            question="Fixture health prompt.",
            conversation=[{"role": "user", "content": "Fixture health prompt."}],
            rubrics=[RubricCriterion("Secret rubric", 10)],
        ),
    ]


def response(first="A", second="B"):
    return {
        "objective_answers": [
            {"id": "medqa_001", "answer": first, "confidence": 90},
            {"id": "medmcqa_001", "answer": second, "confidence": 80},
        ],
        "healthbench_responses": [
            {"id": "healthbench_001", "response": "Fixture response.", "confidence": 70}
        ],
    }


class GptRunnerTest(unittest.TestCase):
    def test_blind_payload_omits_gold_rubric_and_source_fields(self):
        payload = build_blind_payload(cases())
        serialized = json.dumps(payload)
        for forbidden in (
            "correct_answer",
            "Secret rubric",
            "secret-source",
            "source_metadata",
            "rubrics",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(len(payload["objective_questions"]), 2)
        self.assertEqual(len(payload["healthbench_cases"]), 1)

    def test_response_requires_exact_case_coverage(self):
        value = response()
        value["objective_answers"].pop()
        with self.assertRaises(RunnerError):
            validate_response(value, cases())

    def test_event_audit_accepts_reasoning_and_answer_only(self):
        value = response()
        stream = "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "fixture"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "reasoning", "text": "fixture"},
                    }
                ),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": json.dumps(value)},
                    }
                ),
                json.dumps(
                    {"type": "turn.completed", "usage": {"input_tokens": 10}}
                ),
            ]
        )
        parsed, audit = parse_event_stream(stream)
        self.assertEqual(parsed, value)
        self.assertTrue(audit["closed_book_valid"])
        self.assertEqual(audit["forbidden_tool_events"], 0)

    def test_event_audit_rejects_any_tool_event(self):
        stream = json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "web_search", "query": "forbidden"},
            }
        )
        with self.assertRaises(RunnerError):
            parse_event_stream(stream)

    def test_event_audit_records_reconnect_after_completed_turn(self):
        value = response()
        stream = "\n".join(
            [
                json.dumps({"type": "error", "message": "Reconnecting..."}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": json.dumps(value)},
                    }
                ),
                json.dumps({"type": "turn.completed", "usage": {}}),
            ]
        )
        parsed, audit = parse_event_stream(stream)
        self.assertEqual(parsed, value)
        self.assertEqual(audit["transport_warnings"], ["Reconnecting..."])
        self.assertTrue(audit["closed_book_valid"])

    def test_event_audit_rejects_incomplete_reconnect(self):
        stream = json.dumps({"type": "error", "message": "Reconnecting..."})
        with self.assertRaises(RunnerError):
            parse_event_stream(stream)

    def test_scores_two_runs_without_exposing_gold(self):
        summary, rows = score_runs(cases(), [response(), response("B", "B")])
        self.assertEqual(summary["objective"]["question_count"], 2)
        self.assertEqual(summary["objective"]["run_1_accuracy"], 1.0)
        self.assertEqual(summary["objective"]["run_2_accuracy"], 0.5)
        self.assertEqual(summary["objective"]["answer_consistency"], 0.5)
        self.assertEqual(len(rows), 2)
        self.assertNotIn("correct_answer", rows[0])


if __name__ == "__main__":
    unittest.main()
