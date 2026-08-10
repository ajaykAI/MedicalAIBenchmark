from __future__ import annotations

import csv
import unittest

from src.benchmark import load_benchmark
from src.config import PROJECT_ROOT
from src.io_utils import read_json


class Round2GptResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact, cls.cases = load_benchmark(
            PROJECT_ROOT / "data" / "round-2" / "benchmark.json"
        )
        cls.cases_by_id = {case.id: case for case in cls.cases}
        cls.result_root = PROJECT_ROOT / "results" / "round-2" / "gpt-5.6-sol"
        cls.summary = read_json(cls.result_root / "summary.json")
        with (cls.result_root / "objective_results.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_round2_is_new_objective_only_30_10_sample(self) -> None:
        _, prior_cases = load_benchmark(PROJECT_ROOT / "data" / "benchmark.json")
        prior_ids = {(case.dataset, case.source_id) for case in prior_cases}
        round_ids = {(case.dataset, case.source_id) for case in self.cases}
        self.assertEqual(
            self.artifact["case_counts"], {"MedQA": 30, "MedMCQA": 10}
        )
        self.assertEqual(len(self.cases), 40)
        self.assertTrue(all(case.is_objective for case in self.cases))
        self.assertFalse(prior_ids & round_ids)

    def test_two_high_reasoning_closed_book_runs_are_valid(self) -> None:
        self.assertEqual(self.summary["model"], "gpt-5.6-sol")
        self.assertEqual(self.summary["reasoning_effort"], "high")
        self.assertEqual(self.summary["authentication"], "ChatGPT subscription")
        self.assertEqual(self.summary["run_count"], 2)
        self.assertEqual(
            self.summary["benchmark_id"], self.artifact["benchmark_id"]
        )
        self.assertEqual(
            self.summary["protocol_id"], self.artifact["protocol_id"]
        )
        self.assertFalse(self.summary["blind_prompt"]["gold_answers_included"])
        self.assertFalse(self.summary["blind_prompt"]["external_tools_enabled"])
        for audit in self.summary["closed_book_audits"]:
            self.assertTrue(audit["closed_book_valid"])
            self.assertEqual(audit["forbidden_tool_events"], 0)

    def test_saved_rows_and_metrics_match_frozen_key(self) -> None:
        self.assertEqual(len(self.rows), 40)
        self.assertEqual({row["id"] for row in self.rows}, set(self.cases_by_id))
        for row in self.rows:
            case = self.cases_by_id[row["id"]]
            first = row["run_1_answer"]
            second = row["run_2_answer"]
            self.assertIn(first, "ABCD")
            self.assertIn(second, "ABCD")
            self.assertEqual(row["run_1_correct"], str(first == case.correct_answer))
            self.assertEqual(row["run_2_correct"], str(second == case.correct_answer))
            self.assertEqual(row["consistent"], str(first == second))
            self.assertEqual(
                row["correct_and_consistent"],
                str(first == second == case.correct_answer),
            )
        objective = self.summary["objective"]
        self.assertEqual(objective["run_1_correct"], 37)
        self.assertEqual(objective["run_2_correct"], 38)
        self.assertEqual(objective["consistent_answers"], 39)
        self.assertEqual(objective["correct_and_consistent"], 37)


class Round2FableResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact, cls.cases = load_benchmark(
            PROJECT_ROOT / "data" / "round-2" / "benchmark.json"
        )
        cls.cases_by_id = {case.id: case for case in cls.cases}
        cls.result_root = PROJECT_ROOT / "results" / "round-2" / "claude-fable-5"
        cls.summary = read_json(cls.result_root / "summary.json")
        with (cls.result_root / "objective_results.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_two_high_reasoning_closed_book_runs_are_valid(self) -> None:
        self.assertEqual(self.summary["model"], "claude-fable-5")
        self.assertEqual(self.summary["reasoning_effort"], "high")
        self.assertEqual(self.summary["authentication"], "Claude.ai pro subscription")
        self.assertEqual(self.summary["run_count"], 2)
        self.assertEqual(
            self.summary["benchmark_id"], self.artifact["benchmark_id"]
        )
        self.assertEqual(
            self.summary["protocol_id"], self.artifact["protocol_id"]
        )
        self.assertFalse(self.summary["blind_prompt"]["gold_answers_included"])
        self.assertFalse(self.summary["blind_prompt"]["external_tools_enabled"])
        for audit in self.summary["closed_book_audits"]:
            self.assertTrue(audit["closed_book_valid"])
            self.assertEqual(audit["reported_model"], "claude-fable-5")
            self.assertEqual(audit["external_tools_available"], [])
            self.assertEqual(audit["forbidden_tool_events"], 0)
            self.assertEqual(
                audit["usage"]["server_tool_use"],
                {"web_fetch_requests": 0, "web_search_requests": 0},
            )

    def test_saved_rows_and_metrics_match_frozen_key(self) -> None:
        self.assertEqual(len(self.rows), 40)
        self.assertEqual({row["id"] for row in self.rows}, set(self.cases_by_id))
        for row in self.rows:
            case = self.cases_by_id[row["id"]]
            first = row["run_1_answer"]
            second = row["run_2_answer"]
            self.assertEqual(row["run_1_correct"], str(first == case.correct_answer))
            self.assertEqual(row["run_2_correct"], str(second == case.correct_answer))
            self.assertEqual(row["consistent"], str(first == second))
            self.assertEqual(
                row["correct_and_consistent"],
                str(first == second == case.correct_answer),
            )
        objective = self.summary["objective"]
        self.assertEqual(objective["run_1_correct"], 38)
        self.assertEqual(objective["run_2_correct"], 37)
        self.assertEqual(objective["consistent_answers"], 39)
        self.assertEqual(objective["correct_and_consistent"], 37)


class Round2GeminiResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact, cls.cases = load_benchmark(
            PROJECT_ROOT / "data" / "round-2" / "benchmark.json"
        )
        cls.cases_by_id = {case.id: case for case in cls.cases}
        cls.result_root = PROJECT_ROOT / "results" / "round-2" / "gemini-3.6-flash"
        cls.summary = read_json(cls.result_root / "summary.json")
        with (cls.result_root / "objective_results.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_two_high_reasoning_closed_book_runs_are_valid(self) -> None:
        self.assertEqual(self.summary["model"], "gemini-3.6-flash-high")
        self.assertEqual(
            self.summary["model_display_name"], "Gemini 3.6 Flash (High)"
        )
        self.assertEqual(self.summary["reasoning_effort"], "high")
        self.assertEqual(
            self.summary["authentication"],
            "Google consumer OAuth via Antigravity",
        )
        self.assertEqual(self.summary["run_count"], 2)
        self.assertEqual(
            self.summary["benchmark_id"], self.artifact["benchmark_id"]
        )
        self.assertEqual(
            self.summary["protocol_id"], self.artifact["protocol_id"]
        )
        auth = self.summary["authentication_audit"]
        self.assertTrue(auth["consumer_oauth"])
        self.assertFalse(auth["enterprise_cloud_project"])
        self.assertFalse(auth["telemetry_enabled"])
        self.assertFalse(auth["g1_credit_fallback_enabled"])
        self.assertEqual(auth["verified_model"], "gemini-3.6-flash-high")
        self.assertFalse(self.summary["blind_prompt"]["gold_answers_included"])
        self.assertFalse(
            self.summary["blind_prompt"]["external_tool_execution_enabled"]
        )
        conversation_ids = set()
        for audit in self.summary["closed_book_audits"]:
            self.assertTrue(audit["closed_book_valid"])
            self.assertTrue(audit["tool_gate_enabled"])
            self.assertEqual(audit["reported_model"], "gemini-3.6-flash-high")
            self.assertEqual(audit["reasoning_effort"], "high")
            self.assertEqual(audit["external_tool_attempts"], [])
            self.assertEqual(audit["forbidden_tool_events"], 0)
            conversation_ids.add(audit["reported_conversation_id"])
        self.assertEqual(len(conversation_ids), 2)

    def test_saved_rows_and_metrics_match_frozen_key(self) -> None:
        self.assertEqual(len(self.rows), 40)
        self.assertEqual({row["id"] for row in self.rows}, set(self.cases_by_id))
        for row in self.rows:
            case = self.cases_by_id[row["id"]]
            first = row["run_1_answer"]
            second = row["run_2_answer"]
            self.assertIn(first, "ABCD")
            self.assertIn(second, "ABCD")
            self.assertEqual(row["run_1_correct"], str(first == case.correct_answer))
            self.assertEqual(row["run_2_correct"], str(second == case.correct_answer))
            self.assertEqual(row["consistent"], str(first == second))
            self.assertEqual(
                row["correct_and_consistent"],
                str(first == second == case.correct_answer),
            )
        objective = self.summary["objective"]
        self.assertEqual(objective["run_1_correct"], 37)
        self.assertEqual(objective["run_2_correct"], 35)
        self.assertEqual(objective["consistent_answers"], 38)
        self.assertEqual(objective["correct_and_consistent"], 35)


class Round2GrokManualResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact, cls.cases = load_benchmark(
            PROJECT_ROOT / "data" / "round-2" / "benchmark.json"
        )
        cls.cases_by_id = {case.id: case for case in cls.cases}
        cls.result_root = (
            PROJECT_ROOT / "results" / "round-2" / "grok-4.5-fast-manual"
        )
        cls.summary = read_json(cls.result_root / "summary.json")
        with (cls.result_root / "objective_results.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            cls.rows = list(csv.DictReader(handle))

    def test_manual_run_is_labeled_without_automated_tool_audit(self) -> None:
        self.assertEqual(self.summary["model"], "grok-4.5-fast")
        self.assertEqual(self.summary["model_display_name"], "Grok 4.5")
        self.assertEqual(self.summary["reasoning_effort"], "fast")
        self.assertEqual(self.summary["authentication"], "Grok Free consumer web")
        self.assertEqual(
            self.summary["collection_method"],
            "manual transcription from user-provided screenshot",
        )
        self.assertEqual(self.summary["run_count"], 2)
        self.assertEqual(
            self.summary["benchmark_id"], self.artifact["benchmark_id"]
        )
        self.assertEqual(
            self.summary["protocol_id"], self.artifact["protocol_id"]
        )
        self.assertFalse(self.summary["blind_prompt"]["gold_answers_included"])
        self.assertFalse(self.summary["audit"]["automated_tool_event_audit"])

    def test_saved_rows_and_metrics_match_frozen_key(self) -> None:
        self.assertEqual(len(self.rows), 40)
        self.assertEqual({row["id"] for row in self.rows}, set(self.cases_by_id))
        for row in self.rows:
            case = self.cases_by_id[row["id"]]
            first = row["run_1_answer"]
            second = row["run_2_answer"]
            self.assertIn(first, "ABCD")
            self.assertIn(second, "ABCD")
            self.assertEqual(row["run_1_correct"], str(first == case.correct_answer))
            self.assertEqual(row["run_2_correct"], str(second == case.correct_answer))
            self.assertEqual(row["consistent"], str(first == second))
            self.assertEqual(
                row["correct_and_consistent"],
                str(first == second == case.correct_answer),
            )
        objective = self.summary["objective"]
        self.assertEqual(objective["run_1_correct"], 33)
        self.assertEqual(objective["run_2_correct"], 30)
        self.assertEqual(objective["consistent_answers"], 33)
        self.assertEqual(objective["correct_and_consistent"], 29)

    def test_manual_transcription_matches_supplied_screenshot(self) -> None:
        self.assertEqual(
            "".join(row["run_1_answer"] for row in self.rows),
            "BBADBCAAABAACDBCDCABCBCADACCCBACADDABBDA",
        )
        self.assertEqual(
            "".join(row["run_2_answer"] for row in self.rows),
            "BBADACAAABAACDBCDCABCBCADACBDAACADDABCAD",
        )


if __name__ == "__main__":
    unittest.main()
