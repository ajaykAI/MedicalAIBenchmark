"""Model migration checks using synthetic responses, without provider calls."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import run_claude_benchmark as claude
from src import run_gemini_benchmark as gemini
from src import run_gpt_benchmark as gpt
from src import serve_benchmark as viewer
from src.io_utils import write_csv, write_json
from src.models import CURRENT_MODELS, GPT
from src.schemas import BenchmarkCase


class ModelMigrationTest(unittest.TestCase):
    def test_commands_and_output_folders_target_new_versions(self):
        commands = [
            (gpt, gpt.codex_command("codex", Path("schema.json")), "gpt-6-astra", "gpt-6-astra"),
            (claude, claude.claude_command("claude"), "claude-opus-5-5", "claude-opus-5-5"),
            (gemini, gemini.gemini_command("agy", "fixture", Path("/tmp/x"), 60, 1), "gemini-4-argon-high", "gemini-4-argon"),
        ]
        for runner, command, model_id, directory in commands:
            with self.subTest(model=model_id):
                self.assertEqual(command[command.index("--model") + 1], model_id)
                self.assertEqual(runner.parse_args([]).output_dir, Path("results") / directory)
                self.assertEqual(runner.REASONING_EFFORT, "high")

    def test_saved_old_versions_are_rejected_by_all_runners(self):
        for runner, old_id in (
            (gpt, "gpt-5.6-sol"),
            (claude, "claude-fable-5"),
            (gemini, "gemini-3.6-flash-high"),
        ):
            with self.subTest(model=old_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_json(root / "response.json", {
                    "model": old_id, "benchmark_id": "fixture", "protocol_id": "protocol",
                    "reasoning_effort": "high", "run_number": 1, "prompt_sha256": "prompt",
                })
                with self.assertRaisesRegex(RuntimeError, "metadata does not match"):
                    runner.load_saved_run(
                        root / "response.json", root / "events.jsonl", [],
                        benchmark_id="fixture", protocol_id="protocol",
                        prompt_sha256="prompt", run_number=1,
                    )


class CurrentResultsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.case = BenchmarkCase(
            id="medqa_001", source_id="synthetic", dataset="MedQA", split="test",
            category="fixture", question="Synthetic question?",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_answer="A",
        )
        self.rounds = {
            slug: {
                "slug": slug, "display_name": slug, "subtitle": "Synthetic round",
                "benchmark_path": self.root / slug / "benchmark.json",
                "results_root": self.root / slug,
                "response_matrix_path": None, "limitations": {},
                "default_score_view": "adjusted",
            }
            for slug in ("round-1", "round-2")
        }
        for patcher in (
            patch.object(viewer, "ROUND_SPECS", self.rounds),
            patch.object(viewer, "RESULT_SPECS", CURRENT_MODELS),
            patch.object(viewer, "load_benchmark", return_value=(
                {"benchmark_id": "fixture", "protocol_id": "protocol"}, [self.case],
            )),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def save_result(self, round_slug, **overrides):
        response = {"objective_answers": [{"id": self.case.id, "answer": "A", "confidence": 90}]}
        metrics, rows = claude.score_runs([self.case], [response, response])
        summary = {
            **metrics, "model": GPT["model"], "benchmark_id": "fixture",
            "protocol_id": "protocol", "reasoning_effort": "high", "run_count": 2,
            "closed_book_audits": [
                {"closed_book_valid": True, "forbidden_tool_events": 0},
                {"closed_book_valid": True, "forbidden_tool_events": 0},
            ],
            **overrides,
        }
        root = self.root / round_slug / GPT["directory"]
        write_json(root / "summary.json", summary)
        write_csv(root / "objective_results.csv", rows, gpt.OBJECTIVE_FIELDS)
        return root

    def test_no_results_means_pending_targets_without_scores(self):
        payload = viewer.load_results_payload("combined")
        self.assertEqual(payload["models"], [])
        self.assertEqual(len(payload["current_models"]), 4)
        for target in payload["current_models"]:
            self.assertEqual(target["status"], "pending")
            self.assertNotIn("audit_label", target)
            self.assertNotIn("raw", target)

    def test_combined_requires_both_rounds_and_promotes_completed_target(self):
        self.save_result("round-1")
        first = viewer.load_results_payload("round-1")
        self.assertEqual(first["default_model_slug"], GPT["slug"])
        self.assertEqual(first["current_models"][0]["status"], "completed")
        partial = viewer.load_results_payload("combined")
        self.assertEqual(partial["models"], [])
        self.assertEqual(partial["current_models"][0]["status"], "partial")
        self.save_result("round-2")
        combined = viewer.load_results_payload("combined")
        self.assertEqual(combined["default_model_slug"], GPT["slug"])
        self.assertEqual(combined["current_models"][0]["status"], "completed")
        self.assertEqual(combined["models"][0]["raw"]["objective"]["question_count"], 2)
        self.assertFalse(combined["models"][0]["historical"])

    def test_incomplete_artifact_stays_pending(self):
        root = self.root / "round-1" / GPT["directory"]
        write_json(root / "summary.json", {"model": GPT["model"]})
        payload = viewer.load_results_payload("round-1")
        self.assertEqual(payload["models"], [])
        self.assertEqual(payload["current_models"][0]["status"], "pending")

    def test_current_round_one_results_do_not_reuse_historical_matrix_answers(self):
        self.save_result("round-1")
        self.rounds["round-1"]["response_matrix_path"] = self.root / "matrix.csv"
        matrix = {"models": {}, "source_file": "matrix.csv", "sha256": "historical", "question_count": 1}
        with patch.object(viewer, "_load_response_matrix", return_value=matrix):
            payload = viewer.load_results_payload("round-1")
        self.assertEqual(payload["models"][0]["raw"]["objective"]["run_1_correct"], 1)
        self.assertNotEqual(payload["response_source"]["sha256"], "historical")

    def test_old_model_cannot_be_relabelled_as_current_result(self):
        self.save_result("round-1", model="gpt-5.6-sol")
        with self.assertRaisesRegex(ValueError, "model ID mismatch"):
            viewer.load_results_payload("round-1")

    def test_current_results_require_two_passes_and_valid_audits(self):
        for overrides in (
            {"run_count": 1}, {"closed_book_audits": []}, {"reasoning_effort": "low"},
        ):
            with self.subTest(overrides=overrides):
                self.save_result("round-1", **overrides)
                with self.assertRaises(ValueError):
                    viewer.load_results_payload("round-1")


if __name__ == "__main__":
    unittest.main()
