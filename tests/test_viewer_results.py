from __future__ import annotations

import csv
import hashlib
import unittest

from src.config import PROJECT_ROOT
from src.serve_benchmark import (
    load_benchmark_payload,
    load_results_payload,
    load_rounds_payload,
)


class CombinedViewerResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_results_payload("combined")
        cls.models = {model["slug"]: model for model in cls.payload["models"]}

    def test_combined_is_default_and_contains_all_80_objective_questions(self):
        rounds = load_rounds_payload()
        self.assertEqual(rounds["default_round_slug"], "combined")
        self.assertEqual(
            [item["slug"] for item in rounds["rounds"]],
            ["combined", "round-1", "round-2"],
        )
        benchmark = load_benchmark_payload("combined")
        self.assertEqual(len(benchmark["cases"]), 80)
        self.assertEqual(
            benchmark["case_counts"],
            {"MedQA": 60, "MedMCQA": 20, "HealthBench": 0},
        )
        self.assertEqual(len({case["id"] for case in benchmark["cases"]}), 80)
        self.assertEqual(benchmark["cases"][0]["source_round"], "round-1")
        self.assertEqual(benchmark["cases"][40]["source_round"], "round-2")

    def test_combined_adjusted_view_excludes_three_documented_items(self):
        self.assertEqual(
            [
                (item["qid"], item["source_round"], item["source_qid"])
                for item in self.payload["known_limitations"]
            ],
            [
                (35, "round-1", 35),
                (74, "round-2", 34),
                (78, "round-2", 38),
            ],
        )
        for model in self.models.values():
            self.assertEqual(model["raw"]["objective"]["question_count"], 80)
            self.assertEqual(model["adjusted"]["objective"]["question_count"], 77)
            self.assertEqual(len(model["objective_results"]), 80)

    def test_combined_raw_and_adjusted_metrics(self):
        expected_raw = {
            "gpt-5.6-sol": (75, 76, 77, 74),
            "claude-fable-5": (76, 75, 79, 75),
            "gemini-3.6-flash": (74, 72, 77, 72),
            "grok-4.5-fast": (70, 67, 68, 64),
        }
        expected_adjusted = {
            "gpt-5.6-sol": (74, 75, 74, 73),
            "claude-fable-5": (76, 75, 76, 75),
            "gemini-3.6-flash": (74, 72, 75, 72),
            "grok-4.5-fast": (70, 67, 67, 64),
        }
        for slug, model in self.models.items():
            for view, expected in (
                ("raw", expected_raw[slug]),
                ("adjusted", expected_adjusted[slug]),
            ):
                score = model[view]["objective"]
                self.assertEqual(
                    (
                        score["run_1_correct"],
                        score["run_2_correct"],
                        score["consistent_answers"],
                        score["correct_and_consistent"],
                    ),
                    expected,
                )



class Round1ViewerResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_results_payload("round-1")
        cls.models = {model["slug"]: model for model in cls.payload["models"]}

    def test_all_four_models_are_available(self):
        self.assertEqual(
            set(self.models),
            {
                "gpt-5.6-sol",
                "claude-fable-5",
                "gemini-3.6-flash",
                "grok-4.5-fast",
            },
        )
        self.assertEqual(self.payload["default_model_slug"], "grok-4.5-fast")

    def test_imported_response_matrix_drives_all_model_answers(self):
        matrix_path = PROJECT_ROOT / "results" / "medical_ai_benchmark_responses.csv"
        with matrix_path.open(encoding="utf-8-sig", newline="") as handle:
            matrix_rows = [
                row for row in csv.DictReader(handle) if row["qID"].isdigit()
            ]
        self.assertEqual(len(matrix_rows), 40)
        self.assertEqual(
            self.payload["response_source"],
            {
                "kind": "response matrix",
                "source_file": "results/medical_ai_benchmark_responses.csv",
                "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                "question_count": 40,
            },
        )
        columns = {
            "gpt-5.6-sol": ("GPT Pass 1", "GPT Pass 2"),
            "claude-fable-5": ("Fable Pass 1", "Fable Pass 2"),
            "gemini-3.6-flash": ("Gemini Pass 1", "Gemini Pass 2"),
            "grok-4.5-fast": ("Grok Pass 1", "Grok Pass 2"),
        }
        for slug, (first_column, second_column) in columns.items():
            result_rows = self.models[slug]["objective_results"]
            self.assertEqual(len(result_rows), 40)
            for matrix_row, result_row in zip(matrix_rows, result_rows):
                self.assertEqual(result_row["run_1_answer"], matrix_row[first_column])
                self.assertEqual(result_row["run_2_answer"], matrix_row[second_column])

    def test_plotly_bar_charts_are_local_and_wired(self):
        html = (PROJECT_ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        script = (PROJECT_ROOT / "viewer" / "app.js").read_text(encoding="utf-8")
        styles = (PROJECT_ROOT / "viewer" / "styles.css").read_text(encoding="utf-8")
        plotly = PROJECT_ROOT / "viewer" / "vendor" / "plotly-basic-3.7.0.min.js"
        self.assertIn('/vendor/plotly-basic-3.7.0.min.js', html)
        self.assertIn('id="accuracy-chart"', html)
        self.assertIn('id="repeatability-chart"', html)
        self.assertNotIn('<th scope="col">Mode</th>', html)
        self.assertNotIn('id="audit-status"', html)
        self.assertNotIn('class="audit-line"', html)
        self.assertIn("window.Plotly.react", script)
        self.assertIn('method.textContent = "Closed-book run"', script)
        self.assertNotIn('"Manual web entry"', script)
        self.assertNotIn('"Automated run"', script)
        self.assertNotIn('"Original sample"', script)
        self.assertNotIn('"New sample"', script)
        self.assertIn("x: categories", script)
        self.assertIn('range: [lowerBound, 102.5]', script)
        self.assertIn('"#00d5ff"', script)
        self.assertIn('"#ff4fd8"', script)
        self.assertIn("color-scheme: dark", styles)
        self.assertIn("--canvas: #171a1f", styles)
        self.assertIn("--paper: #242830", styles)
        self.assertIn('plot_bgcolor: "#1d2127"', script)
        self.assertGreater(plotly.stat().st_size, 1_000_000)
        self.assertIn("plotly.js (basic - minified) v3.7.0", plotly.read_text()[:200])

    def test_questions_live_only_on_dedicated_bounded_question_bank_page(self):
        results_html = (PROJECT_ROOT / "viewer" / "index.html").read_text(
            encoding="utf-8"
        )
        bank_html = (PROJECT_ROOT / "viewer" / "question-bank.html").read_text(
            encoding="utf-8"
        )
        bank_script = (PROJECT_ROOT / "viewer" / "question-bank.js").read_text(
            encoding="utf-8"
        )
        styles = (PROJECT_ROOT / "viewer" / "styles.css").read_text(
            encoding="utf-8"
        )
        self.assertIn('href="/questions"', results_html)
        self.assertNotIn('id="question-list"', results_html)
        self.assertNotIn('id="objective-template"', results_html)
        self.assertIn('id="question-list"', bank_html)
        self.assertIn('id="question-viewport"', bank_html)
        self.assertIn('id="model-filter"', bank_html)
        self.assertIn('src="/question-bank.js"', bank_html)
        self.assertIn('fetchJson("/api/rounds"', bank_script)
        self.assertIn("overflow-y: auto", styles)
        self.assertIn("height: clamp(430px, 62vh, 760px)", styles)

    def test_grok_manual_results_match_transcription(self):
        grok = self.models["grok-4.5-fast"]
        self.assertEqual(grok["display_name"], "Grok 4.5")
        self.assertEqual(grok["collection_method"], "manual")
        self.assertEqual(grok["reasoning_label"], "Fast mode")
        self.assertEqual(grok["raw"]["objective"]["run_1_correct"], 37)
        self.assertEqual(grok["raw"]["objective"]["run_2_correct"], 37)
        self.assertEqual(grok["raw"]["objective"]["consistent_answers"], 35)
        self.assertAlmostEqual(
            grok["raw"]["objective"]["answer_consistency"], 0.875
        )
        self.assertIsNone(grok["objective_results"][0]["run_1_confidence"])

    def test_adjusted_view_excludes_missing_image_item_for_every_model(self):
        for model in self.models.values():
            self.assertEqual(model["raw"]["objective"]["question_count"], 40)
            self.assertEqual(model["adjusted"]["objective"]["question_count"], 39)
        limitation = self.payload["known_limitations"][0]
        self.assertEqual(limitation["id"], "medmcqa_005")
        self.assertEqual(limitation["qid"], 35)

    def test_grok_inconsistent_question_ids(self):
        grok = self.models["grok-4.5-fast"]
        changed = [
            row["id"]
            for row in grok["objective_results"]
            if not row["consistent"]
        ]
        self.assertEqual(
            changed,
            [
                "medqa_001",
                "medqa_027",
                "medmcqa_002",
                "medmcqa_005",
                "medmcqa_007",
            ],
        )


class Round2ViewerResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_results_payload("round-2")
        cls.models = {model["slug"]: model for model in cls.payload["models"]}

    def test_round_selector_includes_round2_adjusted_reporting(self):
        rounds = load_rounds_payload()
        self.assertEqual(rounds["default_round_slug"], "combined")
        self.assertEqual(
            {item["slug"] for item in rounds["rounds"]},
            {"combined", "round-1", "round-2"},
        )
        self.assertEqual(self.payload["round"]["slug"], "round-2")
        self.assertEqual(self.payload["default_score_view"], "adjusted")
        self.assertEqual(self.payload["response_source"]["kind"], "per-model result archives")

    def test_adjusted_view_excludes_qids_34_and_38_for_every_model(self):
        self.assertEqual(
            [(item["qid"], item["id"]) for item in self.payload["known_limitations"]],
            [(34, "medmcqa_004"), (38, "medmcqa_008")],
        )
        for model in self.models.values():
            self.assertEqual(model["raw"]["objective"]["question_count"], 40)
            self.assertEqual(model["adjusted"]["objective"]["question_count"], 38)
            excluded = [
                index
                for index, row in enumerate(model["objective_results"], start=1)
                if row["excluded_from_adjusted"]
            ]
            self.assertEqual(excluded, [34, 38])

    def test_round2_adjusted_metrics_match_uniform_38_question_rescore(self):
        expected = {
            "gpt-5.6-sol": (36, 37, 37, 36),
            "claude-fable-5": (38, 37, 37, 37),
            "gemini-3.6-flash": (37, 35, 36, 35),
            "grok-4.5-fast": (33, 30, 32, 29),
        }
        for slug, values in expected.items():
            score = self.models[slug]["adjusted"]["objective"]
            self.assertEqual(
                (
                    score["run_1_correct"],
                    score["run_2_correct"],
                    score["consistent_answers"],
                    score["correct_and_consistent"],
                ),
                values,
            )

    def test_frontend_exposes_round_controls_and_adjusted_source_keys(self):
        html = (PROJECT_ROOT / "viewer" / "index.html").read_text(encoding="utf-8")
        script = (PROJECT_ROOT / "viewer" / "app.js").read_text(encoding="utf-8")
        bank_script = (PROJECT_ROOT / "viewer" / "question-bank.js").read_text(
            encoding="utf-8"
        )
        styles = (PROJECT_ROOT / "viewer" / "styles.css").read_text(encoding="utf-8")
        self.assertIn('id="round-selector"', html)
        self.assertNotIn("LOCAL · READ ONLY", html)
        self.assertNotIn('id="benchmark-id"', html)
        self.assertIn('fetchJson("/api/rounds"', script)
        self.assertIn('state.scoreView = resultsPayload.default_score_view', script)
        self.assertIn("Source key:", bank_script)
        self.assertIn("choice--source-key", styles)


if __name__ == "__main__":
    unittest.main()
