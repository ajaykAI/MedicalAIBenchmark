from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.benchmark import load_benchmark
from src.build_benchmark import build
from src.build_objective_round import build_objective_round, requires_missing_visual
from src.schemas import BenchmarkCase


FIXTURES = Path(__file__).parent / "fixtures"
EVALUATION_RULE = {
    "id": "closed_book_no_external_research",
    "version": 1,
    "title": "Closed-book reasoning only",
    "requirement": "Use internal knowledge only.",
    "prohibited": ["Internet access"],
    "enforcement": "Disable external tools.",
}


class ObjectiveRoundBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        raw = self.root / "data" / "raw"
        raw.mkdir(parents=True)
        for name in (
            "medqa_test.jsonl",
            "medmcqa_validation.jsonl",
            "healthbench_oss_eval.jsonl",
        ):
            shutil.copyfile(FIXTURES / name, raw / name)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            json.dumps(
                {
                    "benchmark": {
                        "schema_version": "1.0",
                        "random_seed": 42,
                        "counts": {"MedQA": 1, "MedMCQA": 1, "HealthBench": 1},
                        "evaluation_rules": [EVALUATION_RULE],
                    },
                    "datasets": {
                        "MedQA": {
                            "source": "fixture/medqa",
                            "revision": "fixture-1",
                            "split": "test",
                            "raw_file": "data/raw/medqa_test.jsonl",
                        },
                        "MedMCQA": {
                            "source": "fixture/medmcqa",
                            "revision": "fixture-1",
                            "split": "validation",
                            "raw_file": "data/raw/medmcqa_validation.jsonl",
                        },
                        "HealthBench": {
                            "source": "fixture/healthbench",
                            "revision": "fixture-1",
                            "split": "test",
                            "raw_file": "data/raw/healthbench_oss_eval.jsonl",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        build(self.config_path, root=self.root, offline=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_round_excludes_prior_cases_and_has_only_requested_objective_counts(self):
        artifact = build_objective_round(
            self.config_path,
            root=self.root,
            counts={"MedQA": 1, "MedMCQA": 1},
            offline=True,
        )
        prior_artifact, prior_cases = load_benchmark(
            self.root / "data" / "benchmark.json"
        )
        _, round_cases = load_benchmark(
            self.root / "data" / "round-2" / "benchmark.json"
        )
        self.assertEqual(artifact["case_counts"], {"MedQA": 1, "MedMCQA": 1})
        self.assertEqual(artifact["prior_benchmark_id"], prior_artifact["benchmark_id"])
        self.assertTrue(all(case.is_objective for case in round_cases))
        prior_ids = {(case.dataset, case.source_id) for case in prior_cases}
        round_ids = {(case.dataset, case.source_id) for case in round_cases}
        self.assertFalse(prior_ids & round_ids)
        self.assertTrue(
            (self.root / "data" / "round-2" / "benchmark_manifest.json").exists()
        )

    def test_visual_reference_detection_catches_missing_arrow_item(self):
        case = BenchmarkCase(
            id="fixture",
            source_id="fixture",
            dataset="MedMCQA",
            split="validation",
            category="fixture",
            question="The structure marked with an arrow has which action?",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
        )
        self.assertTrue(requires_missing_visual(case))


if __name__ == "__main__":
    unittest.main()
