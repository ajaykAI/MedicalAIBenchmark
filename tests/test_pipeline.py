from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.benchmark import BenchmarkIntegrityError, load_benchmark
from src.build_benchmark import build
from src.io_utils import read_json, write_json


FIXTURES = Path(__file__).parent / "fixtures"
EVALUATION_RULE = {
    "id": "closed_book_no_external_research",
    "version": 1,
    "title": "Closed-book reasoning only",
    "requirement": "Each model must answer using only its own internal knowledge and reasoning.",
    "prohibited": [
        "Internet access",
        "Web search or browsing",
        "External retrieval, research, or reference tools",
    ],
    "enforcement": "Future runners must disable network and external research tools.",
}


class BenchmarkBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "data" / "raw").mkdir(parents=True)
        for name in (
            "medqa_test.jsonl",
            "medmcqa_validation.jsonl",
            "healthbench_oss_eval.jsonl",
        ):
            shutil.copyfile(FIXTURES / name, self.root / "data" / "raw" / name)
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            json.dumps(
                {
                    "benchmark": {
                        "schema_version": "1.0",
                        "random_seed": 42,
                        "counts": {"MedQA": 2, "MedMCQA": 1, "HealthBench": 2},
                        "evaluation_rules": [EVALUATION_RULE],
                    },
                    "datasets": {
                        "MedQA": {
                            "source": "fixture/medqa",
                            "upstream": "fixture",
                            "revision": "fixture-1",
                            "split": "test",
                            "file": "test.jsonl",
                            "license": "fixture",
                            "raw_file": "data/raw/medqa_test.jsonl",
                        },
                        "MedMCQA": {
                            "source": "fixture/medmcqa",
                            "upstream": "fixture",
                            "revision": "fixture-1",
                            "split": "validation",
                            "config": "default",
                            "license": "fixture",
                            "raw_file": "data/raw/medmcqa_validation.jsonl",
                            "split_note": "fixture",
                        },
                        "HealthBench": {
                            "source": "fixture/healthbench",
                            "upstream": "fixture",
                            "revision": "fixture-1",
                            "split": "test",
                            "file": "eval.jsonl",
                            "license": "fixture",
                            "raw_file": "data/raw/healthbench_oss_eval.jsonl",
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, *, force: bool = False):
        return build(
            self.config_path, root=self.root, offline=True, force=force
        )

    def test_build_freezes_expected_counts_and_exports(self) -> None:
        artifact = self._build()
        self.assertEqual(len(artifact["cases"]), 5)
        self.assertEqual(
            artifact["case_counts"],
            {"MedQA": 2, "MedMCQA": 1, "HealthBench": 2},
        )
        benchmark_path = self.root / "data" / "benchmark.json"
        csv_path = self.root / "data" / "benchmark.csv"
        manifest_path = self.root / "data" / "benchmark_manifest.json"
        self.assertTrue(benchmark_path.exists())
        self.assertTrue(csv_path.exists())
        self.assertTrue(manifest_path.exists())
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5)
        manifest = read_json(manifest_path)
        self.assertEqual(manifest["benchmark_id"], artifact["benchmark_id"])
        self.assertEqual(manifest["protocol_id"], artifact["protocol_id"])
        self.assertEqual(artifact["evaluation_rules"], [EVALUATION_RULE])

    def test_existing_benchmark_is_verified_without_resampling(self) -> None:
        first = self._build()
        first_bytes = (self.root / "data" / "benchmark.json").read_bytes()
        second = self._build()
        second_bytes = (self.root / "data" / "benchmark.json").read_bytes()
        self.assertEqual(first["benchmark_id"], second["benchmark_id"])
        self.assertEqual(first_bytes, second_bytes)

    def test_forced_rebuild_keeps_same_selection_and_identity(self) -> None:
        first = self._build()
        second = self._build(force=True)
        self.assertEqual(first["benchmark_id"], second["benchmark_id"])
        self.assertEqual(
            [case["source_id"] for case in first["cases"]],
            [case["source_id"] for case in second["cases"]],
        )

    def test_frozen_benchmark_detects_tampering(self) -> None:
        self._build()
        benchmark_path = self.root / "data" / "benchmark.json"
        artifact = read_json(benchmark_path)
        artifact["cases"][0]["question"] += " changed"
        write_json(benchmark_path, artifact)
        with self.assertRaises(BenchmarkIntegrityError):
            load_benchmark(benchmark_path)

    def test_rule_change_updates_protocol_without_resampling(self) -> None:
        first = self._build()
        config = read_json(self.config_path)
        config["benchmark"]["evaluation_rules"][0]["requirement"] += " No exceptions."
        write_json(self.config_path, config)
        second = self._build()
        self.assertEqual(first["benchmark_id"], second["benchmark_id"])
        self.assertNotEqual(first["protocol_id"], second["protocol_id"])
        self.assertEqual(
            [case["source_id"] for case in first["cases"]],
            [case["source_id"] for case in second["cases"]],
        )

    def test_protocol_rule_tampering_is_detected(self) -> None:
        self._build()
        benchmark_path = self.root / "data" / "benchmark.json"
        artifact = read_json(benchmark_path)
        artifact["evaluation_rules"][0]["requirement"] += " changed"
        write_json(benchmark_path, artifact)
        with self.assertRaises(BenchmarkIntegrityError):
            load_benchmark(benchmark_path)

    def test_normalized_schema_preserves_objective_gold_and_health_rubrics(self) -> None:
        self._build()
        _, cases = load_benchmark(self.root / "data" / "benchmark.json")
        objective = [case for case in cases if case.is_objective]
        health = [case for case in cases if not case.is_objective]
        self.assertTrue(all(set(case.choices or {}) == set("ABCD") for case in objective))
        self.assertTrue(all(case.correct_answer in set("ABCD") for case in objective))
        self.assertTrue(all(case.conversation for case in health))
        self.assertTrue(all(case.rubrics for case in health))


if __name__ == "__main__":
    unittest.main()
