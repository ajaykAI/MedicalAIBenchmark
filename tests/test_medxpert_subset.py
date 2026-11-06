from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src import run_claude_benchmark as claude
from src import run_gpt_benchmark as gpt
from src.benchmark import load_benchmark
from src.build_medxpert_subset import (
    DEFAULT_RAW, DEFAULT_SELECTION, build, normalize_source, question_payload,
)
from src.io_utils import iter_jsonl, read_json, sha256_file, write_json
from src.schemas import SchemaError, answer_labels_for_cases
from src.select_medxpert_candidates import screen


def source_row():
    return {
        "id": "Text-1", "body_system": "Other / NA", "medical_task": "Diagnosis",
        "question_type": "Reasoning", "question": "Synthetic fixture?\nAnswer Choices: fixture choices",
        "options": {label: "Fixture option " + label for label in "ABCDEFGHIJ"}, "label": "J",
    }


class MedXpertSupportTest(unittest.TestCase):
    def test_ten_choices_are_dataset_specific(self):
        case = normalize_source(source_row(), "MXH-001")
        self.assertEqual(case.choices, source_row()["options"])
        self.assertEqual(case.correct_answer, "J")
        self.assertEqual(case.question, "Synthetic fixture?")
        self.assertEqual(answer_labels_for_cases([case]), list("ABCDEFGHIJ"))
        with self.assertRaises(SchemaError):
            replace(case, dataset="MedQA").validate()
        with self.assertRaises(SchemaError):
            replace(case, choices={k: k for k in "ABCD"}).validate()

    def test_both_runners_score_j_but_reject_j_for_four_choice_case(self):
        case = normalize_source(source_row(), "MXH-001")
        response = {"objective_answers": [{"id": case.id, "answer": "J", "confidence": 90}],
                    "healthbench_responses": []}
        for runner, error in [(gpt, gpt.RunnerError), (claude, claude.ClaudeRunnerError)]:
            runner.validate_response(response, [case])
            summary, _ = runner.score_runs([case], [response, response])
            self.assertEqual(summary["objective"]["mean_accuracy"], 1.0)
            four = replace(case, dataset="MedQA", choices={k: k for k in "ABCD"}, correct_answer="A")
            with self.assertRaises(error):
                runner.validate_response(response, [four])
            schema = runner.response_schema(answer_labels_for_cases([case]))
            self.assertEqual(schema["properties"]["objective_answers"]["items"]["properties"]["answer"]["enum"], list("ABCDEFGHIJ"))

    def test_blind_inputs_exclude_keys_and_selection_notes(self):
        case = replace(normalize_source(source_row(), "MXH-001"),
                       source_metadata={"selection_reason": "PRIVATE_SELECTION_RATIONALE"})
        for runner in (gpt, claude):
            payload = runner.build_blind_payload([case])
            prompt = runner.build_prompt(payload)
            self.assertIn("A–J", prompt)
            self.assertNotIn("A, B, C, or D", prompt)
            for secret in ("correct_answer", "source_metadata", "Text-1", "PRIVATE_SELECTION_RATIONALE"):
                self.assertNotIn(secret, prompt)
        self.assertEqual(set(question_payload([case])["questions"][0]), {"id", "question", "choices"})

    def test_live_call_schema_is_built_from_case_choices_without_calling_a_model(self):
        case = normalize_source(source_row(), "MXH-001")
        def inspect_gpt(command, **kwargs):
            schema = read_json(Path(command[command.index("--output-schema") + 1]))
            self.assertIn("J", schema["properties"]["objective_answers"]["items"]["properties"]["answer"]["enum"])
            raise RuntimeError("stop-before-model-call")
        def inspect_claude(command, **kwargs):
            schema = json.loads(command[command.index("--json-schema") + 1])
            self.assertIn("J", schema["properties"]["objective_answers"]["items"]["properties"]["answer"]["enum"])
            raise RuntimeError("stop-before-model-call")
        for runner, inspect in [(gpt, inspect_gpt), (claude, inspect_claude)]:
            with patch.object(runner.subprocess, "run", side_effect=inspect):
                with self.assertRaisesRegex(RuntimeError, "stop-before-model-call"):
                    runner.run_once("unused-cli", "synthetic prompt", [case], Path("unused-events"), timeout_seconds=1)

    def test_screening_cannot_depend_on_gold_label(self):
        first = source_row()
        second = dict(first, label="A")
        self.assertEqual(screen([first]), screen([second]))


@unittest.skipUnless(DEFAULT_RAW.exists(), "Pinned source data intentionally stays local")
class MedXpertFrozenSelectionTest(unittest.TestCase):
    def test_rebuild_fidelity_blindness_and_freeze_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "subset"
            artifact = build(output=output)
            _, cases = load_benchmark(output / "benchmark.json")
            self.assertEqual(len(cases), 100)
            self.assertEqual(len({c.source_id for c in cases}), 100)
            source = {r["id"]: r for r in iter_jsonl(DEFAULT_RAW)}
            exported = list(iter_jsonl(output / "selected_source.jsonl"))
            for case, original in zip(cases, exported):
                self.assertEqual(original, source[case.source_id])
                self.assertEqual(case.question, original["question"].split("Answer Choices:", 1)[0].rstrip())
                self.assertEqual(list(case.choices.items()), list(original["options"].items()))
                self.assertEqual(case.correct_answer, original["label"])
                self.assertEqual(original["question_type"], "Reasoning")
            manifest = read_json(output / "manifest.json")
            for name, digest in manifest["files"].items():
                self.assertEqual(sha256_file(output / name), digest)
            old_manifest = (output / "manifest.json").read_bytes()
            self.assertEqual(build(output=output)["benchmark_id"], artifact["benchmark_id"])
            self.assertEqual((output / "manifest.json").read_bytes(), old_manifest)
            serialized = (output / "questions.json").read_text()
            for forbidden in ("correct_answer", "source_id", "reason", "Text-"):
                # Do not search clinical prose for the English word 'reason'.
                self.assertNotIn('"' + forbidden + '"', serialized)
            altered = read_json(DEFAULT_SELECTION)
            altered["presentation_seed"] += 1
            altered_path = Path(directory) / "altered-selection.json"
            write_json(altered_path, altered)
            with self.assertRaisesRegex(ValueError, "Existing frozen benchmark differs"):
                build(selection_path=altered_path, output=output)
            bad_raw = Path(directory) / "wrong-source.jsonl"
            bad_raw.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "Source checksum mismatch"):
                build(raw=bad_raw, output=output)


if __name__ == "__main__":
    unittest.main()
