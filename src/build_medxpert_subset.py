"""Freeze the reviewed MedXpertQA selection without sampling or model calls."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src.benchmark import compute_benchmark_id, compute_protocol_id, load_benchmark
from src.io_utils import (
    atomic_write_bytes, canonical_json_bytes, iter_jsonl, read_json,
    sha256_bytes, sha256_file, write_json, write_jsonl,
)
from src.schemas import BenchmarkCase, SCHEMA_VERSION, validate_benchmark_cases
from src.select_medxpert_candidates import SCREENING_VERSION, screen, shortlist

DEFAULT_RAW = Path("data/raw/medxpertqa_text_test_7e7c465a.jsonl")
DEFAULT_SELECTION = Path("selections/medxpertqa-hard100-v1.json")
DEFAULT_OUTPUT = Path("data/medxpert-hard100")

RULES = [{
    "id": "closed_book_no_external_research", "version": 2,
    "title": "Closed-book reasoning only",
    "requirement": "Use internal knowledge and reasoning with the supplied question and choices only.",
    "prohibited": ["Web search or browsing", "External retrieval or reference tools",
                   "Answer keys, selection rationales, prior responses, and conversation memory",
                   "Shell, filesystem, browser, plugin, or other agent tools"],
    "enforcement": "Use a fresh isolated session with tools and memory disabled; audit the event log. "
                   "Network transport to the hosted model is permitted, external information lookup is not.",
}]


def normalize_source(row: Dict[str, Any], case_id: str) -> BenchmarkCase:
    question, separator, _ = row["question"].partition("Answer Choices:")
    if not separator:
        raise ValueError("Missing original answer-choice delimiter: %s" % row["id"])
    case = BenchmarkCase(
        id=case_id, source_id=row["id"], dataset="MedXpertQA", split="test",
        category=row["medical_task"], question=question.rstrip(),
        choices=dict(row["options"]), correct_answer=row["label"],
        source_metadata={
            "body_system": row["body_system"], "medical_task": row["medical_task"],
            "question_type": row["question_type"],
            "source_row_sha256": sha256_bytes(canonical_json_bytes(row)),
            "normalization": "Split the existing Answer Choices delimiter; trim trailing stem whitespace only. "
                             "Original options, option order, and label are unchanged.",
        },
    )
    case.validate()
    return case


def question_payload(cases: List[BenchmarkCase]) -> Dict[str, Any]:
    # Explicit allowlist: no source IDs, keys, categories, or curation notes.
    return {"questions": [{"id": c.id, "question": c.question, "choices": c.choices} for c in cases]}


def build(raw: Path = DEFAULT_RAW, selection_path: Path = DEFAULT_SELECTION,
          output: Path = DEFAULT_OUTPUT) -> Dict[str, Any]:
    selection = read_json(selection_path)
    source = selection["source"]
    if sha256_file(raw) != source["sha256"]:
        raise ValueError("Source checksum mismatch; refusing to change the frozen questions")
    rows = list(iter_jsonl(raw))
    by_id = {row["id"]: row for row in rows}
    if len(rows) != source["row_count"] or len(by_id) != len(rows):
        raise ValueError("Wrong source row count or duplicate source IDs")
    reviews = selection["review_records"]
    method = selection["method"]
    if method["screening_version"] != SCREENING_VERSION:
        raise ValueError("Screening implementation does not match the selection")
    screening = screen(rows)
    expected_pool = shortlist(screening, method["shortlist_per_body_system"])
    if (len(reviews) != method["content_reviewed_rows"]
            or len({r["source_id"] for r in reviews}) != len(reviews)
            or set(expected_pool) != {r["source_id"] for r in reviews}):
        raise ValueError("Review records do not match the reproducible screening pool")
    chosen = [r for r in reviews if r["decision"] == "selected"]
    if len(chosen) != 100 or selection["question_count"] != 100:
        raise ValueError("The frozen selection must contain exactly 100 questions")
    if any(not r.get("reason", "").strip() for r in chosen):
        raise ValueError("Every selected question requires a curation rationale")
    chosen_ids = sorted(r["source_id"] for r in chosen)
    random.Random(selection["presentation_seed"]).shuffle(chosen_ids)
    selected_rows = [by_id[k] for k in chosen_ids]
    cases = [normalize_source(row, "MXH-%03d" % (i + 1)) for i, row in enumerate(selected_rows)]
    validate_benchmark_cases(cases)
    selection_hash = sha256_file(selection_path)
    artifact = {
        "schema_version": SCHEMA_VERSION, "random_seed": selection["presentation_seed"],
        "round_id": selection["selection_id"], "title": selection["display_name"],
        "sources": {"MedXpertQA": dict(source, selection_sha256=selection_hash,
                                        selection_id=selection["selection_id"])},
        "case_counts": {"MedXpertQA": 100}, "cases": [c.to_dict() for c in cases],
        "evaluation_rules": RULES,
    }
    artifact["benchmark_id"] = compute_benchmark_id(artifact)
    artifact["protocol_id"] = compute_protocol_id(artifact)
    artifact_path = output / "benchmark.json"
    if artifact_path.exists():
        previous, _ = load_benchmark(artifact_path)
        if (previous["benchmark_id"], previous["protocol_id"]) != (artifact["benchmark_id"], artifact["protocol_id"]):
            raise ValueError("Existing frozen benchmark differs; use a new version and output directory")
    write_json(artifact_path, artifact)
    write_jsonl(output / "selected_source.jsonl", selected_rows)
    write_json(output / "questions.json", question_payload(cases))
    write_json(output / "answer_key.json", {
        "benchmark_id": artifact["benchmark_id"],
        "answers": [{"id": c.id, "source_id": c.source_id, "answer": c.correct_answer} for c in cases],
    })
    packet = ["# Medical benchmark — 100 multiple-choice questions\n",
              "Choose exactly one label A–J for each question using your own internal knowledge and reasoning. "
              "Do not browse, search, consult references, use tools, or access answer keys.\n"]
    for c in cases:
        packet.extend(["## %s\n" % c.id, c.question + "\n"])
        packet.extend("**%s.** %s\n" % (label, value) for label, value in c.choices.items())
    atomic_write_bytes(output / "questions.md", ("\n".join(packet) + "\n").encode("utf-8"))
    counts = {
        "body_system": dict(sorted(Counter(r["body_system"] for r in selected_rows).items())),
        "medical_task": dict(sorted(Counter(r["medical_task"] for r in selected_rows).items())),
    }
    report = ["# MedXpertQA Text — Curated Hard 100\n",
              "100 original ten-option MCQs selected from the pinned 2,450-question Text test split.\n",
              "This is an AI-curated hard subset, not an empirically established hardest-100 ranking. "
              "All 2,450 rows were screened; 288 shortlisted questions received AI content review. "
              "No target-model responses or answer keys were used to choose items. "
              "The source answers were subsequently copied unchanged for scoring.\n",
              "Screening prioritizes evidence integration, changing clinical course, treatment constraints, "
              "and diagnostic decisions. Its small capped length contribution is not a difficulty rating. "
              "The top 24 eligible candidates per source body-system label were reviewed. "
              "Final selection used clinical reasoning demands and apparent item quality, without final quotas. "
              "Presentation order is shuffled with a fixed seed and is not a difficulty ranking.\n",
              "The original keys have not received a new clinician review. The source’s published review "
              "does not establish that every item is unambiguous or current with 2026 guidance. "
              "Do not treat this subset as a measure of patient-facing safety or real-world outcomes.\n",
              "Source: https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA\n",
              "Source revision: `%s`.\n" % source["revision"],
              "Benchmark ID: `%s`.\n" % artifact["benchmark_id"],
              "Selection SHA-256: `%s`.\n" % selection_hash,
              "## Coverage\n", "| Source body-system label | Questions |\n| --- | ---: |"]
    report.extend("| %s | %d |" % pair for pair in counts["body_system"].items())
    report.append("\n| Source task label | Questions |\n| --- | ---: |")
    report.extend("| %s | %d |" % pair for pair in counts["medical_task"].items())
    report.extend(["\n## Selection rationale — keep separate from model input\n",
                   "| Packet ID | Source ID | Reason retained |\n| --- | --- | --- |"])
    reasons = {r["source_id"]: r["reason"] for r in chosen}
    report.extend("| %s | %s | %s |" % (c.id, c.source_id, reasons[c.source_id]) for c in cases)
    report.append("\n## Limitations\n")
    report.extend("- " + item for item in method["limitations"])
    atomic_write_bytes(output / "selection_report.md", ("\n".join(report) + "\n").encode("utf-8"))
    write_json(output / "screening.json", screening)
    files = ["benchmark.json", "selected_source.jsonl", "questions.json", "questions.md",
             "answer_key.json", "selection_report.md", "screening.json"]
    write_json(output / "manifest.json", {
        "selection_id": selection["selection_id"], "selection_sha256": selection_hash,
        "benchmark_id": artifact["benchmark_id"], "protocol_id": artifact["protocol_id"],
        "source": source, "counts": counts, "method": method,
        "files": {name: sha256_file(output / name) for name in files},
    })
    load_benchmark(artifact_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = build(args.raw, args.selection, args.output)
    print("Frozen 100 MedXpertQA questions: %s" % artifact["benchmark_id"])


if __name__ == "__main__":
    main()
