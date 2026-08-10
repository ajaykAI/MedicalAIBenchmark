"""Download, normalize, sample once, and freeze the 60-case benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from src.benchmark import (
    compute_benchmark_id,
    compute_protocol_id,
    load_benchmark,
    validate_evaluation_rules,
)
from src.config import PROJECT_ROOT, load_config, project_path
from src.datasets import healthbench, medmcqa, medqa
from src.datasets.common import dataset_seed
from src.io_utils import sha256_file, write_csv, write_json
from src.schemas import BenchmarkCase, validate_benchmark_cases


LOGGER = logging.getLogger("medical_benchmark.build")
DATASET_MODULES = {
    "MedQA": medqa,
    "MedMCQA": medmcqa,
    "HealthBench": healthbench,
}
ID_PREFIXES = {
    "MedQA": "medqa",
    "MedMCQA": "medmcqa",
    "HealthBench": "healthbench",
}
CSV_FIELDS = [
    "id",
    "source_id",
    "dataset",
    "split",
    "category",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "correct_answer",
    "conversation_json",
    "rubrics_json",
    "source_metadata_json",
]


def _normalized_question(case: BenchmarkCase) -> str:
    return " ".join(case.question.casefold().split())


def _select_without_duplicates(
    candidates: Sequence[BenchmarkCase],
    count: int,
    *,
    seed: int,
    dataset: str,
    seen_questions: Set[str],
) -> List[BenchmarkCase]:
    unique = {}
    for case in candidates:
        key = _normalized_question(case)
        if key and key not in unique:
            unique[key] = case
    ordered = list(unique.values())
    random.Random(dataset_seed(seed, dataset)).shuffle(ordered)
    selected = []
    for case in ordered:
        key = _normalized_question(case)
        if key in seen_questions:
            continue
        selected.append(case)
        seen_questions.add(key)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            "%s has only %d usable unique cases after cross-dataset deduplication; %d requested"
            % (dataset, len(selected), count)
        )
    return selected


def _assign_ids(cases: Iterable[BenchmarkCase]) -> List[BenchmarkCase]:
    grouped: Dict[str, List[BenchmarkCase]] = {}
    for case in cases:
        grouped.setdefault(case.dataset, []).append(case)
    assigned = []
    for dataset in ("MedQA", "MedMCQA", "HealthBench"):
        selected = sorted(grouped.get(dataset, []), key=lambda item: item.source_id)
        for index, case in enumerate(selected, start=1):
            assigned.append(
                replace(case, id="%s_%03d" % (ID_PREFIXES[dataset], index))
            )
    return assigned


def _csv_row(case: BenchmarkCase) -> Dict[str, Any]:
    choices = case.choices or {}
    return {
        "id": case.id,
        "source_id": case.source_id,
        "dataset": case.dataset,
        "split": case.split,
        "category": case.category,
        "question": case.question,
        "choice_A": choices.get("A", ""),
        "choice_B": choices.get("B", ""),
        "choice_C": choices.get("C", ""),
        "choice_D": choices.get("D", ""),
        "correct_answer": case.correct_answer or "",
        "conversation_json": json.dumps(case.conversation or [], ensure_ascii=False),
        "rubrics_json": json.dumps(
            [
                {"criterion": item.criterion, "points": item.points, "tags": item.tags}
                for item in (case.rubrics or [])
            ],
            ensure_ascii=False,
        ),
        "source_metadata_json": json.dumps(case.source_metadata, ensure_ascii=False),
    }


def _write_manifest(
    manifest_path: Path,
    benchmark_path: Path,
    csv_path: Path,
    artifact: Dict[str, Any],
) -> None:
    manifest = {
        "schema_version": artifact["schema_version"],
        "benchmark_id": artifact["benchmark_id"],
        "protocol_id": artifact["protocol_id"],
        "evaluation_rules": artifact["evaluation_rules"],
        "benchmark_file_sha256": sha256_file(benchmark_path),
        "benchmark_csv_sha256": sha256_file(csv_path),
        "random_seed": artifact["random_seed"],
        "case_counts": artifact["case_counts"],
        "source_revisions": {
            item["dataset"]: item["revision"] for item in artifact["sources"]
        },
    }
    write_json(manifest_path, manifest)


def build(
    config_path: Path,
    *,
    root: Path = PROJECT_ROOT,
    offline: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    config = load_config(config_path)
    benchmark_path = Path(root) / "data" / "benchmark.json"
    csv_path = Path(root) / "data" / "benchmark.csv"
    manifest_path = Path(root) / "data" / "benchmark_manifest.json"
    evaluation_rules = config["benchmark"].get("evaluation_rules")
    validate_evaluation_rules(evaluation_rules)
    if benchmark_path.exists() and not force:
        artifact, cases = load_benchmark(benchmark_path)
        expected_protocol_id = compute_protocol_id(
            {
                "benchmark_id": artifact["benchmark_id"],
                "evaluation_rules": evaluation_rules,
            }
        )
        if (
            artifact.get("evaluation_rules") != evaluation_rules
            or artifact.get("protocol_id") != expected_protocol_id
        ):
            artifact["evaluation_rules"] = evaluation_rules
            artifact["protocol_id"] = expected_protocol_id
            write_json(benchmark_path, artifact)
            _write_manifest(
                manifest_path, benchmark_path, csv_path, artifact
            )
            LOGGER.info(
                "Updated benchmark protocol rules without resampling: %s",
                artifact["protocol_id"],
            )
        LOGGER.info(
            "Frozen benchmark already exists and passed integrity checks: %s (%d cases, protocol %s)",
            artifact["benchmark_id"],
            len(cases),
            artifact["protocol_id"],
        )
        return artifact

    seed = int(config["benchmark"]["random_seed"])
    requested_counts = config["benchmark"]["counts"]
    all_selected: List[BenchmarkCase] = []
    seen_questions: Set[str] = set()
    sources = []

    for dataset in ("MedQA", "MedMCQA", "HealthBench"):
        dataset_config = dict(config["datasets"][dataset])
        raw_path = project_path(dataset_config["raw_file"], root)
        revision_prefix = str(dataset_config["revision"])[:8]
        if not offline and revision_prefix not in raw_path.name:
            raise ValueError(
                "%s raw_file must contain revision prefix %s to prevent stale-cache provenance"
                % (dataset, revision_prefix)
            )
        if not raw_path.exists():
            if offline:
                raise FileNotFoundError(
                    "Offline mode requires the raw file: %s" % raw_path
                )
            LOGGER.info("Acquiring %s %s split", dataset, dataset_config["split"])
            DATASET_MODULES[dataset].acquire(dataset_config, raw_path)
        LOGGER.info("Normalizing %s from %s", dataset, raw_path)
        candidates = DATASET_MODULES[dataset].load(raw_path, dataset_config["split"])
        requested = int(requested_counts[dataset])
        selected = _select_without_duplicates(
            candidates,
            requested,
            seed=seed,
            dataset=dataset,
            seen_questions=seen_questions,
        )
        all_selected.extend(selected)
        source_record = {
            "dataset": dataset,
            "source": dataset_config["source"],
            "upstream": dataset_config.get("upstream"),
            "revision": dataset_config["revision"],
            "split": dataset_config["split"],
            "license": dataset_config.get("license"),
            "raw_sha256": sha256_file(raw_path),
            "raw_rows_usable": len(candidates),
        }
        if dataset_config.get("split_note"):
            source_record["split_note"] = dataset_config["split_note"]
        sources.append(source_record)

    cases = _assign_ids(all_selected)
    validate_benchmark_cases(cases)
    actual_counts = dict(Counter(case.dataset for case in cases))
    expected_total = sum(int(value) for value in requested_counts.values())
    if len(cases) != expected_total or actual_counts != {
        key: int(value) for key, value in requested_counts.items()
    }:
        raise RuntimeError("Selected benchmark counts do not match config")

    artifact: Dict[str, Any] = {
        "schema_version": config["benchmark"]["schema_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "case_counts": actual_counts,
        "sources": sources,
        "cases": [case.to_dict() for case in cases],
        "evaluation_rules": evaluation_rules,
    }
    artifact["benchmark_id"] = compute_benchmark_id(artifact)
    artifact["protocol_id"] = compute_protocol_id(artifact)
    write_json(benchmark_path, artifact)
    write_csv(csv_path, (_csv_row(case) for case in cases), CSV_FIELDS)
    _write_manifest(manifest_path, benchmark_path, csv_path, artifact)
    LOGGER.info(
        "Frozen %d cases as benchmark %s (protocol %s)",
        len(cases),
        artifact["benchmark_id"],
        artifact["protocol_id"],
    )
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--offline", action="store_true", help="Use existing data/raw files only"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly rebuild and replace an existing frozen benchmark",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    build(args.config, offline=args.offline, force=args.force)


if __name__ == "__main__":
    main()
