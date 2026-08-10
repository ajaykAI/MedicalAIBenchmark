"""Build a new objective-only benchmark round without reusing prior questions."""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.benchmark import (
    compute_benchmark_id,
    compute_protocol_id,
    load_benchmark,
    validate_evaluation_rules,
)
from src.build_benchmark import (
    CSV_FIELDS,
    DATASET_MODULES,
    _assign_ids,
    _csv_row,
    _normalized_question,
    _select_without_duplicates,
)
from src.config import PROJECT_ROOT, load_config, project_path
from src.io_utils import sha256_file, write_csv, write_json
from src.schemas import BenchmarkCase, validate_benchmark_cases


LOGGER = logging.getLogger("medical_benchmark.build_objective_round")
DEFAULT_COUNTS = {"MedQA": 30, "MedMCQA": 10}
VISUAL_REFERENCE_PATTERNS = (
    re.compile(r"\bmarked\b.{0,30}\barrow\b", re.IGNORECASE),
    re.compile(r"\barrow[- ]marked\b", re.IGNORECASE),
    re.compile(
        r"\b(?:shown|depicted|pictured|illustrated)\b.{0,40}"
        r"\b(?:image|figure|diagram|photograph|picture)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:image|figure|diagram|photograph|picture)\b.{0,40}"
        r"\b(?:shown|above|below|following)\b",
        re.IGNORECASE,
    ),
)


def requires_missing_visual(case: BenchmarkCase) -> bool:
    """Return true when a text-only source explicitly depends on an unseen visual."""
    return any(pattern.search(case.question) for pattern in VISUAL_REFERENCE_PATTERNS)


def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_round_manifest(
    manifest_path: Path,
    benchmark_path: Path,
    csv_path: Path,
    artifact: Dict[str, Any],
) -> None:
    write_json(
        manifest_path,
        {
            "schema_version": artifact["schema_version"],
            "round_id": artifact["round_id"],
            "benchmark_id": artifact["benchmark_id"],
            "prior_benchmark_id": artifact["prior_benchmark_id"],
            "protocol_id": artifact["protocol_id"],
            "evaluation_rules": artifact["evaluation_rules"],
            "benchmark_file_sha256": sha256_file(benchmark_path),
            "benchmark_csv_sha256": sha256_file(csv_path),
            "random_seed": artifact["random_seed"],
            "case_counts": artifact["case_counts"],
            "exclusion_policy": artifact["exclusion_policy"],
            "source_revisions": {
                item["dataset"]: item["revision"] for item in artifact["sources"]
            },
        },
    )


def build_objective_round(
    config_path: Path,
    *,
    prior_benchmark_path: Path = Path("data/benchmark.json"),
    output_dir: Path = Path("data/round-2"),
    root: Path = PROJECT_ROOT,
    round_id: str = "round-2",
    seed: int = 43,
    counts: Optional[Dict[str, int]] = None,
    offline: bool = True,
    force: bool = False,
) -> Dict[str, Any]:
    """Freeze a deterministic objective round that excludes every prior item."""
    requested_counts = dict(counts or DEFAULT_COUNTS)
    if set(requested_counts) != {"MedQA", "MedMCQA"}:
        raise ValueError("Objective rounds require MedQA and MedMCQA counts")
    if any(isinstance(value, bool) or int(value) <= 0 for value in requested_counts.values()):
        raise ValueError("Objective round counts must be positive integers")
    requested_counts = {key: int(value) for key, value in requested_counts.items()}

    root = Path(root)
    config = load_config(config_path)
    evaluation_rules = config["benchmark"].get("evaluation_rules")
    validate_evaluation_rules(evaluation_rules)
    prior_path = _resolve(Path(prior_benchmark_path), root)
    destination = _resolve(Path(output_dir), root)
    benchmark_path = destination / "benchmark.json"
    csv_path = destination / "benchmark.csv"
    manifest_path = destination / "benchmark_manifest.json"
    prior_artifact, prior_cases = load_benchmark(prior_path)

    if benchmark_path.exists() and not force:
        artifact, cases = load_benchmark(benchmark_path)
        if artifact.get("round_id") != round_id:
            raise ValueError("Existing objective round has a different round_id")
        if artifact.get("prior_benchmark_id") != prior_artifact["benchmark_id"]:
            raise ValueError("Existing objective round excludes a different benchmark")
        if artifact.get("case_counts") != requested_counts:
            raise ValueError("Existing objective round has different case counts")
        if artifact.get("random_seed") != seed:
            raise ValueError("Existing objective round has a different random seed")
        LOGGER.info(
            "Frozen objective round already exists and passed integrity checks: %s (%d cases)",
            artifact["benchmark_id"],
            len(cases),
        )
        return artifact

    prior_source_ids: Set[Tuple[str, str]] = {
        (case.dataset, case.source_id) for case in prior_cases
    }
    seen_questions: Set[str] = {
        _normalized_question(case) for case in prior_cases
    }
    selected_cases: List[BenchmarkCase] = []
    sources = []

    for dataset in ("MedQA", "MedMCQA"):
        dataset_config = dict(config["datasets"][dataset])
        raw_path = project_path(dataset_config["raw_file"], root)
        if not raw_path.exists():
            if offline:
                raise FileNotFoundError(
                    "Offline mode requires the raw file: %s" % raw_path
                )
            DATASET_MODULES[dataset].acquire(dataset_config, raw_path)
        candidates = DATASET_MODULES[dataset].load(
            raw_path, dataset_config["split"]
        )
        eligible = [
            case
            for case in candidates
            if (case.dataset, case.source_id) not in prior_source_ids
            and _normalized_question(case) not in seen_questions
            and not requires_missing_visual(case)
        ]
        selected = _select_without_duplicates(
            eligible,
            requested_counts[dataset],
            seed=seed,
            dataset="%s-%s" % (dataset, round_id),
            seen_questions=seen_questions,
        )
        selected_cases.extend(selected)
        source_record = {
            "dataset": dataset,
            "source": dataset_config["source"],
            "upstream": dataset_config.get("upstream"),
            "revision": dataset_config["revision"],
            "split": dataset_config["split"],
            "license": dataset_config.get("license"),
            "raw_sha256": sha256_file(raw_path),
            "raw_rows_usable": len(candidates),
            "rows_eligible_after_exclusions": len(eligible),
            "excluded_prior_benchmark_id": prior_artifact["benchmark_id"],
        }
        if dataset_config.get("split_note"):
            source_record["split_note"] = dataset_config["split_note"]
        sources.append(source_record)

    cases = _assign_ids(selected_cases)
    validate_benchmark_cases(cases)
    actual_counts = dict(Counter(case.dataset for case in cases))
    if actual_counts != requested_counts:
        raise RuntimeError("Selected objective round counts do not match request")
    if any(
        (case.dataset, case.source_id) in prior_source_ids for case in cases
    ):
        raise RuntimeError("Objective round overlaps the prior benchmark")
    if any(requires_missing_visual(case) for case in cases):
        raise RuntimeError("Objective round contains an unresolved visual reference")

    artifact: Dict[str, Any] = {
        "schema_version": config["benchmark"]["schema_version"],
        "round_id": round_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "case_counts": actual_counts,
        "prior_benchmark_id": prior_artifact["benchmark_id"],
        "exclusion_policy": {
            "prior_source_ids": True,
            "prior_normalized_questions": True,
            "unresolved_visual_references": True,
        },
        "sources": sources,
        "cases": [case.to_dict() for case in cases],
        "evaluation_rules": evaluation_rules,
    }
    artifact["benchmark_id"] = compute_benchmark_id(artifact)
    artifact["protocol_id"] = compute_protocol_id(artifact)
    write_json(benchmark_path, artifact)
    write_csv(csv_path, (_csv_row(case) for case in cases), CSV_FIELDS)
    _write_round_manifest(manifest_path, benchmark_path, csv_path, artifact)
    LOGGER.info(
        "Frozen %s with %d objective cases as benchmark %s",
        round_id,
        len(cases),
        artifact["benchmark_id"],
    )
    return artifact


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument(
        "--prior-benchmark", type=Path, default=Path("data/benchmark.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/round-2"))
    parser.add_argument("--round-id", default="round-2")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--medqa-count", type=int, default=30)
    parser.add_argument("--medmcqa-count", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    artifact = build_objective_round(
        args.config,
        prior_benchmark_path=args.prior_benchmark,
        output_dir=args.output_dir,
        round_id=args.round_id,
        seed=args.seed,
        counts={"MedQA": args.medqa_count, "MedMCQA": args.medmcqa_count},
        offline=args.offline,
        force=args.force,
    )
    print(
        "%s %s %s"
        % (artifact["round_id"], artifact["benchmark_id"], artifact["case_counts"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
