"""Build eight answer-free Round 2 PDFs containing five questions each."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from src.build_question_packet_pdf import (
    CHUNK_SIZE,
    _footer,
    _instruction_box,
    _question_flowables,
    _register_fonts,
    _styles,
    load_question_only_rows,
)


def build_chunk(
    rows: Sequence[Mapping[str, Any]], chunk_index: int, output_path: Path
) -> None:
    styles = _styles()
    first_qid = int(rows[0]["qid"])
    last_qid = int(rows[-1]["qid"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.67 * inch,
        title="Medical AI Benchmark Round 2 - Questions %d-%d"
        % (first_qid, last_qid),
        author="Medical AI Benchmark",
        subject="Answer-free closed-book evaluation questions",
    )

    story: List[Any] = [
        Paragraph("Round 2 - Chunk %d of 8" % chunk_index, styles["section"]),
        Paragraph(
            "Questions %d-%d | Five objective questions | Answers not included"
            % (first_qid, last_qid),
            styles["section_meta"],
        ),
        _instruction_box(styles),
        Spacer(1, 15),
    ]
    for row in rows:
        question_flowables = _question_flowables(row, styles)
        story.append(KeepTogether(question_flowables[1:]))
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def build_split_packets(benchmark_path: Path, output_dir: Path) -> List[Path]:
    rows = load_question_only_rows(benchmark_path)
    _register_fonts()
    outputs: List[Path] = []
    for offset in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[offset : offset + CHUNK_SIZE]
        first_qid = int(chunk[0]["qid"])
        last_qid = int(chunk[-1]["qid"])
        output = output_dir / (
            "medical_benchmark_round2_questions_%02d-%02d.pdf"
            % (first_qid, last_qid)
        )
        build_chunk(chunk, offset // CHUNK_SIZE + 1, output)
        outputs.append(output)
    return outputs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/round-2/benchmark.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/pdf/round-2-grok")
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for output in build_split_packets(args.benchmark, args.output_dir):
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
