"""Verify the eight answer-free Round 2 question PDFs and render them for QA."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

from src.build_question_packet_pdf import _normalized_text


CHUNK_SIZE = 5
EXPECTED_COUNT = 40
OPTION_LABELS = ("A", "B", "C", "D")
FORBIDDEN_MARKERS = (
    "correct answer",
    "correct_answer",
    "answer key",
    "source metadata",
    "source_id",
    "benchmark_id",
    "protocol_id",
    "medqa_",
    "medmcqa_",
    "healthbench",
)


class VerificationError(RuntimeError):
    """Raised when a split packet is incomplete or leaks benchmark metadata."""


def _normalize(value: str) -> str:
    return " ".join(_normalized_text(value).split())


def _load_questions(path: Path) -> List[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        artifact = json.load(handle)
    cases = artifact.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_COUNT:
        raise VerificationError("benchmark must contain exactly 40 questions")
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("choices"), Mapping):
            raise VerificationError("benchmark contains a non-objective case")
    return cases


def verify_packets(
    benchmark_path: Path, pdf_dir: Path, render_dir: Path
) -> Dict[str, Any]:
    cases = _load_questions(benchmark_path)
    pdfs = sorted(pdf_dir.glob("medical_benchmark_round2_questions_*.pdf"))
    if len(pdfs) != EXPECTED_COUNT // CHUNK_SIZE:
        raise VerificationError("expected exactly eight chunk PDFs")

    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True)

    all_qids: List[int] = []
    rendered: List[Path] = []
    packet_results: List[Dict[str, Any]] = []
    for chunk_index, pdf_path in enumerate(pdfs, start=1):
        start = (chunk_index - 1) * CHUNK_SIZE
        expected_cases = cases[start : start + CHUNK_SIZE]
        expected_qids = list(range(start + 1, start + CHUNK_SIZE + 1))
        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        content_text = text.replace("Medical AI Benchmark - Questions Only", "")
        content_text = re.sub(r"(?m)^Page\s+\d+\s*$", "", content_text)
        normalized = _normalize(content_text)
        lowered = normalized.casefold()
        for marker in FORBIDDEN_MARKERS:
            if marker.casefold() in lowered:
                raise VerificationError(
                    "%s contains forbidden marker %r" % (pdf_path.name, marker)
                )
        qids = [int(value) for value in re.findall(r"\bqID\s+(\d+)\b", text)]
        if qids != expected_qids:
            raise VerificationError(
                "%s has qIDs %r, expected %r" % (pdf_path.name, qids, expected_qids)
            )
        all_qids.extend(qids)

        for qid, case in zip(expected_qids, expected_cases):
            question = case["question"]
            choices = case["choices"]
            if _normalize(question) not in normalized:
                raise VerificationError("question %d text is missing or changed" % qid)
            for label in OPTION_LABELS:
                if _normalize(choices[label]) not in normalized:
                    raise VerificationError(
                        "question %d choice %s is missing or changed" % (qid, label)
                    )

        document = pdfium.PdfDocument(str(pdf_path))
        for page_index in range(len(document)):
            page = document[page_index]
            image = page.render(scale=1.55).to_pil().convert("RGB")
            image_path = render_dir / (
                "chunk-%02d-page-%02d.png" % (chunk_index, page_index + 1)
            )
            image.save(image_path, format="PNG", optimize=True)
            rendered.append(image_path)
        packet_results.append(
            {
                "file": pdf_path.name,
                "page_count": len(reader.pages),
                "qids": qids,
            }
        )

    if all_qids != list(range(1, EXPECTED_COUNT + 1)):
        raise VerificationError("combined qID sequence is incomplete")
    sheets = build_contact_sheets(rendered, render_dir)
    return {
        "packet_count": len(pdfs),
        "question_count": len(all_qids),
        "forbidden_markers": 0,
        "source_text_integrity": True,
        "rendered_pages": len(rendered),
        "packets": packet_results,
        "contact_sheets": [str(path) for path in sheets],
    }


def build_contact_sheets(page_paths: Sequence[Path], render_dir: Path) -> List[Path]:
    sheets: List[Path] = []
    columns = 3
    per_sheet = 9
    thumb_width = 300
    gutter = 24
    label_height = 28
    font = ImageFont.load_default()
    for sheet_index, start in enumerate(range(0, len(page_paths), per_sheet), start=1):
        group = page_paths[start : start + per_sheet]
        thumbs = []
        for path in group:
            with Image.open(path) as source:
                image = source.convert("RGB")
                height = round(image.height * thumb_width / image.width)
                image.thumbnail((thumb_width, height), Image.Resampling.LANCZOS)
                thumbs.append((path, image.copy()))
        rows = (len(thumbs) + columns - 1) // columns
        thumb_height = max(image.height for _, image in thumbs)
        sheet = Image.new(
            "RGB",
            (
                columns * thumb_width + (columns + 1) * gutter,
                rows * (thumb_height + label_height) + (rows + 1) * gutter,
            ),
            (226, 232, 240),
        )
        draw = ImageDraw.Draw(sheet)
        for position, (path, image) in enumerate(thumbs):
            row, column = divmod(position, columns)
            x = gutter + column * (thumb_width + gutter)
            y = gutter + row * (thumb_height + label_height)
            sheet.paste(image, (x, y + label_height))
            draw.text((x, y + 7), path.stem, fill=(23, 43, 77), font=font)
        sheet_path = render_dir / ("contact-sheet-%02d.png" % sheet_index)
        sheet.save(sheet_path, format="PNG", optimize=True)
        sheets.append(sheet_path)
    return sheets


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/round-2/benchmark.json")
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=Path("output/pdf/round-2-grok")
    )
    parser.add_argument(
        "--render-dir", type=Path, default=Path("tmp/pdfs/round-2-grok-render")
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = verify_packets(args.benchmark, args.pdf_dir, args.render_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
