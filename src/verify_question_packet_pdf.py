"""Verify and render the answer-free objective-question PDF."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader


EXPECTED_QUESTION_COUNT = 40
EXPECTED_CHUNK_COUNT = 8
OPTION_LABELS = ("A", "B", "C", "D")
FORBIDDEN_EXPORT_MARKERS = (
    "correct_answer",
    "source_metadata",
    "source_id",
    "benchmark_id",
    "protocol_id",
    "medqa_",
    "medmcqa_",
    "healthbench",
)


class VerificationError(RuntimeError):
    """Raised when packet QA detects an integrity problem."""


def _normalize(value: str) -> str:
    return " ".join(
        value.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .split()
    )


def _load_objective_cases(path: Path) -> List[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        raise VerificationError("benchmark cases are missing")
    objective = [
        case
        for case in cases
        if isinstance(case, Mapping) and isinstance(case.get("choices"), Mapping)
    ]
    if len(objective) != EXPECTED_QUESTION_COUNT:
        raise VerificationError("unexpected objective-question count")
    return objective


def verify_text(pdf_path: Path, benchmark_path: Path) -> Mapping[str, Any]:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    content_text = text.replace("Medical AI Benchmark - Questions Only", "")
    content_text = re.sub(r"(?m)^Page\s+\d+\s*$", "", content_text)
    normalized = _normalize(content_text)
    lowered = normalized.casefold()

    for marker in FORBIDDEN_EXPORT_MARKERS:
        if marker.casefold() in lowered:
            raise VerificationError("forbidden export marker found: %s" % marker)

    qids = [int(value) for value in re.findall(r"\bqID\s+(\d+)\b", text)]
    if qids != list(range(1, EXPECTED_QUESTION_COUNT + 1)):
        raise VerificationError("qID sequence is incomplete or duplicated")
    chunks = [int(value) for value in re.findall(r"Chunk\s+(\d+)\s+of\s+8", text)]
    if chunks != list(range(1, EXPECTED_CHUNK_COUNT + 1)):
        raise VerificationError("chunk sequence is incomplete or duplicated")

    cases = _load_objective_cases(benchmark_path)
    for index, case in enumerate(cases, start=1):
        question = case.get("question")
        choices = case.get("choices")
        if not isinstance(question, str) or not isinstance(choices, Mapping):
            raise VerificationError("source question %d is malformed" % index)
        if _normalize(question) not in normalized:
            raise VerificationError("question %d text is missing or changed" % index)
        for label in OPTION_LABELS:
            choice = choices.get(label)
            if not isinstance(choice, str) or _normalize(choice) not in normalized:
                raise VerificationError(
                    "question %d choice %s is missing or changed" % (index, label)
                )

    metadata = reader.metadata or {}
    return {
        "page_count": len(reader.pages),
        "question_count": len(qids),
        "chunk_count": len(chunks),
        "forbidden_export_markers": 0,
        "source_text_integrity": True,
        "title": metadata.get("/Title"),
    }


def render_pages(pdf_path: Path, render_dir: Path) -> List[Path]:
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    rendered: List[Path] = []
    for index in range(len(document)):
        page = document[index]
        image = page.render(scale=1.55).to_pil().convert("RGB")
        path = render_dir / ("page-%03d.png" % (index + 1))
        image.save(path, format="PNG", optimize=True)
        rendered.append(path)
    return rendered


def build_contact_sheets(page_paths: Sequence[Path], render_dir: Path) -> List[Path]:
    sheets: List[Path] = []
    columns = 3
    per_sheet = 9
    thumb_width = 300
    gutter = 24
    label_height = 28
    background = (226, 232, 240)
    font = ImageFont.load_default()

    for sheet_index, start in enumerate(range(0, len(page_paths), per_sheet), start=1):
        group = page_paths[start : start + per_sheet]
        thumbnails = []
        for path in group:
            with Image.open(path) as source:
                thumb = source.convert("RGB")
                height = round(thumb.height * thumb_width / thumb.width)
                thumb.thumbnail((thumb_width, height), Image.Resampling.LANCZOS)
                thumbnails.append((path, thumb.copy()))
        rows = (len(thumbnails) + columns - 1) // columns
        thumb_height = max(image.height for _, image in thumbnails)
        sheet = Image.new(
            "RGB",
            (
                columns * thumb_width + (columns + 1) * gutter,
                rows * (thumb_height + label_height) + (rows + 1) * gutter,
            ),
            background,
        )
        draw = ImageDraw.Draw(sheet)
        for position, (path, image) in enumerate(thumbnails):
            row, column = divmod(position, columns)
            x = gutter + column * (thumb_width + gutter)
            y = gutter + row * (thumb_height + label_height)
            sheet.paste(image, (x, y + label_height))
            draw.text((x, y + 7), path.stem.replace("-", " ").title(), fill=(23, 43, 77), font=font)
        sheet_path = render_dir / ("contact-sheet-%02d.png" % sheet_index)
        sheet.save(sheet_path, format="PNG", optimize=True)
        sheets.append(sheet_path)
    return sheets


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.json"))
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=Path("tmp/pdfs/question_packet_render"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = dict(verify_text(args.pdf, args.benchmark))
    pages = render_pages(args.pdf, args.render_dir)
    sheets = build_contact_sheets(pages, args.render_dir)
    result.update(
        {
            "rendered_pages": len(pages),
            "contact_sheets": [str(path) for path in sheets],
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
