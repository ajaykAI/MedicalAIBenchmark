"""Build an answer-free PDF containing the 40 objective benchmark questions.

The packet is divided into eight five-question chunks for manual, closed-book
model evaluation. Only sequential qIDs, question text, and A-D choices are
rendered; benchmark labels, source metadata, rubrics, and gold responses are
never written to the document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


CHUNK_SIZE = 5
EXPECTED_QUESTION_COUNT = 40
OPTION_LABELS = ("A", "B", "C", "D")
FONT_REGULAR = Path("/Library/Fonts/Arial Unicode.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#2563EB")
PALE_BLUE = colors.HexColor("#EFF6FF")
INK = colors.HexColor("#172B4D")
MUTED = colors.HexColor("#52667A")
LINE = colors.HexColor("#D9E2EC")
PAPER = colors.HexColor("#F7FAFC")


class PacketError(RuntimeError):
    """Raised when the source cannot produce the expected answer-free packet."""


def _normalized_text(value: str) -> str:
    """Preserve wording while using PDF-safe ASCII hyphens."""
    return (
        value.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )


def _paragraph_text(value: str) -> str:
    return escape(_normalized_text(value)).replace("\n", "<br/>")


def load_question_only_rows(path: Path) -> List[Dict[str, Any]]:
    """Return only the fields needed for the manual question packet."""
    with path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    cases = artifact.get("cases")
    if not isinstance(cases, list):
        raise PacketError("benchmark cases must be an array")

    rows: List[Dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise PacketError("benchmark case must be an object")
        choices = case.get("choices")
        if not isinstance(choices, Mapping):
            continue
        if set(choices) != set(OPTION_LABELS):
            raise PacketError("objective case must contain choices A-D")
        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            raise PacketError("objective case has an empty question")
        if not all(isinstance(choices[label], str) for label in OPTION_LABELS):
            raise PacketError("objective choice text must be a string")
        rows.append(
            {
                "qid": len(rows) + 1,
                "question": question,
                "choices": {label: choices[label] for label in OPTION_LABELS},
            }
        )

    if len(rows) != EXPECTED_QUESTION_COUNT:
        raise PacketError(
            "expected %d objective questions, found %d"
            % (EXPECTED_QUESTION_COUNT, len(rows))
        )
    return rows


def _register_fonts() -> None:
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise PacketError("required Arial fonts are unavailable")
    pdfmetrics.registerFont(TTFont("PacketArial", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("PacketArialBold", str(FONT_BOLD)))


def _styles() -> Dict[str, ParagraphStyle]:
    defaults = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=defaults["Title"],
            fontName="PacketArialBold",
            fontSize=28,
            leading=33,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=defaults["BodyText"],
            fontName="PacketArial",
            fontSize=12,
            leading=18,
            textColor=MUTED,
            spaceAfter=18,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=defaults["Heading1"],
            fontName="PacketArialBold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceAfter=5,
        ),
        "section_meta": ParagraphStyle(
            "SectionMeta",
            parent=defaults["BodyText"],
            fontName="PacketArial",
            fontSize=9.5,
            leading=13,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "instructions": ParagraphStyle(
            "Instructions",
            parent=defaults["BodyText"],
            fontName="PacketArial",
            fontSize=9.5,
            leading=14,
            textColor=INK,
        ),
        "qid": ParagraphStyle(
            "Qid",
            parent=defaults["Heading2"],
            fontName="PacketArialBold",
            fontSize=12,
            leading=15,
            textColor=colors.white,
            spaceAfter=0,
        ),
        "question": ParagraphStyle(
            "Question",
            parent=defaults["BodyText"],
            fontName="PacketArial",
            fontSize=10.2,
            leading=14.6,
            textColor=INK,
            spaceAfter=7,
            allowWidows=0,
            allowOrphans=0,
        ),
        "choice_label": ParagraphStyle(
            "ChoiceLabel",
            parent=defaults["BodyText"],
            fontName="PacketArialBold",
            fontSize=9.7,
            leading=13.5,
            alignment=TA_CENTER,
            textColor=NAVY,
        ),
        "choice": ParagraphStyle(
            "Choice",
            parent=defaults["BodyText"],
            fontName="PacketArial",
            fontSize=9.7,
            leading=13.5,
            textColor=INK,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=defaults["BodyText"],
            fontName="PacketArial",
            fontSize=8.5,
            leading=12,
            textColor=MUTED,
        ),
    }


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(0.62 * inch, 0.48 * inch, width - 0.62 * inch, 0.48 * inch)
    canvas.setFont("PacketArial", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.62 * inch, 0.29 * inch, "Medical AI Benchmark - Questions Only")
    canvas.drawRightString(
        width - 0.62 * inch, 0.29 * inch, "Page %d" % document.page
    )
    canvas.restoreState()


def _instruction_box(styles: Mapping[str, ParagraphStyle]) -> Table:
    text = (
        "<b>Instructions for Grok:</b> Complete this closed-book medical "
        "benchmark using only your internal knowledge and reasoning. Do not "
        "search the internet, browse, open links, use tools, consult outside "
        "sources, or ask another agent. Answer all five questions exactly once. "
        "Return only five rows in the format <b>qID [tab] Response</b>, using "
        "one option label: A, B, C, or D."
    )
    box = Table(
        [[Paragraph(text, styles["instructions"])]],
        colWidths=[7.0 * inch],
        hAlign="LEFT",
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#93C5FD")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return box


def _question_flowables(
    row: Mapping[str, Any], styles: Mapping[str, ParagraphStyle]
) -> List[Any]:
    qid = int(row["qid"])
    choices = row["choices"]
    header = Table(
        [[Paragraph("qID %d" % qid, styles["qid"])]],
        colWidths=[7.0 * inch],
        hAlign="LEFT",
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    flowables: List[Any] = [CondPageBreak(1.65 * inch), header, Spacer(1, 7)]
    flowables.append(
        Paragraph(_paragraph_text(str(row["question"])), styles["question"])
    )
    for label in OPTION_LABELS:
        choice = Table(
            [
                [
                    Paragraph(label, styles["choice_label"]),
                    Paragraph(_paragraph_text(str(choices[label])), styles["choice"]),
                ]
            ],
            colWidths=[0.36 * inch, 6.64 * inch],
            hAlign="LEFT",
        )
        choice.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#E6EEF7")),
                    ("BACKGROUND", (1, 0), (1, 0), PAPER),
                    ("BOX", (0, 0), (-1, -1), 0.45, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        flowables.extend([choice, Spacer(1, 3)])
    flowables.append(Spacer(1, 13))
    return flowables


def build_packet(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    _register_fonts()
    styles = _styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.68 * inch,
        bottomMargin=0.67 * inch,
        title="Medical AI Benchmark - Objective Questions",
        author="Medical AI Benchmark",
        subject="Answer-free manual evaluation packet",
    )

    story: List[Any] = [
        Spacer(1, 0.42 * inch),
        Paragraph("Medical AI Benchmark", styles["cover_title"]),
        Paragraph(
            "Objective question packet for manual, closed-book evaluation",
            styles["cover_subtitle"],
        ),
    ]
    summary = Table(
        [
            ["40", "8", "5", "A-D"],
            ["Questions", "Chunks", "Per chunk", "Response format"],
        ],
        colWidths=[1.75 * inch] * 4,
        hAlign="LEFT",
    )
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "PacketArialBold"),
                ("FONTSIZE", (0, 0), (-1, 0), 17),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, 1), PAPER),
                ("TEXTCOLOR", (0, 1), (-1, 1), MUTED),
                ("FONTNAME", (0, 1), (-1, 1), "PacketArial"),
                ("FONTSIZE", (0, 1), (-1, 1), 8.5),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.extend(
        [
            summary,
            Spacer(1, 0.34 * inch),
            Paragraph("How to use this packet", styles["section"]),
            Paragraph(
                "Run each chunk twice in separate, fresh Grok Free chats. Keep "
                "Search and DeepSearch off. Record the five labels exactly as "
                "returned. If Grok displays web citations or says it searched, "
                "discard that attempt and rerun the chunk closed-book.",
                styles["cover_subtitle"],
            ),
            _instruction_box(styles),
            Spacer(1, 0.24 * inch),
            Paragraph(
                "This document contains only qIDs, question text, and answer "
                "choices. It does not include benchmark scoring fields or source "
                "metadata.",
                styles["small"],
            ),
        ]
    )

    chunk_count = (len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for chunk_index in range(chunk_count):
        start = chunk_index * CHUNK_SIZE
        chunk = rows[start : start + CHUNK_SIZE]
        first_qid = int(chunk[0]["qid"])
        last_qid = int(chunk[-1]["qid"])
        story.extend(
            [
                PageBreak(),
                Paragraph("Chunk %d of %d" % (chunk_index + 1, chunk_count), styles["section"]),
                Paragraph(
                    "Questions %d-%d | Run this complete chunk twice in two fresh chats"
                    % (first_qid, last_qid),
                    styles["section_meta"],
                ),
                _instruction_box(styles),
                Spacer(1, 15),
            ]
        )
        for row in chunk:
            story.extend(_question_flowables(row, styles))

        response_rows: List[List[Any]] = [
            [
                Paragraph("qID", styles["choice_label"]),
                Paragraph("Response", styles["choice_label"]),
            ]
        ]
        response_rows.extend(
            [
                Paragraph(str(row["qid"]), styles["choice_label"]),
                Paragraph("", styles["choice"]),
            ]
            for row in chunk
        )
        response_table = Table(
            response_rows,
            colWidths=[1.0 * inch, 2.0 * inch],
            rowHeights=[0.28 * inch] * len(response_rows),
            hAlign="LEFT",
        )
        response_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EEF7")),
                    ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ]
            )
        )
        story.extend(
            [
                CondPageBreak(2.2 * inch),
                Paragraph("Response capture", styles["section_meta"]),
                response_table,
            ]
        )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/benchmark.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/medical_benchmark_40_questions.pdf"),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = load_question_only_rows(args.benchmark)
    build_packet(rows, args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
