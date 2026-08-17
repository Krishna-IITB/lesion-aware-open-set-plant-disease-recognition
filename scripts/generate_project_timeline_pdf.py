"""Generate the phase-level December 2025-January 2026 project timeline PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "project-timeline-dec-2025-jan-2026.pdf"

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#188977")
GREEN = colors.HexColor("#51B48E")
PALE = colors.HexColor("#EAF5F1")
INK = colors.HexColor("#22313F")
MUTED = colors.HexColor("#607180")
LINE = colors.HexColor("#C8D8D3")
WHITE = colors.white


def register_fonts() -> tuple[str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    regular = next((path for path in candidates if path.exists()), None)
    bold = next((path for path in bold_candidates if path.exists()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("TimelineSans", str(regular)))
        pdfmetrics.registerFont(TTFont("TimelineSans-Bold", str(bold)))
        return "TimelineSans", "TimelineSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 7.5)
    canvas.drawString(18 * mm, 9 * mm, "Lesion-Aware Open-Set Plant Disease Recognition")
    canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def phase_card(month: str, period: str, title: str, bullets: list[str], styles) -> Table:  # type: ignore[no-untyped-def]
    label = Table(
        [[Paragraph(month, styles["month"]), Paragraph(period, styles["period"])]],
        colWidths=[31 * mm, 38 * mm],
    )
    label.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), TEAL),
                ("BACKGROUND", (1, 0), (1, 0), PALE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    body = [Paragraph(title, styles["phase_title"])]
    body.extend(
        Paragraph(f"<font color='#188977'>&#8226;</font> {item}", styles["bullet"])
        for item in bullets
    )
    content = Table([[body]], colWidths=[99 * mm])
    content.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    row = Table([[label, content]], colWidths=[69 * mm, 99 * mm], hAlign="LEFT")
    row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return row


def build_pdf() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="eyebrow",
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10,
            textColor=GREEN,
            tracking=1.2,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="title_main",
            fontName=FONT_BOLD,
            fontSize=24,
            leading=28,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="subtitle",
            fontName=FONT,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#D9E8E4"),
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="section",
            fontName=FONT_BOLD,
            fontSize=14,
            leading=17,
            textColor=NAVY,
            spaceBefore=5,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="month",
            fontName=FONT_BOLD,
            fontSize=9,
            leading=11,
            textColor=WHITE,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="period",
            fontName=FONT_BOLD,
            fontSize=8,
            leading=10,
            textColor=TEAL,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="phase_title",
            fontName=FONT_BOLD,
            fontSize=10,
            leading=12,
            textColor=NAVY,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="bullet",
            fontName=FONT,
            fontSize=8.2,
            leading=11,
            textColor=INK,
            leftIndent=0,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(name="body_small", fontName=FONT, fontSize=8.5, leading=12, textColor=INK)
    )
    styles.add(
        ParagraphStyle(
            name="metric",
            fontName=FONT_BOLD,
            fontSize=9.2,
            leading=11,
            textColor=NAVY,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="step_code",
            fontName=FONT_BOLD,
            fontSize=11,
            leading=13,
            textColor=WHITE,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="metric_label",
            fontName=FONT,
            fontSize=7.2,
            leading=9,
            textColor=MUTED,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(name="note", fontName=FONT, fontSize=7.7, leading=10.5, textColor=MUTED)
    )

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=20 * mm,
        title="Project Timeline: December 2025-January 2026",
        author="Lesion-Aware Open-Set Plant Disease Recognition project",
        subject="Academic project timeline",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates(PageTemplate(id="timeline", frames=[frame], onPage=footer))

    story = []
    hero = Table(
        [
            [
                [
                    Paragraph("ACADEMIC PROJECT TIMELINE", styles["eyebrow"]),
                    Paragraph(
                        "Lesion-Aware Open-Set<br/>Plant Disease Recognition", styles["title_main"]
                    ),
                    Paragraph(
                        "Extension of ACM MM'24 MVPDR  |  December 2025-January 2026",
                        styles["subtitle"],
                    ),
                ]
            ]
        ],
        colWidths=[174 * mm],
    )
    hero.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
            ]
        )
    )
    story.extend(
        [
            hero,
            Spacer(1, 8 * mm),
            Paragraph("December 2025 - foundation and lesion-aware recognition", styles["section"]),
        ]
    )

    december = [
        (
            "DEC 2025",
            "EARLY",
            "Scope and evaluation design",
            [
                "Defined the student-scale MVPDR extension and its four-component boundary.",
                "Set the few-shot lab-to-field, held-out-class, and validation-only "
                "evaluation contract.",
            ],
        ),
        (
            "DEC 2025",
            "MID",
            "CLIP baseline and frozen DINOv3 local features",
            [
                "Structured textual and global visual prototype evidence.",
                "Added dense local features for lesion texture, colour, and shape with "
                "cache provenance.",
            ],
        ),
        (
            "DEC 2025",
            "LATE",
            "PlantSeg-supervised lesion decoder",
            [
                "Trained a lightweight decoder with BCE + Dice supervision over frozen "
                "patch features.",
                "Pooled classification features using predicted masks; test masks remained "
                "evaluation-only.",
            ],
        ),
    ]
    for month, period, title, bullets in december:
        story.extend(
            [KeepTogether(phase_card(month, period, title, bullets, styles)), Spacer(1, 3.5 * mm)]
        )

    story.append(
        Paragraph(
            "January 2026 - fusion, open-set evaluation, and consolidation", styles["section"]
        )
    )
    january = [
        (
            "JAN 2026",
            "EARLY",
            "Three-view gated fusion",
            [
                "Fused textual, global, and lesion-localized prototype evidence with a "
                "per-image gate.",
                "Preserved a parameter-efficient design: 0.8% trainable parameters.",
            ],
        ),
        (
            "JAN 2026",
            "MID",
            "Calibration and held-out disease detection",
            [
                "Added temperature scaling and validation-selected open-set rejection.",
                "Evaluated 20-shot lab-to-field recognition and held-out disease detection.",
            ],
        ),
        (
            "JAN 2026",
            "LATE",
            "Analysis and academic project handoff",
            [
                "Consolidated the A-E experimental progression and documented method boundaries.",
                "Closed the academic project period in January 2026.",
            ],
        ),
    ]
    for month, period, title, bullets in january:
        story.extend(
            [KeepTogether(phase_card(month, period, title, bullets, styles)), Spacer(1, 3.5 * mm)]
        )

    story.extend(
        [
            PageBreak(),
            Paragraph("PROJECT SUMMARY", styles["eyebrow"]),
            Paragraph("From baseline to open-set recognition", styles["section"]),
            Paragraph(
                "The project advanced through five controlled stages, with each stage adding one "
                "testable capability while keeping both foundation encoders frozen.",
                styles["body_small"],
            ),
            Spacer(1, 5 * mm),
        ]
    )

    progression = [
        ("A", "MVPDR-style baseline", "CLIP text + global prototypes"),
        ("B", "Local representation", "+ frozen DINOv3 patch features"),
        ("C", "Lesion supervision", "+ predicted-lesion feature pooling"),
        ("D", "Three-view fusion", "+ learned per-image evidence gate"),
        ("E", "Open-set recognition", "+ calibrated unknown rejection"),
    ]
    progression_cells = []
    for code, title, detail in progression:
        progression_cells.append(
            [
                Paragraph(code, styles["step_code"]),
                Paragraph(
                    f"<b>{title}</b><br/><font color='#607180'>{detail}</font>", styles["bullet"]
                ),
            ]
        )
    progression_table = Table(progression_cells, colWidths=[15 * mm, 159 * mm])
    progression_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), TEAL),
                ("TEXTCOLOR", (0, 0), (0, -1), WHITE),
                ("ROWBACKGROUNDS", (1, 0), (1, -1), [WHITE, colors.HexColor("#F4F7F6")]),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend(
        [
            progression_table,
            Spacer(1, 7 * mm),
            Paragraph("Historical outcomes", styles["section"]),
        ]
    )
    metrics = [
        ("0.8%", "trainable parameters"),
        ("0.76", "mean binary-lesion Dice"),
        ("67%", "20-shot lab-to-field macro-F1"),
        ("+6 pp", "vs same-split MVPDR reproduction"),
        ("0.90", "open-set AUROC"),
        ("40%", "FPR95"),
    ]
    cells = [
        [Paragraph(value, styles["metric"]) for value, _ in metrics],
        [Paragraph(label, styles["metric_label"]) for _, label in metrics],
    ]
    metric_table = Table(cells, colWidths=[29 * mm] * 6, rowHeights=[8 * mm, 11 * mm])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend(
        [
            metric_table,
            Spacer(1, 7 * mm),
            Paragraph("Evaluation safeguards", styles["section"]),
        ]
    )
    guardrails = [
        [
            Paragraph(
                "<b>Leakage control</b><br/>Training samples build prototypes; validation selects "
                "temperature and rejection thresholds; target-domain test data is evaluated once.",
                styles["body_small"],
            ),
            Paragraph(
                "<b>Mask integrity</b><br/>Ground-truth masks supervise and evaluate the decoder. "
                "Classification inference pools only under predicted lesion masks.",
                styles["body_small"],
            ),
        ],
        [
            Paragraph(
                "<b>Held-out diseases</b><br/>Unknown classes are excluded from training "
                "prototypes "
                "and represented in validation and test for open-set calibration and evaluation.",
                styles["body_small"],
            ),
            Paragraph(
                "<b>Student-scale scope</b><br/>CLIP and DINOv3 remain frozen; only the "
                "lightweight "
                "decoder, evidence gate, and calibration components are trained.",
                styles["body_small"],
            ),
        ],
    ]
    guardrail_table = Table(guardrails, colWidths=[87 * mm, 87 * mm])
    guardrail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7F6")),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([guardrail_table, Spacer(1, 7 * mm)])
    note = (
        "<b>Provenance.</b> These are author-supplied historical results from the "
        "December 2025-January 2026 academic project. The current Git repository is a later "
        "reproducibility reconstruction and has not independently rerun the corresponding "
        "real-data experiments. "
        "This phase timeline is not a fabricated commit or activity log."
    )
    provenance = Table([[Paragraph(note, styles["note"])]], colWidths=[174 * mm])
    provenance.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F7F6")),
                ("LINEBEFORE", (0, 0), (0, 0), 3, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(provenance)

    doc.build(story)


if __name__ == "__main__":
    build_pdf()
