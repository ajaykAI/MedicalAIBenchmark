"""Reproducible complexity screening, NOT an empirical difficulty estimator.

Scores only prioritize questions for AI-assisted content review. Original
answer keys and evaluated model responses are never used to rank candidates.
"""

import re
from collections import defaultdict
from typing import Any, Dict, List, Sequence

SCREENING_VERSION = "clinical-complexity-screen-v1"
EVIDENCE_PATTERNS = {
    "laboratory": r"\b(?:laboratory|serum|plasma|blood count|creatinine|sodium|potassium|hemoglobin|bicarbonate)\b",
    "imaging": r"\b(?:CT|MRI|radiograph|ultrasound|echocardiogra\w*|tomography)\b",
    "pathology": r"\b(?:biopsy|histolog\w*|microscop\w*|cytolog\w*|stain\w*)\b",
    "physiology": r"\b(?:pH|PaCO2|pCO2|arterial blood gas|osmolality|anion gap|ejection fraction|FEV1|FVC|ECG|electrocardiogram)\b",
    "microbiology": r"\b(?:culture|PCR|antibod\w*|antigen|serolog\w*)\b",
    "medications": r"\b(?:medications?|therapy|treated|treatment|administered|prescribed)\b",
}
COMPLEXITY_PATTERNS = {
    "competing_conditions": r"\b(?:history of|diabet\w*|pregnan\w*|renal|kidney|hepatic|liver|immunosuppress\w*|anticoagul\w*|transplant\w*)\b",
    "evolution_or_response": r"\b(?:despite|worsen\w*|persist\w*|recur\w*|subsequent\w*|follow-up|failed|refractory|previously|initially)\b",
    "interpretation": r"\b(?:mechanism|underlying|explanation|interpret\w*|calculate\w*|expected|responsible|differentiat\w*)\b",
    "decision": r"\b(?:next step|most appropriate|best next|management|contraindicat\w*|treatment|intervention)\b",
}
VISUAL_REFERENCE = re.compile(
    r"\b(?:shown|pictured|depicted|illustrated)\b.{0,70}\b(?:figure|image|photograph|diagram)\b"
    r"|\b(?:figure|image|photograph|diagram)\s+(?:[A-Z]|\d|below|above)\b",
    re.I,
)


def stem(row: Dict[str, Any]) -> str:
    return row["question"].split("Answer Choices:", 1)[0].rstrip()


def screen(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for row in rows:
        text = stem(row)
        reasons = []
        if row.get("question_type") != "Reasoning":
            reasons.append("outside reasoning-focused scope")
        if VISUAL_REFERENCE.search(text):
            reasons.append("possible unresolved visual reference")
        features = {
            name: min(len(re.findall(pattern, text, re.I)), 3)
            for name, pattern in COMPLEXITY_PATTERNS.items()
        }
        evidence = [name for name, pattern in EVIDENCE_PATTERNS.items() if re.search(pattern, text, re.I)]
        features["evidence_domains"] = len(evidence)
        features["numerical_findings"] = min(len(re.findall(r"\b\d+(?:\.\d+)?\b", text)), 10)
        # Length has a small capped contribution; it is not a difficulty label.
        features["vignette_length"] = min(len(text.split()) // 100, 3)
        score = (
            3 * features["evidence_domains"]
            + 2 * features["evolution_or_response"]
            + 2 * features["interpretation"]
            + features["competing_conditions"] + features["decision"]
            + features["numerical_findings"] / 2 + features["vignette_length"]
        )
        ranked.append({
            "source_id": row["id"], "body_system": row["body_system"],
            "medical_task": row["medical_task"], "screen_score": score,
            "features": features, "evidence_domains": evidence,
            "exclusions": reasons,
        })
    return sorted(ranked, key=lambda item: (-item["screen_score"], item["source_id"]))


def shortlist(screened: Sequence[Dict[str, Any]], per_system: int = 12) -> List[str]:
    groups = defaultdict(list)
    for item in screened:
        if not item["exclusions"] and len(groups[item["body_system"]]) < per_system:
            groups[item["body_system"]].append(item["source_id"])
    return [source_id for system in sorted(groups) for source_id in groups[system]]
