"""Active evaluation targets, separate from immutable historical results."""

GPT = {
    "slug": "gpt-6-astra",
    "model": "gpt-6-astra",
    "directory": "gpt-6-astra",
    "display_name": "GPT-6 Astra",
    "reasoning_label": "High reasoning",
    "collection_method": "automated",
    "audit_label": "2/2 audits passed",
}
CLAUDE = {
    "slug": "claude-opus-5-5",
    "model": "claude-opus-5-5",
    "directory": "claude-opus-5-5",
    "display_name": "Claude Opus 5.5",
    "reasoning_label": "High reasoning",
    "collection_method": "automated",
    "audit_label": "2/2 audits passed",
}
GEMINI = {
    "slug": "gemini-4-argon",
    # Antigravity uses an effort-qualified slug, unlike the Gemini API.
    "model": "gemini-4-argon-high",
    "directory": "gemini-4-argon",
    "display_name": "Gemini 4 Argon",
    "runner_display_name": "Gemini 4 Argon (High)",
    "reasoning_label": "High reasoning",
    "collection_method": "automated",
    "audit_label": "2/2 audits passed",
}
GROK = {
    "slug": "grok-4.8",
    "model": "grok-4.8",
    "directory": "grok-4.8-manual",
    "display_name": "Grok 4.8",
    "reasoning_label": "Record mode when collecting responses",
    "collection_method": "manual",
    "audit_label": "Manual entry",
    "availability_note": (
        "Requested target; availability unverified. The public xAI model "
        "catalog lists Grok 4.7 as of October 21, 2026."
    ),
}

CURRENT_MODELS = (GPT, CLAUDE, GEMINI, GROK)
