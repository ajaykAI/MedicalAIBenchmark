# Medical AI Benchmark — Dataset Builder

This project implements the dataset download, normalization, deterministic sampling, frozen 60-case benchmark export, a local read-only question viewer, and closed-book model runners for the frozen benchmark.

This is an evaluation-data project, not a medical advice system. Never train or tune on the frozen benchmark.

## Frozen benchmark design

| Dataset | Cases | Source split | Reason |
|---|---:|---|---|
| MedQA | 30 | Labeled USMLE four-option test | Public gold answers support later exact-match evaluation |
| MedMCQA | 10 | Labeled validation | The public test split withholds gold answers (`cop=-1`) |
| HealthBench | 20 | OSS evaluation/test | Preserves the multi-turn conversation and physician-written rubrics |

The builder samples once with seed `42`. Every future model will use this same local artifact; no model-specific sample is generated.

The current local benchmark ID is:

```text
ed3bd3f03b1844570edfd90b8e04504c40ffe31c20afac69597644295ed04904
```

## Sources

- MedQA: [`awinml/medqa`](https://huggingface.co/datasets/awinml/medqa), a pinned mirror of [`jind11/MedQA`](https://github.com/jind11/MedQA), revision `afdb980b627374a7f10a0a11ed68455596c52742`.
- MedMCQA: [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa), from [`medmcqa/medmcqa`](https://github.com/medmcqa/medmcqa), revision `91c6572c454088bf71b679ad90aa8dffcd0d5868`.
- HealthBench: [`openai/healthbench`](https://huggingface.co/datasets/openai/healthbench), with its reference implementation in [`openai/simple-evals`](https://github.com/openai/simple-evals), revision `40ee1968852fc57f625934251ac22be47077a8fb`.

Review each dataset's license and distribution guidance before publishing data. HealthBench requests that benchmark examples not be revealed online. Consequently, raw data and frozen benchmark exports are ignored by Git and should remain local.

## Requirements

Python 3.9 or newer. The builder uses only the Python standard library, so no package installation is required.

`config.yaml` uses JSON syntax, which is valid YAML and can be parsed without a YAML dependency.

## Build

```bash
python -m src.build_benchmark
```

On the first run, the builder:

1. Downloads each pinned evaluation source into a revision-named file under `data/raw/`.
2. Normalizes MedQA and MedMCQA into four-choice cases with gold labels.
3. Preserves HealthBench conversations and rubric criteria separately from objective cases.
4. Removes duplicate normalized questions within and across datasets.
5. Applies a deterministic per-dataset shuffle derived from seed `42`.
6. Selects exactly 30 + 10 + 20 cases and assigns stable local IDs.
7. Atomically writes `data/benchmark.json`, `data/benchmark.csv`, and `data/benchmark_manifest.json`.

Raw online-cache filenames must contain the configured revision prefix. This prevents a config update from silently reusing a stale source file.

If `data/benchmark.json` already exists, the command validates its schemas, counts, uniqueness, and benchmark digest, then exits without downloading or resampling.

Build from previously downloaded raw files without network access:

```bash
python -m src.build_benchmark --offline
```

An explicit rebuild is available for development:

```bash
python -m src.build_benchmark --force
```

Do not use `--force` after model evaluation begins. Later result artifacts must be tied to the frozen benchmark ID.

## Local question viewer

Start the read-only local frontend:

```bash
python -m src.serve_benchmark
```

Then open `http://127.0.0.1:8765`. The Results page opens on the combined adjusted 77-question reporting view, built from all 80 objective questions across both rounds after three documented exclusions. Round 1 and Round 2 remain available as separate audit views. Questions are kept off the results pages and live in a dedicated Question Bank at `http://127.0.0.1:8765/questions`, with round, search, dataset, category, and model-response controls above a bounded scroll area. Objective keys remain hidden by default, and Round 1 HealthBench conversations retain expandable rubrics. The site binds only to localhost, disables caching, and makes no model or external API calls.

## Evaluation protocol rule

Every model evaluation is closed-book. A model must answer using only its own internal knowledge and reasoning. Internet access, browsing, web search, external retrieval/research/reference tools, opening links, and consulting outside sources are prohibited for the entire run.

GPT and Claude receive blind prompts without answer keys or scoring rubrics. Their isolated CLI sessions disable browsing and external tools; GPT also explicitly disables plugins, and Claude starts in safe mode with an empty MCP configuration. The event auditors reject tool activity and unrecognized event types. Claude additionally rejects nonzero provider-reported search/fetch usage, even when no tool call appears in the content. Its internal `StructuredOutput` formatter is permitted only to serialize the answer; it does not retrieve information.

Resuming a saved GPT or Claude response requires its original event log to pass the audit again and contain the same answer. A saved `closed_book_valid` flag alone is insufficient. These checks also apply to MedXpertQA runs. The curated MedXpertQA importer and GPT/Claude A–J schemas, validation, and scoring are implemented.

Question selection and model evaluation use separate contexts. GPT runs start fresh without resuming or forking this project's planning conversation; local memory use/generation and `AGENTS.md` injection are explicitly disabled. Claude's safe mode disables auto memory and custom instructions. Selection notes, solutions, and difficulty rationales must stay out of the blind prompt. Isolation prevents selection notes from leaking into an evaluation, but selecting questions with an AI model can still bias which abilities the subset tests.

Closed-book means no external lookup during evaluation. The hosted model service still needs a connection to generate responses. This protocol cannot establish whether a public benchmark question appeared in a model's training data or distinguish reasoning from memorized knowledge.

This rule is stored structurally in `config.yaml`, `data/benchmark.json`, and `data/benchmark_manifest.json`, and is displayed in the local viewer. Future runners must disable network and external research tools rather than relying only on prompt instructions.

The question sample has its own stable `benchmark_id`. The benchmark plus evaluation rules has a separate `protocol_id`, so rule changes are detectable without resampling or changing the 60 selected cases.

## Artifacts

### MedXpertQA Text: Curated Hard 100

The separate selection in `selections/medxpertqa-hard100-v1.json` freezes 100 original ten-option MCQs from [MedXpertQA Text](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA), revision `7e7c465a68eb2b866926bfa59c8c9d17a8daba65`. All 2,450 rows were screened; 288 shortlisted candidates received AI content review. Selection prioritizes integration of clinical evidence, treatment constraints, diagnostic sequencing, and apparent item quality. Screening scores only prioritize review; they are not measured difficulty scores. The manifest records retained IDs, individual rationales, exclusions, and limitations.

This is an **AI-curated hard subset**, not a proven ranking of the hardest 100 questions. No target-model results or answer keys were used to select items. Original keys were then copied unchanged, without new clinician validation or a comprehensive update against 2026 guidelines. The subset measures exam-style MCQ performance, not patient-facing safety or real-world clinical outcomes.

Rebuild the same frozen selection from the pinned local source file:

```bash
python3 -m src.build_medxpert_subset
```

If the source is absent, download `Text/test.jsonl` from the pinned revision into `data/raw/medxpertqa_text_test_7e7c465a.jsonl`. The builder requires SHA-256 `07ea977f7e326909b51c883b65f6ad7f399fc6db07408ef3630f4e3048ec71b2` and refuses to overwrite a different frozen set.

Local outputs in `data/medxpert-hard100/`:

- `questions.md` and `questions.json`: question-only packets, without keys, source IDs, or selection notes.
- `answer_key.json`: separate original answer key.
- `benchmark.json`: canonical scoring artifact with original labels and provenance.
- `selected_source.jsonl`: complete original selected source rows, including the original combined question/choices field.
- `selection_report.md`: coverage and per-item selection rationales; keep out of evaluation inputs.
- `manifest.json` and `screening.json`: source, selection, protocol, and export integrity records.

The stem is split from the original `Answer Choices:` suffix, with only trailing stem whitespace trimmed. Choices, their order, and labels are preserved. Presentation order is deterministic and does not imply difficulty rank. The source body-system/task labels are preserved as supplied; they are not newly adjudicated clinical categories.

The existing viewer and historical combined scores continue to show Round 1 and Round 2; this new set is available through the files above. The dedicated answer-only collector runs every question in a fresh isolated session, saves and audits each response immediately, and resumes only from verified event logs. It pins GPT-6 Astra and Claude Opus 5.5 at `xhigh` for two passes without loading the answer key or scoring during collection. See `ANSWER_COLLECTION.md` for the workflow and artifact layout. Gemini and legacy PDF exporters have not been extended for this subset.

`data/benchmark.json` contains:

- schema version, creation time, seed, dataset counts, and pinned source metadata;
- a SHA-256 benchmark ID over the canonical seed, sources, and cases;
- objective questions with choices and gold labels;
- HealthBench conversations and physician rubrics without pretending they are multiple-choice cases.

`data/benchmark.csv` is a flattened audit/export view. Conversation, rubric, and source metadata fields are serialized as JSON inside CSV cells.

`data/benchmark_manifest.json` records the benchmark ID, JSON and CSV file hashes, source revisions, seed, and counts. Each source record in the benchmark also includes the downloaded raw-file SHA-256 hash.

## Current evaluation targets

Updated October 21, 2026. `src/models.py` defines the active lineup shared by the runners and viewer:

| Model | Runner model ID | Round 1 result directory | Collection |
|---|---|---|---|
| GPT-6 Astra | `gpt-6-astra` | `results/gpt-6-astra/` | Codex, high reasoning |
| Claude Opus 5.5 | `claude-opus-5-5` | `results/claude-opus-5-5/` | Claude Code, high effort |
| Gemini 4 Argon | `gemini-4-argon-high` | `results/gemini-4-argon/` | Antigravity, high reasoning |
| Grok 4.8 | `grok-4.8` (requested target, unverified) | `results/grok-4.8-manual/` | Manual consumer-web collection |

The dashboard displays these targets as awaiting results until fresh two-pass artifacts exist. A combined score requires both rounds for the same model. Completed current models are selected ahead of historical versions. Existing GPT-5.6 Sol, Claude Fable 5, Gemini 3.6 Flash, and Grok 4.5 scores remain historical records; they are never renamed or reused as scores for the new lineup. Updating the configuration does not run an evaluation.

Verified references: [OpenAI GPT-6 migration](https://developers.openai.com/api/docs/guides/latest-model), [Anthropic model IDs](https://platform.claude.com/docs/en/models/overview), and [Antigravity headless model IDs](https://antigravity.google/docs/cli/headless/). The Gemini API uses `gemini-4-argon`; this project uses Antigravity's effort-qualified CLI ID instead. The [public xAI catalog](https://docs.x.ai/developers/models) lists Grok 4.7 as of the update date, so Grok 4.8 availability remains unverified. No automatic substitution is made.

## GPT-6 Astra evaluation

The runner sends the same blind 60-case prompt through two independent ephemeral sessions:

```bash
python -m src.run_gpt_benchmark
```

It requires `codex login status` to report ChatGPT authentication, deliberately rejecting API-key authentication. The model runs in an isolated temporary directory with user config, project rules, shell, apps, agents, browser/computer use, MCP integrations, image tools, and web search disabled. JSON event logs are audited, and any event other than reasoning or the final agent message invalidates the run.

The model receives all 40 objective questions and all 20 HealthBench conversations, but never receives objective answer keys, HealthBench rubrics, or source metadata. After both runs finish, the local scorer writes aggregate metrics and per-question answer/correctness flags under `results/gpt-6-astra/`. HealthBench responses are preserved but are not assigned objective accuracy or exact-text consistency metrics.

### Objective Round 2

Round 2 is a separate 40-question objective benchmark with 30 MedQA and 10 MedMCQA cases. Its builder excludes every Round 1 source ID and normalized question, rejects prompts that explicitly depend on a missing arrow or figure, and uses only the already-pinned local raw datasets:

```bash
python -m src.build_objective_round --offline
python -m src.run_gpt_benchmark \
  --benchmark data/round-2/benchmark.json \
  --output-dir results/round-2/gpt-6-astra
```

### Archived Round 2 outcomes

GPT-5.6 Sol at high reasoning completed both closed-book passes with zero forbidden tool events. It scored 37/40 (92.5%) on Pass 1 and 38/40 (95.0%) on Pass 2, with 39/40 (97.5%) exact answer consistency and 37/40 (92.5%) correct in both passes.

Claude Fable 5 at high reasoning also completed both passes through the Claude.ai subscription runner. The audits reported no external tools, zero web-search or web-fetch requests, and zero forbidden tool events. Fable scored 38/40 (95.0%) on Pass 1 and 37/40 (92.5%) on Pass 2, with 39/40 (97.5%) exact answer consistency and 37/40 (92.5%) correct in both passes.

Gemini 3.6 Flash at high reasoning completed both passes through Google consumer OAuth. The audits verified the exact `gemini-3.6-flash-high` model, separate conversation IDs, an active hard tool gate, zero external-tool attempts, and zero forbidden tool events. Gemini scored 37/40 (92.5%) on Pass 1 and 35/40 (87.5%) on Pass 2, with 38/40 (95.0%) exact answer consistency and 35/40 (87.5%) correct in both passes.

Grok 4.5 completed both passes manually through the Grok Free consumer web interface. The user-provided response screenshot was transcribed into the project and is explicitly labeled as lacking an automated browser/tool-event audit. Grok scored 33/40 (82.5%) on Pass 1 and 30/40 (75.0%) on Pass 2, with 33/40 (82.5%) exact answer consistency and 29/40 (72.5%) correct in both passes.

Round 2 contains no HealthBench cases. All four historical models completed the same frozen Round 2 sample. The dashboard exposes it independently and as the second half of the combined objective comparison.

## Claude Opus 5.5 evaluation

Run the 40 objective cases twice with Claude Opus 5.5 at high effort:

```bash
python -m src.run_claude_benchmark
```

This runner requires a Claude.ai subscription login and rejects API-billed authentication. It uses isolated temporary directories, disables all tools and customizations, supplies an empty strict MCP configuration, disables Chrome, and audits Claude's initialization stream to require an empty tool list before local scoring.

## Gemini 4 Argon evaluation

Run the 40 objective cases twice with Gemini 4 Argon at high reasoning:

```bash
python -m src.run_gemini_benchmark
```

This runner uses Google consumer OAuth through the Antigravity CLI, rejects API-key and Vertex environment routes, and refuses optional G1 credit fallback. Each pass uses a new isolated workspace and conversation. Antigravity advertises its standard tool catalog to the model, so the isolated workspace installs a `PreToolUse` safety hook that hard-denies every tool except the harness's internal structured-output completion action. Slash-command expansion is disabled and the workspace MCP configuration is empty. A separate `PreInvocation` audit proves that the gate was loaded; any attempted external tool or model substitution invalidates the run before the local scorer may use the answer key.

For Round 2, use the same frozen benchmark path and model-specific output folders:

```bash
python -m src.run_claude_benchmark \
  --benchmark data/round-2/benchmark.json \
  --output-dir results/round-2/claude-opus-5-5
python -m src.run_gemini_benchmark \
  --benchmark data/round-2/benchmark.json \
  --output-dir results/round-2/gemini-4-argon
```

The Antigravity CLI (`agy`) must be installed and signed in; the runner verifies the exact model is listed for the account before generation.

## Grok 4.8 evaluation

Grok remains a manual workflow. Confirm the exact model and mode in the consumer interface before collecting two independent closed-book passes. Save newly scored artifacts in `results/grok-4.8-manual/` (Round 1) or `results/round-2/grok-4.8-manual/` (Round 2), using the existing manual archive format: `summary.json` and `objective_results.csv`. The summary must identify `model: "grok-4.8"`, `run_count: 2`, and the frozen benchmark/protocol IDs. No Grok 4.8 results or automated tool audit are claimed by this update.

## Historical results and reporting

The Round 1 record for qID 35 (`medmcqa_005`) refers to an arrow-marked muscle but contains no source image, so its adjusted view contains 39 questions. Round 2 excludes qID 34 (`medmcqa_004`) because its oxygen-device setup is underspecified and qID 38 (`medmcqa_008`) because its source explanation contradicts the single keyed answer. In the combined sequence these are qIDs 35, 74, and 78. The frontend therefore defaults to the combined adjusted 77-question view, while the raw 80-question results remain available for auditability.

The historical results compare GPT-5.6 Sol, Claude Fable 5, Gemini 3.6 Flash, and Grok 4.5 in a primary combined score and in separate Round 1 and Round 2 drill-downs. Select a model row to inspect per-question answers, correctness, confidence when available, pass-to-pass consistency, source round, and exclusion notes.

The Round 1 user-maintained response sheet is preserved verbatim at `results/medical_ai_benchmark_responses.csv`. The viewer validates its gold labels and cross-checks every response against the per-model archives. Round 2 is calculated directly from its four validated per-model archives. Each round exposes a source SHA-256 through `/api/results` for traceability.

Two responsive Plotly bar charts summarize pass accuracy, exact answer consistency, and the rate of questions answered correctly in both passes. They follow each round's raw/adjusted control and dynamically expand the y-axis while retaining complete bars. Plotly.js Basic 3.7.0 is vendored under `viewer/vendor/`, so the charts do not require a CDN or internet connection at runtime.

## Reproducibility and safety checks

- Fixed seed and revision-pinned inputs.
- Versioned closed-book protocol: internal reasoning only, with external research disabled.
- One sample shared by every future model.
- Stable IDs based on sorted sampled source IDs.
- Required A–D choices and gold labels for objective cases.
- Required conversations and rubrics for HealthBench cases.
- No duplicate benchmark IDs, source IDs, or normalized questions.
- Exact configured case counts.
- Atomic writes to avoid partial benchmark files.
- Integrity verification on every later load.
- No model runner sends an answer key, rubric, or source metadata to the evaluated model.
- Model-run event logs are rejected if external tools are available or used.

## Tests

Tests use small synthetic JSONL fixtures and do not download or print real benchmark examples:

```bash
python -m unittest discover -s tests -v
```

On macOS system Python in a restricted workspace, redirect its external bytecode cache if needed:

```bash
PYTHONPYCACHEPREFIX=/tmp/medical_benchmark_pycache \
  python -m unittest discover -s tests -v
```

The suite verifies exact counts and exports, no-op frozen rebuilds, deterministic forced rebuilds, tamper detection, normalized objective/HealthBench schemas, blind-prompt construction, exact model-response coverage, closed-book event auditing, and two-run scoring.

## Project layout

```text
config.yaml
data/
  raw/
  benchmark.json             # local, ignored
  benchmark.csv              # local, ignored
  benchmark_manifest.json    # local, ignored
src/
  benchmark.py               # load and integrity verification
  build_benchmark.py         # CLI and deterministic freeze
  serve_benchmark.py         # localhost-only question viewer server
  run_gpt_benchmark.py       # isolated closed-book GPT-6 Astra runner/scorer
  run_claude_benchmark.py    # isolated closed-book Claude Opus 5.5 runner/scorer
  run_gemini_benchmark.py    # isolated closed-book Gemini 4 Argon runner/scorer
  models.py                  # active model IDs and result directories
  config.py
  io_utils.py                # downloads, hashes, atomic JSON/CSV writes
  schemas.py
  datasets/
    common.py
    medqa.py
    medmcqa.py
    healthbench.py
tests/
  fixtures/                  # synthetic only
  test_pipeline.py
viewer/
  index.html
  app.js
  styles.css
```

## License

Project code is MIT licensed. Dataset content remains subject to its source license and distribution guidance.
