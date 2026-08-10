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

This rule is stored structurally in `config.yaml`, `data/benchmark.json`, and `data/benchmark_manifest.json`, and is displayed in the local viewer. Future runners must disable network and external research tools rather than relying only on prompt instructions.

The question sample has its own stable `benchmark_id`. The benchmark plus evaluation rules has a separate `protocol_id`, so rule changes are detectable without resampling or changing the 60 selected cases.

## Artifacts

`data/benchmark.json` contains:

- schema version, creation time, seed, dataset counts, and pinned source metadata;
- a SHA-256 benchmark ID over the canonical seed, sources, and cases;
- objective questions with choices and gold labels;
- HealthBench conversations and physician rubrics without pretending they are multiple-choice cases.

`data/benchmark.csv` is a flattened audit/export view. Conversation, rubric, and source metadata fields are serialized as JSON inside CSV cells.

`data/benchmark_manifest.json` records the benchmark ID, JSON and CSV file hashes, source revisions, seed, and counts. Each source record in the benchmark also includes the downloaded raw-file SHA-256 hash.

## GPT-5.6 Sol evaluation

The runner sends the same blind 60-case prompt through two independent ephemeral sessions:

```bash
python -m src.run_gpt_benchmark
```

It requires `codex login status` to report ChatGPT authentication, deliberately rejecting API-key authentication. The model runs in an isolated temporary directory with user config, project rules, shell, apps, agents, browser/computer use, MCP integrations, image tools, and web search disabled. JSON event logs are audited, and any event other than reasoning or the final agent message invalidates the run.

The model receives all 40 objective questions and all 20 HealthBench conversations, but never receives objective answer keys, HealthBench rubrics, or source metadata. After both runs finish, the local scorer writes aggregate metrics and per-question answer/correctness flags under `results/gpt-5.6-sol/`. HealthBench responses are preserved but are not assigned objective accuracy or exact-text consistency metrics.

### Objective Round 2

Round 2 is a separate 40-question objective benchmark with 30 MedQA and 10 MedMCQA cases. Its builder excludes every Round 1 source ID and normalized question, rejects prompts that explicitly depend on a missing arrow or figure, and uses only the already-pinned local raw datasets:

```bash
python -m src.build_objective_round --offline
python -m src.run_gpt_benchmark \
  --benchmark data/round-2/benchmark.json \
  --output-dir results/round-2/gpt-5.6-sol
```

GPT-5.6 Sol at high reasoning completed both closed-book passes with zero forbidden tool events. It scored 37/40 (92.5%) on Pass 1 and 38/40 (95.0%) on Pass 2, with 39/40 (97.5%) exact answer consistency and 37/40 (92.5%) correct in both passes.

Claude Fable 5 at high reasoning also completed both passes through the Claude.ai subscription runner. The audits reported no external tools, zero web-search or web-fetch requests, and zero forbidden tool events. Fable scored 38/40 (95.0%) on Pass 1 and 37/40 (92.5%) on Pass 2, with 39/40 (97.5%) exact answer consistency and 37/40 (92.5%) correct in both passes.

Gemini 3.6 Flash at high reasoning completed both passes through Google consumer OAuth. The audits verified the exact `gemini-3.6-flash-high` model, separate conversation IDs, an active hard tool gate, zero external-tool attempts, and zero forbidden tool events. Gemini scored 37/40 (92.5%) on Pass 1 and 35/40 (87.5%) on Pass 2, with 38/40 (95.0%) exact answer consistency and 35/40 (87.5%) correct in both passes.

Grok 4.5 completed both passes manually through the Grok Free consumer web interface. The user-provided response screenshot was transcribed into the project and is explicitly labeled as lacking an automated browser/tool-event audit. Grok scored 33/40 (82.5%) on Pass 1 and 30/40 (75.0%) on Pass 2, with 33/40 (82.5%) exact answer consistency and 29/40 (72.5%) correct in both passes.

Round 2 contains no HealthBench cases. All four models have completed the same frozen Round 2 sample. The dashboard exposes it independently and as the second half of the combined objective comparison.

## Claude Fable 5 evaluation

Run the 40 objective cases twice with Claude Fable 5 at high effort:

```bash
python -m src.run_claude_benchmark
```

This runner requires a Claude.ai subscription login and rejects API-billed authentication. It uses isolated temporary directories, disables all tools and customizations, supplies an empty strict MCP configuration, disables Chrome, and audits Claude's initialization stream to require an empty tool list before local scoring.

## Gemini 3.6 Flash evaluation

Run the 40 objective cases twice with Gemini 3.6 Flash at high reasoning:

```bash
python -m src.run_gemini_benchmark
```

This runner uses Google consumer OAuth through the Antigravity CLI, rejects API-key and Vertex environment routes, and refuses optional G1 credit fallback. Each pass uses a new isolated workspace and conversation. Antigravity advertises its standard tool catalog to the model, so the isolated workspace installs a `PreToolUse` safety hook that hard-denies every tool except the harness's internal structured-output completion action. Slash-command expansion is disabled and the workspace MCP configuration is empty. A separate `PreInvocation` audit proves that the gate was loaded; any attempted external tool or model substitution invalidates the run before the local scorer may use the answer key.

## Grok 4.5 evaluation


The Round 1 record for qID 35 (`medmcqa_005`) refers to an arrow-marked muscle but contains no source image, so its adjusted view contains 39 questions. Round 2 excludes qID 34 (`medmcqa_004`) because its oxygen-device setup is underspecified and qID 38 (`medmcqa_008`) because its source explanation contradicts the single keyed answer. In the combined sequence these are qIDs 35, 74, and 78. The frontend therefore defaults to the combined adjusted 77-question view, while the raw 80-question results remain available for auditability.

The local viewer compares GPT-5.6 Sol, Claude Fable 5, Gemini 3.6 Flash, and Grok 4.5 in a primary combined score and in separate Round 1 and Round 2 drill-downs. Select a model row to inspect per-question answers, correctness, confidence when available, pass-to-pass consistency, source round, and exclusion notes.

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
  run_gpt_benchmark.py       # isolated closed-book GPT-5.6 Sol runner/scorer
  run_claude_benchmark.py    # isolated closed-book Claude Fable 5 runner/scorer
  run_gemini_benchmark.py    # isolated closed-book Gemini 3.6 Flash runner/scorer
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
