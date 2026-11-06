# Frozen MedXpertQA answer collection

Run the dedicated collector from the repository root:

```sh
python3 -m src.collect_medxpert_answers
```

The only benchmark input is `data/medxpert-hard100/questions.json`. The collector
does not import the legacy benchmark loaders, schemas, selection code, or scorers.
It rejects any other packet path and unexpected fields. Question text and A–J
choice strings are preserved when the single question is serialized into a prompt.

Targets are pinned to `gpt-6-astra` and `claude-opus-5-5`, both at `xhigh`, with
two passes each. The local subscription logins, model settings, and a synthetic
nonclinical response are checked before benchmark collection. No fallback model
is configured. API-key authentication is rejected. The Claude CLI requires access
to the user's Keychain. Normal subscription limits still apply.

Each attempt starts a new CLI process in a fresh temporary directory. Claude
safe mode and an empty settings source disable project and user customizations;
its tool inventory must be empty. Codex ignores user configuration and project
instructions, disables memory, skills, apps, plugins, agent delegation, browsing,
shell, and code execution, and uses a minimal respondent instruction. Both clients
run ephemeral sessions and receive no previous question, answer, correction, hint,
reference answer, or feedback. One worker runs each model/pass, with independent
fresh sessions for each question.

The model returns only the original question ID, one selected label, and a brief
clinical explanation. Validation checks structure, completion, identity, and tool
events. It never checks the selected answer against a reference.

Artifacts are written under `results/medxpert-hard100-answer-only/`:

- Four `*_pass_N_answers.json` files contain answers, explanations, completion
  status, and audit metadata.
- `comparison.csv` and `comparison.json` place the four responses side by side.
- `completion.json` records completed responses and verified tool audits per pass.
- `manifest.json` pins the packet hash and prompt protocol.
- `preflight.json` records subscription and exact model checks.
- `<model>/pass_N/<question-id>/attempt-NNNN/` retains each prompt, invocation,
  streamed events, stderr, process/exit metadata, response, and audit. Failed and
  interrupted attempts are retained in their original directories.

Every event line is flushed and synced as it arrives. Completed responses and
their audit are written atomically before progress is reported. Exports are
refreshed after each completion. On restart, the collector re-audits raw events,
checks the prompt/model/effort metadata, recovers a complete stream if the final
response file was interrupted, and skips verified answers. Conflicting saved
responses cause a stop. Failed attempts receive a new numbered attempt directory.
The export checks that no completed sessions were reused between questions or
passes. A file lock prevents simultaneous collector processes in one output folder.

Claude's initialization and assistant response identify the returned model and
its empty tool inventory; provider usage fields are also audited for search/fetch
calls. Codex CLI's JSON event format does not expose a full tool inventory or
response model field. Its audit records the exact CLI model, supported catalog
settings, disabled capability configuration, and complete event stream. Its
startup notice that Code Mode fails closed is retained in the audit. The audit
does not claim visibility into provider-internal operations.

Useful commands:

```sh
python3 -m unittest tests.test_answer_collection -v
python3 -m src.collect_medxpert_answers --preflight-only
python3 -m src.collect_medxpert_answers --export-only
python3 -m src.run_gpt_benchmark --answer-only
python3 -m src.run_claude_benchmark --answer-only
```

The old runner entry points route `--answer-only` to this collector. Their legacy
scoring path rejects `medxpert-hard100` inputs. Existing unrelated historical
results and workflows are preserved. Do not run legacy dataset tests or viewer
scoring code as part of this collection.
