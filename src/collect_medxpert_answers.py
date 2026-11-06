"""Isolated, subscription-only answer collection. No scoring dependencies.

Only questions.json is read. Every attempt has a fresh process and working
directory; durable event logs are the source of truth for resumable exports.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "data/medxpert-hard100/questions.json"
DEFAULT_OUTPUT = ROOT / "results/medxpert-hard100-answer-only"
MODELS = {"gpt": "gpt-6-astra", "claude": "claude-opus-5-5"}
EFFORT = "xhigh"
PROTOCOL = "medxpert-single-question-answer-only-v1"
SYSTEM = "You are a closed-book medical question respondent. Use only internal knowledge and reasoning. Do not invoke tools or consult external information."
INSTRUCTIONS = """Answer this one medical multiple-choice question using only your internal knowledge and reasoning.
Do not use tools, browsing, search, retrieval, files, memory, other agents, or external references.
Return only a JSON object with exactly these fields:
- id: the original question ID
- answer: one selected label from the supplied A–J choices
- explanation: a brief clinical explanation (one to three sentences)
Do not discuss the evaluation protocol.

QUESTION:
"""
SCHEMA = {"type": "object", "properties": {
    "id": {"type": "string"},
    "answer": {"type": "string", "enum": list("ABCDEFGHIJ")},
    "explanation": {"type": "string"}},
    "required": ["id", "answer", "explanation"], "additionalProperties": False}


class CollectionError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def durable(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save(path, value):
    durable(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_packet(path=PACKET):
    path = Path(path).resolve()
    if path != PACKET.resolve():
        raise CollectionError("Only the frozen data/medxpert-hard100/questions.json packet is permitted")
    raw = path.read_bytes()
    packet = json.loads(raw)
    if set(packet) != {"questions"} or len(packet["questions"]) != 100:
        raise CollectionError("Expected a 100-question packet")
    ids = set()
    for q in packet["questions"]:
        if set(q) != {"id", "question", "choices"}:
            raise CollectionError("Unexpected packet fields")
        if not isinstance(q["id"], str) or not re.fullmatch(r"MXH-\d{3}", q["id"]) or q["id"] in ids:
            raise CollectionError("Invalid or duplicate question ID")
        ids.add(q["id"])
        if set(q["choices"]) != set("ABCDEFGHIJ") or not isinstance(q["question"], str):
            raise CollectionError("Question or A–J choices missing")
        if not all(isinstance(v, str) for v in q["choices"].values()):
            raise CollectionError("Choices must be original strings")
    return packet["questions"], digest(raw)


def prompt_for(question):
    return INSTRUCTIONS + json.dumps(question, ensure_ascii=False, indent=2)


def environment():
    # Do not inherit model, project, session, provider, API-key or effort overrides.
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ("ANTHROPIC_", "OPENAI_", "CLAUDE_", "CLAUDECODE", "CODEX_"))}
    env.update({"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1"})
    return env


def command_for(provider, workspace):
    if provider == "claude":
        return [shutil.which("claude") or "claude", "--print", "--model", MODELS[provider],
                "--effort", EFFORT, "--tools", "", "--safe-mode", "--setting-sources", "",
                "--settings", json.dumps({"enabledPlugins": {
                    "cc-plugin-agents-md@builtin": False,
                    "cc-plugin-plugin-authoring@builtin": False,
                    "cc-plugin-telemetry@builtin": False},
                    "autoMemoryEnabled": False, "disableAllHooks": True}),
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--no-chrome", "--disable-slash-commands", "--no-session-persistence",
                "--session-id", str(uuid.uuid4()), "--max-turns", "1",
                "--prompt-suggestions", "false", "--system-prompt", SYSTEM,
                "--output-format", "stream-json", "--verbose"]
    save(workspace / "response-schema.json", SCHEMA)
    durable(workspace / "instructions.txt", SYSTEM)
    cmd = [shutil.which("codex") or "codex", "exec", "--model", MODELS[provider],
           "--sandbox", "read-only", "--ephemeral", "--skip-git-repo-check",
           "--ignore-user-config", "--ignore-rules", "--json",
           "--output-schema", str(workspace / "response-schema.json")]
    config = {"forced_login_method": "chatgpt", "model_reasoning_effort": EFFORT,
              "approval_policy": "never", "project_doc_max_bytes": 0,
              "web_search": "disabled", "tools.web_search": False,
              "tools.update_plan.enabled": False, "tools.experimental_request_user_input.enabled": False,
              "agents.enabled": False, "memories.use_memories": False,
              "memories.generate_memories": False,
              "cloud.skills.enabled": False, "skills.include_instructions": False,
              "model_instructions_file": str(workspace / "instructions.txt"),
              "include_apps_instructions": False,
              "include_collaboration_mode_instructions": False,
              "include_environment_context": False}
    for name in ("memories", "shell_tool", "unified_exec", "apps", "plugins", "hooks",
                 "computer_use", "browser_use", "browser_use_external", "in_app_browser",
                 "view_image", "image_generation", "multi_agent", "multi_agent_v2", "goals",
                 "code_mode", "code_mode_host", "code_mode_only", "skill_search",
                 "skill_mcp_dependency_install", "tool_suggest", "sleep_tool",
                 "shell_snapshot", "workspace_dependencies", "remote_plugin",
                 "in_app_local_automation", "realtime_conversation"):
        config["features." + name] = False
    config["features.skip_host_skill_discovery"] = True
    for key, value in config.items():
        cmd += ["--config", key + "=" + json.dumps(value)]
    return cmd + ["-"]


def validate_answer(response, question):
    if not isinstance(response, dict) or set(response) != {"id", "answer", "explanation"}:
        raise CollectionError("Response must contain only id, answer and explanation")
    if response["id"] != question["id"] or response["answer"] not in question["choices"]:
        raise CollectionError("Response ID or selected label is invalid")
    if not isinstance(response["explanation"], str) or not response["explanation"].strip():
        raise CollectionError("A brief explanation is required")
    return response


def audit_stream(provider, raw, question):
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not events or any(not isinstance(e, dict) for e in events):
        raise CollectionError("Missing or malformed event stream")
    sessions, answers = set(), []
    complete = False
    types = {}
    initialized = False
    startup_notices = []
    turn_started = False
    for e in events:
        t = e.get("type")
        types[t] = types.get(t, 0) + 1
        if provider == "gpt":
            if t not in {"thread.started", "turn.started", "turn.completed", "turn.failed",
                         "error", "item.started", "item.updated", "item.completed"}:
                raise CollectionError("Unaudited GPT event: " + str(t))
            if t == "thread.started":
                sessions.add(e.get("thread_id"))
            if t == "turn.started":
                turn_started = True
            if t == "turn.failed":
                raise CollectionError("GPT turn failed")
            if t == "turn.completed":
                complete = True
            if t.startswith("item."):
                item = e.get("item", {})
                if item.get("type") == "error" and not turn_started and (
                    item.get("message", "").startswith("Under-development features enabled: skip_host_skill_discovery.")
                    or item.get("message", "") == "Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed; enable `features.code_mode_host` and install `codex-code-mode-host`."
                ):
                    startup_notices.append(item["message"])
                    continue
                if item.get("type") not in {"reasoning", "agent_message"}:
                    raise CollectionError("Forbidden GPT tool event: " + str(item.get("type")))
                if t == "item.completed" and item["type"] == "agent_message":
                    answers.append(item["text"])
        else:
            if t not in {"system", "assistant", "result", "rate_limit_event"}:
                raise CollectionError("Unaudited Claude event: " + str(t))
            if e.get("session_id"):
                sessions.add(e["session_id"])
            if t == "system":
                if e.get("subtype") == "thinking_tokens" and initialized:
                    if set(e) - {"type", "subtype", "estimated_tokens", "estimated_tokens_delta", "session_id", "uuid"}:
                        raise CollectionError("Unknown fields in Claude reasoning-token notification")
                    if not all(isinstance(e.get(k), int) and e[k] >= 0 for k in ("estimated_tokens", "estimated_tokens_delta")):
                        raise CollectionError("Malformed Claude reasoning-token notification")
                    continue
                if e.get("subtype") != "init" or initialized:
                    raise CollectionError("Unexpected Claude initialization")
                initialized = True
                if e.get("tools") != [] or e.get("mcp_servers") or e.get("skills") or e.get("plugins"):
                    raise CollectionError("Claude tools or customizations were enabled")
                if e.get("model") != MODELS[provider]:
                    raise CollectionError("Claude exact model mismatch: " + str(e.get("model")))
            msg = e.get("message", {})
            if t == "assistant":
                if msg.get("model") != MODELS[provider]:
                    raise CollectionError("Claude response model mismatch")
                for block in msg.get("content", []):
                    if block.get("type") not in {"text", "thinking", "redacted_thinking"}:
                        raise CollectionError("Forbidden Claude tool block: " + str(block.get("type")))
            usages = [e.get("usage", {}), msg.get("usage", {})]
            usages += list(e.get("modelUsage", {}).values())
            for usage in usages:
                for key in ("web_search_requests", "web_fetch_requests", "webSearchRequests", "webFetchRequests"):
                    if usage.get(key, 0):
                        raise CollectionError("Provider reported external tool usage")
                if any(usage.get("server_tool_use", {}).values()):
                    raise CollectionError("Provider reported server tool usage")
            if t == "result":
                if e.get("is_error") or e.get("subtype") != "success":
                    raise CollectionError("Claude result failed: " + str(e.get("result", ""))[:300])
                complete = True
                answers.append(e.get("result", ""))
    if not complete or len(sessions) != 1 or None in sessions or len(answers) != 1:
        raise CollectionError("Expected one complete fresh session and one final response")
    if provider == "claude" and not initialized:
        raise CollectionError("Missing Claude tool inventory")
    value = answers[0].strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # Preserve the full response in events; extract only an unambiguous JSON
        # answer if a client also emitted prose or a Markdown fence. No model retry.
        candidates = []
        decoder = json.JSONDecoder()
        for match in re.finditer(r"(?m)^\s*\{", value):
            try:
                obj, _ = decoder.raw_decode(value[match.start():].lstrip())
                validate_answer(obj, question)
                candidates.append(obj)
            except (ValueError, CollectionError):
                pass
        if len(candidates) != 1:
            raise CollectionError("Expected exactly one unambiguous JSON answer in the saved response")
        parsed = candidates[0]
    answer = validate_answer(parsed, question)
    return answer, {"status": "verified", "session_id": next(iter(sessions)),
                    "forbidden_tool_events": 0, "event_types": types,
                    "startup_notices": startup_notices,
                    "external_tools_configured": False,
                    "tool_inventory_evidence": "empty initialization inventory" if provider == "claude" else "disabled CLI configuration; event stream audited",
                    "model_evidence": "initialization and response model" if provider == "claude" else "exact CLI model; subscription catalog; no fallback configured",
                    "reasoning_effort": EFFORT}


def execute(provider, question, attempt, timeout):
    attempt.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="medxpert-isolated-") as temp:
        work = Path(temp)
        cmd = command_for(provider, work)
        prompt = prompt_for(question)
        durable(attempt / "prompt.txt", prompt)
        save(attempt / "request.json", {"protocol": PROTOCOL, "model": MODELS[provider],
             "reasoning_effort": EFFORT, "question_id": question["id"], "command": cmd,
             "cwd": str(work), "prompt_sha256": digest(prompt.encode()), "started_at": now(),
             "system_instructions": SYSTEM, "fresh_process": True, "history": False,
             "memory": False, "project_instructions": False, "external_tools": False})
        with (attempt / "events.jsonl").open("wb", buffering=0) as out, (attempt / "stderr.log").open("wb", buffering=0) as err:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                 cwd=work, env=environment(), start_new_session=True)
            save(attempt / "process.json", {"pid": p.pid, "started_at": now()})
            def pump():
                for line in iter(p.stdout.readline, b""):
                    out.write(line)
                    os.fsync(out.fileno())
            thread = threading.Thread(target=pump, daemon=True)
            thread.start()
            interrupted = None
            try:
                p.stdin.write(prompt.encode())
                p.stdin.close()
                code = p.wait(timeout=timeout)
            except BaseException as exc:
                interrupted = exc
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    code = p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    code = p.wait()
            thread.join(timeout=10)
            os.fsync(err.fileno())
            save(attempt / "exit.json", {"returncode": code, "ended_at": now(),
                                          "interruption": type(interrupted).__name__ if interrupted else None})
            if interrupted:
                raise interrupted
    if code:
        detail = (attempt / "stderr.log").read_text()[-1500:]
        for line in (attempt / "events.jsonl").read_text().splitlines():
            try:
                event = json.loads(line)
                if event.get("type") == "result" and event.get("is_error"):
                    detail = str(event.get("result", detail))[-1500:]
                elif event.get("type") == "error":
                    detail = str(event.get("message", detail))[-1500:]
            except (ValueError, AttributeError):
                pass
        raise CollectionError("Client exited %d: %s" % (code, detail))


def recover(provider, question, folder, packet_hash, pass_number):
    expected = {"protocol": PROTOCOL, "packet_sha256": packet_hash, "model": MODELS[provider],
                "reasoning_effort": EFFORT, "pass": pass_number,
                "prompt_sha256": digest(prompt_for(question).encode())}
    for attempt in sorted(folder.glob("attempt-*")):
        if not (attempt / "request.json").exists():
            continue
        request = read(attempt / "request.json")
        if request.get("prompt_sha256") != expected["prompt_sha256"] or request.get("model") != MODELS[provider] or request.get("reasoning_effort") != EFFORT:
            raise CollectionError("Historical attempt metadata mismatch: " + str(attempt))
        try:
            raw = (attempt / "events.jsonl").read_text()
            answer, audit = audit_stream(provider, raw, question)
        except (OSError, ValueError, CollectionError) as exc:
            if (attempt / "response.json").exists():
                raise CollectionError("Previously verified response no longer audits: " + str(attempt)) from exc
            if not (attempt / "audit.json").exists():
                save(attempt / "audit.json", {"status": "failed_or_interrupted", "detail": str(exc), "checked_at": now()})
            continue
        audit["events_sha256"] = digest(raw.encode())
        result = {**expected, **answer, "completion_status": "completed", "tool_audit_status": "verified",
                  "attempt_path": str(attempt.relative_to(folder.parents[2])), "tool_audit": audit}
        if (attempt / "response.json").exists():
            saved = read(attempt / "response.json")
            if saved != result:
                raise CollectionError("Saved response or audit differs from verified events: " + str(attempt))
        else:
            if (attempt / "audit.json").exists():
                save(attempt / "audit_history" / (uuid.uuid4().hex + ".json"), read(attempt / "audit.json"))
            save(attempt / "audit.json", audit)
            save(attempt / "response.json", result)
        return result
    return None


def collect_one(provider, question, output, packet_hash, pass_number, timeout, retries):
    folder = output / MODELS[provider] / ("pass_%d" % pass_number) / question["id"]
    found = recover(provider, question, folder, packet_hash, pass_number)
    if found:
        return found
    for _ in range(retries):
        number = len(list(folder.glob("attempt-*"))) + 1
        attempt = folder / ("attempt-%04d" % number)
        try:
            execute(provider, question, attempt, timeout)
            found = recover(provider, question, folder, packet_hash, pass_number)
            if found:
                return found
            detail = read(attempt / "audit.json").get("detail", "unknown event")
            raise CollectionError("Audit review required; completed client response preserved: " + detail)
        except (CollectionError, subprocess.TimeoutExpired) as exc:
            save(attempt / "failure.json", {"status": "failed", "detail": str(exc), "ended_at": now()})
            if not (attempt / "audit.json").exists():
                save(attempt / "audit.json", {"status": "failed", "detail": str(exc)})
            print(json.dumps({"model": MODELS[provider], "pass": pass_number, "id": question["id"],
                              "status": "attempt_failed", "detail": str(exc)}), flush=True)
            # Authentication and quota failures must not trigger a benchmark-wide retry storm.
            if any(s in str(exc).lower() for s in ("limit", "auth", "login", "not available", "not supported")):
                raise
            if "Audit review required" in str(exc):
                raise
    raise CollectionError("Attempts exhausted for %s %s pass %d" % (provider, question["id"], pass_number))


def export(output, questions, packet_hash):
    all_rows = {}
    sessions = set()
    summary = {"protocol": PROTOCOL, "packet_sha256": packet_hash, "updated_at": now(), "passes": {}}
    for provider, model in MODELS.items():
        for pass_number in (1, 2):
            rows = []
            for question in questions:
                folder = output / model / ("pass_%d" % pass_number) / question["id"]
                row = recover(provider, question, folder, packet_hash, pass_number)
                if row:
                    session = row["tool_audit"]["session_id"]
                    if session in sessions:
                        raise CollectionError("A session was reused across questions or passes")
                    sessions.add(session)
                else:
                    row = {"id": question["id"], "model": model, "pass": pass_number,
                           "answer": None, "explanation": None, "completion_status": "pending",
                           "tool_audit_status": "not_completed"}
                rows.append(row)
                all_rows[(question["id"], provider, pass_number)] = row
            save(output / (model + "_pass_%d_answers.json" % pass_number),
                 {"protocol": PROTOCOL, "packet_sha256": packet_hash, "model": model,
                  "reasoning_effort": EFFORT, "pass": pass_number, "answers": rows})
            summary["passes"][model + "/pass_%d" % pass_number] = {
                "completed": sum(r["completion_status"] == "completed" for r in rows), "total": len(questions),
                "tool_audits_verified": sum(r["tool_audit_status"] == "verified" for r in rows)}
    comparison = []
    for q in questions:
        row = {"question_id": q["id"]}
        for provider in MODELS:
            for n in (1, 2):
                r = all_rows[(q["id"], provider, n)]
                for field in ("answer", "explanation", "completion_status", "tool_audit_status"):
                    row["%s_pass_%d_%s" % (MODELS[provider], n, field)] = r[field]
        comparison.append(row)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(comparison[0]))
    writer.writeheader()
    writer.writerows(comparison)
    durable(output / "comparison.csv", text.getvalue())
    save(output / "comparison.json", comparison)
    save(output / "completion.json", summary)
    return summary


def preflight(output, timeout):
    output.mkdir(parents=True, exist_ok=True)
    env = environment()
    history_path = output / "preflight_history" / (uuid.uuid4().hex + ".json")
    evidence = {"checked_at": now(), "models": {}, "documentation": [
        "https://learn.chatgpt.com/docs/models",
        "https://platform.claude.com/docs/en/build-with-claude/effort"]}
    for provider in MODELS:
        cli = "codex" if provider == "gpt" else "claude"
        auth_cmd = [cli, "login", "status"] if provider == "gpt" else [cli, "auth", "status"]
        result = subprocess.run(auth_cmd, capture_output=True, text=True, env=env, timeout=30)
        if provider == "gpt":
            valid = result.returncode == 0 and "Logged in using ChatGPT" in result.stdout + result.stderr
            auth = {"method": "ChatGPT subscription", "verified": valid}
            catalog = read(Path.home() / ".codex/models_cache.json")
            entry = next((m for m in catalog["models"] if m.get("slug") == MODELS[provider]), {})
            levels = [x["effort"] for x in entry.get("supported_reasoning_levels", [])]
        else:
            status = json.loads(result.stdout)
            valid = result.returncode == 0 and status.get("loggedIn") and status.get("authMethod") == "claude.ai" and status.get("subscriptionType")
            auth = {k: status.get(k) for k in ("loggedIn", "authMethod", "subscriptionType")}
            help_text = subprocess.run([cli, "--help"], capture_output=True, text=True, env=env, timeout=30).stdout
            levels = ["low", "medium", "high", "xhigh", "max"] if "low, medium, high, xhigh, max" in help_text else []
        evidence["models"][provider] = {"model": MODELS[provider], "requested_effort": EFFORT,
            "available_settings": levels, "authentication": auth,
            "version": subprocess.run([cli, "--version"], capture_output=True, text=True, env=env, timeout=30).stdout.strip()}
        if not valid or EFFORT not in levels:
            save(output / "preflight.json", evidence)
            save(history_path, evidence)
            raise CollectionError("Exact target preflight failed; available settings: " + json.dumps(evidence))
    # Synthetic format check. It never enters the benchmark answer files.
    question = {"id": "SMOKE-ONLY", "question": "Select the label whose choice is the word amber. Explain briefly.",
                "choices": dict(zip("ABCDEFGHIJ", ["amber", "blue", "cyan", "denim", "ecru", "fawn", "green", "hazel", "indigo", "jade"]))}
    for provider in MODELS:
        attempt = output / "preflight" / provider / ("attempt-" + uuid.uuid4().hex)
        try:
            execute(provider, question, attempt, timeout)
            answer, audit = audit_stream(provider, (attempt / "events.jsonl").read_text(), question)
            save(attempt / "response.json", answer)
            save(attempt / "audit.json", audit)
            evidence["models"][provider]["smoke_status"] = "verified"
            evidence["models"][provider]["smoke_attempt"] = str(attempt)
        except BaseException as exc:
            save(attempt / "failure.json", {"detail": str(exc), "ended_at": now()})
            save(attempt / "audit.json", {"status": "failed", "detail": str(exc)})
            evidence["models"][provider]["smoke_status"] = "failed"
            save(output / "preflight.json", evidence)
            save(history_path, evidence)
            raise
    evidence["status"] = "verified"
    save(output / "preflight.json", evidence)
    save(history_path, evidence)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=PACKET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--provider", choices=list(MODELS))
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--attempts", type=int, default=2)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "collector.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        questions, packet_hash = load_packet(args.questions)
        manifest = {"protocol": PROTOCOL, "packet_path": str(PACKET), "packet_sha256": packet_hash,
                    "question_count": 100, "models": MODELS, "reasoning_effort": EFFORT,
                    "passes": 2, "question_per_session": 1, "prompt_instructions": INSTRUCTIONS,
                    "system_instructions": SYSTEM}
        if (output / "manifest.json").exists() and read(output / "manifest.json") != manifest:
            raise CollectionError("Existing collection manifest differs; preserve it and use a new output directory")
        save(output / "manifest.json", manifest)
        if args.export_only:
            print(json.dumps(export(output, questions, packet_hash)), flush=True)
            return 0
        preflight(output, args.timeout_seconds)
        if args.preflight_only:
            print("Exact models, subscription authentication, and synthetic tool audits verified.", flush=True)
            return 0
        export_lock = threading.Lock()
        export(output, questions, packet_hash)
        def run_pass(provider, n):
            for q in questions:
                collect_one(provider, q, output, packet_hash, n, args.timeout_seconds, args.attempts)
                with export_lock:
                    export(output, questions, packet_hash)
                print(json.dumps({"model": MODELS[provider], "pass": n, "id": q["id"], "status": "completed", "tool_audit": "verified"}), flush=True)
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(run_pass, p, n): (p, n) for p in ([args.provider] if args.provider else MODELS) for n in (1, 2)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"provider": futures[future], "detail": str(exc)})
                    print(json.dumps(failures[-1]), flush=True)
        summary = export(output, questions, packet_hash)
        if failures:
            save(output / ("interruption-" + uuid.uuid4().hex + ".json"), failures)
        print(json.dumps(summary), flush=True)
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
