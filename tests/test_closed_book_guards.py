"""Reject lookup attempts and unverified cached runs without calling providers."""

import json
import tempfile
import unittest
from pathlib import Path

from src import run_claude_benchmark as claude
from src import run_gpt_benchmark as gpt
from src.io_utils import write_json
from src.schemas import BenchmarkCase


def answer(label="A"):
    return {
        "objective_answers": [{"id": "fixture", "answer": label, "confidence": 80}],
        "healthbench_responses": [],
    }


def events(runner, extra=None, label="A"):
    if runner is gpt:
        items = [{"type": "thread.started", "thread_id": "fixture"}]
        ending = [
            {"type": "item.completed", "item": {
                "type": "agent_message", "text": json.dumps(answer(label)),
            }},
            {"type": "turn.completed", "usage": {}},
        ]
    else:
        items = [{"type": "system", "subtype": "init", "tools": [], "model": claude.MODEL}]
        ending = [{
            "type": "result", "subtype": "success", "is_error": False,
            "structured_output": answer(label), "usage": {},
        }]
    return "\n".join(json.dumps(item) for item in items + (extra or []) + ending)


class ClosedBookGuardsTest(unittest.TestCase):
    def test_gpt_disables_cross_session_memory_and_instruction_injection(self):
        command = gpt.codex_command("codex", Path("schema.json"))
        overrides = {
            command[index + 1].split("=", 1)[0]: command[index + 1].split("=", 1)[1]
            for index, value in enumerate(command) if value == "--config"
        }
        for key in ("features.memories", "memories.use_memories", "memories.generate_memories"):
            self.assertEqual(overrides[key], "false")
        self.assertEqual(overrides["project_doc_max_bytes"], "0")
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--ephemeral", command)
        self.assertNotIn("resume", command)
        self.assertNotIn("fork", command)

    def test_gpt_rejects_all_external_tool_items_even_with_a_valid_answer(self):
        for tool in ("web_search", "command_execution", "mcp_tool_call", "collab_tool_call"):
            with self.subTest(tool=tool), self.assertRaises(gpt.RunnerError):
                gpt.parse_event_stream(events(gpt, [{
                    "type": "item.started", "item": {"type": tool},
                }]))

    def test_unknown_top_level_events_cannot_bypass_audit(self):
        for runner in (gpt, claude):
            with self.subTest(runner=runner.MODEL), self.assertRaises(RuntimeError):
                runner.parse_event_stream(events(runner, [{"type": "web_search.started"}]))

    def test_gpt_rejects_missing_item(self):
        with self.assertRaises(gpt.RunnerError):
            gpt.parse_event_stream(events(gpt, [{"type": "item.completed"}]))

    def test_claude_rejects_external_content_even_with_a_valid_answer(self):
        for block in (
            {"type": "tool_use", "name": "WebSearch"},
            {"type": "tool_use", "name": "Read"},
            {"type": "tool_use", "name": "Bash"},
            {"type": "server_tool_use", "name": "web_search"},
            {"type": "web_fetch_tool_result"},
            {"type": "mcp_tool_result"},
            {"type": "tool_result", "tool_use_id": "unrecognized"},
            {"type": "future_external_tool"},
        ):
            with self.subTest(block=block), self.assertRaises(claude.ClaudeRunnerError):
                claude.parse_event_stream(events(claude, [{
                    "type": "assistant", "message": {"content": [block]},
                }]))

    def test_claude_rejects_usage_only_evidence_of_lookup(self):
        for usage in (
            {"server_tool_use": {"web_search_requests": 1}},
            {"server_tool_use": {"web_fetch_requests": 1}},
            {"webSearchRequests": 1},
            {"webFetchRequests": 1},
        ):
            for location in ("result", "message", "modelUsage"):
                with self.subTest(usage=usage, location=location):
                    records = [json.loads(line) for line in events(claude).splitlines()]
                    if location == "result":
                        records[-1]["usage"] = usage
                    elif location == "modelUsage":
                        records[-1]["modelUsage"] = {claude.MODEL: usage}
                    else:
                        records.insert(1, {"type": "assistant", "message": {
                            "content": [{"type": "text", "text": "fixture"}],
                            "usage": usage,
                        }})
                    with self.assertRaises(claude.ClaudeRunnerError):
                        claude.parse_event_stream("\n".join(map(json.dumps, records)))

    def test_claude_rejects_connected_tools_and_missing_model_identity(self):
        for override in (
            {"tools": ["WebSearch"]},
            {"mcp_servers": [{"name": "external", "status": "connected"}]},
            {"model": None},
        ):
            with self.subTest(override=override):
                records = [json.loads(line) for line in events(claude).splitlines()]
                records[0].update(override)
                with self.assertRaises(claude.ClaudeRunnerError):
                    claude.parse_event_stream("\n".join(map(json.dumps, records)))

    def test_resume_requires_matching_clean_event_log(self):
        case = BenchmarkCase(
            id="fixture", source_id="private", dataset="MedQA", split="test",
            category="fixture", question="Synthetic question?",
            choices={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_answer="A",
        )
        for runner in (gpt, claude):
            for scenario in ("clean", "missing", "lookup", "different_answer"):
                with self.subTest(model=runner.MODEL, scenario=scenario):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        metadata = dict(
                            benchmark_id="fixture", protocol_id="protocol",
                            prompt_sha256="prompt", run_number=1,
                        )
                        write_json(root / "response.json", {
                            **metadata, "model": runner.MODEL, "reasoning_effort": "high",
                            "responses": answer(),
                            "closed_book_audit": {
                                "closed_book_valid": True, "forbidden_tool_events": 0,
                                "unverified_marker": "must not be trusted",
                            },
                        })
                        log_path = root / "events.jsonl"
                        if scenario != "missing":
                            log_path.write_text(events(
                                runner,
                                extra=[{"type": "web_search.started"}] if scenario == "lookup" else None,
                                label="B" if scenario == "different_answer" else "A",
                            ), encoding="utf-8")
                        if scenario == "clean":
                            response, audit = runner.load_saved_run(
                                root / "response.json", log_path, [case], **metadata,
                            )
                            self.assertEqual(response, answer())
                            self.assertTrue(audit["closed_book_valid"])
                            self.assertNotIn("unverified_marker", audit)
                        else:
                            with self.assertRaises(RuntimeError):
                                runner.load_saved_run(
                                    root / "response.json", log_path, [case], **metadata,
                                )


if __name__ == "__main__":
    unittest.main()
