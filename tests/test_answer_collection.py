import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import collect_medxpert_answers as c

Q = {"id": "MXH-001", "question": "Original question\nunchanged?", "choices": {k: k + " text" for k in "ABCDEFGHIJ"}}
A = {"id": Q["id"], "answer": "J", "explanation": "A brief clinical explanation."}


def gpt_events(answer=A, tool=None, session="unique-test-session"):
    events = [{"type": "thread.started", "thread_id": session}, {"type": "turn.started"}]
    if tool:
        events.append({"type": "item.started", "item": {"type": tool}})
    events.extend([{"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(answer)}}, {"type": "turn.completed"}])
    return "\n".join(map(json.dumps, events)) + "\n"


def claude_events(tools=None, model=None, blocks=None, usage=None):
    model = model or c.MODELS["claude"]
    common = {"session_id": "claude-test-session"}
    return "\n".join(map(json.dumps, [
        {**common, "type": "system", "subtype": "init", "tools": tools or [], "mcp_servers": [], "model": model},
        {**common, "type": "assistant", "message": {"model": model, "content": blocks or [{"type": "text", "text": json.dumps(A)}]}},
        {**common, "type": "result", "subtype": "success", "is_error": False, "result": json.dumps(A), "usage": usage or {}}
    ]))


class AnswerCollectionTests(unittest.TestCase):
    def test_only_packet_is_opened(self):
        original = Path.read_bytes
        opened = []
        def tracked(p):
            opened.append(p.resolve())
            self.assertEqual(p.resolve(), c.PACKET.resolve())
            return original(p)
        with patch.object(Path, "read_bytes", tracked):
            questions, _ = c.load_packet()
        self.assertEqual(len(questions), 100)
        self.assertEqual(opened, [c.PACKET.resolve()])
        with self.assertRaises(c.CollectionError):
            c.load_packet(Path("data/benchmark.json"))

    def test_prompt_preserves_one_original_question(self):
        self.assertEqual(json.loads(c.prompt_for(Q).split("QUESTION:\n", 1)[1]), Q)
        self.assertEqual(c.validate_answer(A, Q), A)
        for answer in [{**A, "answer": "K"}, {**A, "id": "OTHER"}, {**A, "explanation": ""}, {**A, "extra": 1}]:
            with self.assertRaises(c.CollectionError):
                c.validate_answer(answer, Q)

    def test_gpt_audit_rejects_tools_and_incomplete_turns(self):
        self.assertEqual(c.audit_stream("gpt", gpt_events(), Q)[0], A)
        for tool in ["command_execution", "web_search", "mcp_tool_call", "file_change", "collab_tool_call"]:
            with self.assertRaises(c.CollectionError):
                c.audit_stream("gpt", gpt_events(tool=tool), Q)
        with self.assertRaises(c.CollectionError):
            c.audit_stream("gpt", gpt_events().rsplit('\n', 2)[0], Q)

    def test_claude_audit_rejects_tools_model_substitution_and_server_search(self):
        self.assertEqual(c.audit_stream("claude", claude_events(), Q)[0], A)
        variants = [dict(tools=["WebSearch"]), dict(model="another-model"),
                    dict(blocks=[{"type": "tool_use", "name": "Read"}]),
                    dict(usage={"server_tool_use": {"web_search_requests": 1}})]
        for variant in variants:
            with self.assertRaises(c.CollectionError):
                c.audit_stream("claude", claude_events(**variant), Q)

    def test_claude_reasoning_progress_is_auditable(self):
        events = [json.loads(line) for line in claude_events().splitlines()]
        event = {"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 42,
                 "estimated_tokens_delta": 42, "session_id": "claude-test-session", "uuid": "progress"}
        events.insert(1, event)
        raw = "\n".join(map(json.dumps, events))
        self.assertEqual(c.audit_stream("claude", raw, Q)[0], A)
        event["tool"] = "WebSearch"
        with self.assertRaises(c.CollectionError):
            c.audit_stream("claude", "\n".join(map(json.dumps, events)), Q)

    def test_prose_and_fences_recover_without_changing_answer(self):
        for wrapped in ['Brief explanation.\n\n' + json.dumps(A), 'Brief explanation.\n\n```json\n' + json.dumps(A) + '\n```']:
            events = [json.loads(line) for line in claude_events().splitlines()]
            events[-1]['result'] = wrapped
            self.assertEqual(c.audit_stream('claude', '\n'.join(map(json.dumps, events)), Q)[0], A)
        events[-1]['result'] = json.dumps(A) + '\n' + json.dumps({**A, 'answer': 'I'})
        with self.assertRaises(c.CollectionError):
            c.audit_stream('claude', '\n'.join(map(json.dumps, events)), Q)

    def test_recover_complete_logs_without_repeating_and_preserve_failed_attempt(self):
        with tempfile.TemporaryDirectory() as t:
            output = Path(t)
            folder = output / c.MODELS["gpt"] / "pass_1" / Q["id"]
            request = {"prompt_sha256": c.digest(c.prompt_for(Q).encode()), "model": c.MODELS["gpt"], "reasoning_effort": "xhigh"}
            for i, raw in [(1, '{"type":"turn.failed"}\n'), (2, gpt_events())]:
                attempt = folder / ("attempt-%04d" % i)
                c.save(attempt / "request.json", request)
                c.durable(attempt / "events.jsonl", raw)
            with patch.object(c, "execute", side_effect=AssertionError("Must not rerun verified response")):
                result = c.collect_one("gpt", Q, output, "packet-hash", 1, 1, 1)
                self.assertEqual(result["answer"], "J")
                c.collect_one("gpt", Q, output, "packet-hash", 1, 1, 1)
            self.assertEqual((folder / "attempt-0001/events.jsonl").read_text(), '{"type":"turn.failed"}\n')
            c.durable(folder / "attempt-0002/events.jsonl", gpt_events({**A, "answer": "I"}))
            with self.assertRaises(c.CollectionError):
                c.recover("gpt", Q, folder, "packet-hash", 1)

    def test_commands_disable_context_and_pin_exact_targets(self):
        with tempfile.TemporaryDirectory() as t:
            for provider in c.MODELS:
                cmd = c.command_for(provider, Path(t))
                self.assertEqual(cmd[cmd.index("--model") + 1], c.MODELS[provider])
                self.assertNotIn("--resume", cmd)
                self.assertNotIn("--continue", cmd)
                self.assertNotIn("--fallback-model", cmd)
            cmd = c.command_for("claude", Path(t))
            self.assertEqual(cmd[cmd.index("--tools") + 1], "")
            self.assertEqual(cmd[cmd.index("--effort") + 1], "xhigh")
            self.assertIn("--safe-mode", cmd)

    def test_exports_have_no_evaluation_columns(self):
        with tempfile.TemporaryDirectory() as t:
            c.export(Path(t), [Q], "packet")
            text = (Path(t) / "comparison.csv").read_text()
            for forbidden in ["correct", "accuracy", "score", "gold"]:
                self.assertNotIn(forbidden, text)
            self.assertEqual(len(list(Path(t).glob("*_answers.json"))), 4)


if __name__ == "__main__":
    unittest.main()
