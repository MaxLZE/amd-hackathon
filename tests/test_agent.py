"""End-to-end tests for agent.py against a stub OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StubFireworks(BaseHTTPRequestHandler):
    """Per-model canned behavior: 'good-*' answers, 'missing-model' 404s,
    'truncator' always returns finish_reason=length."""

    calls: list[str] = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        model = body.get("model", "")
        type(self).calls.append(model)

        if model == "missing-model":
            self._send(404, {
                "error": {
                    "message": "Model not found, inaccessible, and/or not deployed",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            })
            return

        truncated = model == "truncator"
        self._send(200, {
            "id": "stub",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "PARTIAL" if truncated else f"answer from {model}"},
                "finish_reason": "length" if truncated else "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence request logging
        pass


class AgentEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StubFireworks)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def run_agent(self, tasks, allowed: str, overrides: dict[str, str] | None = None):
        StubFireworks.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "tasks.json"
            results_file = Path(tmp) / "results.json"
            tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
            env = os.environ.copy()
            env.pop("LOCAL_MODEL_PATH", None)
            env.update({
                "FIREWORKS_API_KEY": "test-key",
                "FIREWORKS_BASE_URL": self.base_url,
                "ALLOWED_MODELS": allowed,
                "TASKS_FILE": str(tasks_file),
                "RESULTS_FILE": str(results_file),
                "MAX_RUNTIME_SECONDS": "60",
            })
            env.update(overrides or {})
            proc = subprocess.run(
                [sys.executable, str(ROOT / "agent.py")],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(ROOT),
                check=False,
            )
            results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else None
        return proc, results

    def test_happy_path_skips_malformed_tasks_and_exits_zero(self) -> None:
        tasks = [
            {"task_id": "t1", "prompt": "What is the capital of Australia?"},
            "not-an-object",
            {"prompt": "no task id"},
            {"task_id": "t2", "prompt": 12345},
        ]
        proc, results = self.run_agent(tasks, "good-easy", {"FRUGAL_MODEL_EASY": "good-easy"})

        self.assertEqual(proc.returncode, 0, proc.stderr)
        by_id = {r["task_id"]: r["answer"] for r in results}
        self.assertEqual(set(by_id), {"t1", "t2"})
        self.assertEqual(by_id["t1"], "answer from good-easy")
        self.assertIn("skipping task 1", proc.stderr)
        self.assertIn("skipping task 2", proc.stderr)

    def test_rejected_model_is_disqualified_not_retried(self) -> None:
        tasks = [{"task_id": "t1", "prompt": "What is the capital of Australia?"}]
        proc, results = self.run_agent(
            tasks,
            "missing-model,good-easy",
            {"FRUGAL_MODEL_EASY": "missing-model"},
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(results[0]["answer"], "answer from good-easy")
        self.assertIn("model disqualified: missing-model", proc.stderr)
        self.assertEqual(StubFireworks.calls.count("missing-model"), 1)

    def test_truncated_answer_is_kept_not_discarded(self) -> None:
        tasks = [{"task_id": "t1", "prompt": "What is the capital of Australia?"}]
        proc, results = self.run_agent(tasks, "truncator", {"FRUGAL_MODEL_EASY": "truncator"})

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(results[0]["answer"], "PARTIAL")

    def test_empty_answers_still_exit_zero_with_valid_results(self) -> None:
        # No Fireworks config at all, no local model: results must still be
        # written with the right schema and the exit code must be non-zero
        # only for this catastrophic no-backend case.
        StubFireworks.calls = []
        with tempfile.TemporaryDirectory() as tmp:
            tasks_file = Path(tmp) / "tasks.json"
            results_file = Path(tmp) / "results.json"
            tasks_file.write_text(json.dumps([{"task_id": "t1", "prompt": "hi"}]), encoding="utf-8")
            env = os.environ.copy()
            for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS", "LOCAL_MODEL_PATH"):
                env.pop(var, None)
            env.update({"TASKS_FILE": str(tasks_file), "RESULTS_FILE": str(results_file)})
            proc = subprocess.run(
                [sys.executable, str(ROOT / "agent.py")],
                env=env, capture_output=True, text=True, timeout=60, cwd=str(ROOT), check=False,
            )
            results = json.loads(results_file.read_text(encoding="utf-8"))

        self.assertEqual(proc.returncode, 1)
        self.assertEqual(results, [{"task_id": "t1", "answer": ""}])


if __name__ == "__main__":
    unittest.main()
