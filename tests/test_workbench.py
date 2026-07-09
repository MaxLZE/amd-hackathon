from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from frugal_checks import parse_agent_summary, validate_results, validate_tasks
from router_core import RuntimeConfig, load_runtime_config, parse_allowed_models, resolve_models, save_runtime_config
from workbench_server import RunManager, auto_runtime_config, prompt_to_tasks


class ValidationTests(unittest.TestCase):
    def test_validate_tasks_rejects_duplicates_and_bad_prompts(self) -> None:
        tasks, report = validate_tasks(
            [
                {"task_id": "a", "prompt": "Explain HTTP."},
                {"task_id": "a", "prompt": "Explain HTTPS."},
                {"task_id": "b", "prompt": ""},
            ]
        )

        self.assertFalse(report["valid"])
        self.assertEqual(len(tasks), 2)
        self.assertIn("a", report["duplicates"])
        self.assertTrue(any("duplicate task_id" in error for error in report["errors"]))
        self.assertTrue(any("prompt must be" in error for error in report["errors"]))

    def test_validate_results_detects_missing_extra_duplicate_and_empty(self) -> None:
        tasks = [{"task_id": "a", "prompt": "Explain HTTP."}, {"task_id": "b", "prompt": "Explain HTTPS."}]
        report = validate_results(
            [
                {"task_id": "a", "answer": ""},
                {"task_id": "a", "answer": "duplicate"},
                {"task_id": "c", "answer": "extra"},
            ],
            tasks,
        )

        self.assertFalse(report["valid"])
        self.assertEqual(report["missing_task_ids"], ["b"])
        self.assertEqual(report["extra_task_ids"], ["c"])
        self.assertEqual(report["duplicate_result_ids"], ["a"])
        self.assertIn("b", report["empty_task_ids"])

    def test_parse_agent_summary(self) -> None:
        summary = parse_agent_summary(
            "done in 1.2s | tokens: 15 (prompt 6, completion 9) | by category: {'math': 15}"
        )

        self.assertTrue(summary["found"])
        self.assertEqual(summary["total_tokens"], 15)
        self.assertEqual(summary["by_category"], {"math": 15})


class ConfigTests(unittest.TestCase):
    def test_save_load_and_exact_override_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = RuntimeConfig(max_runtime_seconds=120, model_overrides={"easy": "allowed-a"})
            save_runtime_config(config.public_dict(), path)
            loaded = load_runtime_config(path, {})
            resolved = resolve_models(["allowed-a", "allowed-b"], loaded)

        self.assertEqual(loaded.max_runtime_seconds, 120)
        self.assertEqual(resolved["easy"], "allowed-a")

    def test_parse_allowed_models_accepts_json_array_and_expands_bare_names(self) -> None:
        # Bare names 404 against the Fireworks API even when the model is
        # live under accounts/fireworks/models/ — expand them.
        models = parse_allowed_models('["minimax-m3", "kimi-k2p7-code"]')

        self.assertEqual(
            models,
            ["accounts/fireworks/models/minimax-m3", "accounts/fireworks/models/kimi-k2p7-code"],
        )

    def test_parse_allowed_models_keeps_pathed_ids_verbatim(self) -> None:
        models = parse_allowed_models(
            "accounts/fireworks/models/kimi-k2p7-code, accounts/customer-team/models/private-ft, glm-5p2"
        )

        self.assertEqual(
            models,
            [
                "accounts/fireworks/models/kimi-k2p7-code",
                "accounts/customer-team/models/private-ft",
                "accounts/fireworks/models/glm-5p2",
            ],
        )

    def test_prompt_to_tasks_accepts_plain_prompt(self) -> None:
        tasks, report = prompt_to_tasks("Explain how HTTPS works.")

        self.assertTrue(report["valid"])
        self.assertEqual(report["count"], 1)
        self.assertEqual(tasks[0]["prompt"], "Explain how HTTPS works.")
        self.assertTrue(tasks[0]["task_id"].startswith("prompt-"))

    def test_auto_runtime_config_scales_heavy_prompts(self) -> None:
        tasks = [
            {
                "task_id": "logic",
                "prompt": "A puzzle: exactly one of three boxes contains a prize. Which box has it?",
            },
            {"task_id": "code", "prompt": "Write a Python function that debounces another function."},
        ]
        runtime, prediction = auto_runtime_config(tasks, {"ALLOWED_MODELS": "allowed-a"})

        self.assertIn(prediction["scale"], {"medium", "large", "batch"})
        self.assertGreaterEqual(runtime.per_call_timeout, 40)
        self.assertIn("logic", prediction["categories"])


class RunManagerTests(unittest.TestCase):
    def test_mocked_run_writes_results_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_agent = tmp_path / "fake_agent.py"
            fake_agent.write_text(
                "\n".join(
                    [
                        "import json, os, sys",
                        "tasks = json.load(open(os.environ['TASKS_FILE'], encoding='utf-8'))",
                        "payload = [{'task_id': t['task_id'], 'answer': 'ok'} for t in tasks]",
                        "json.dump(payload, open(os.environ['RESULTS_FILE'], 'w', encoding='utf-8'))",
                        "print(\"done in 0.1s | tokens: 3 (prompt 1, completion 2) | by category: {'factual': 3}\", file=sys.stderr)",
                    ]
                ),
                encoding="utf-8",
            )
            manager = RunManager(
                root=Path(__file__).resolve().parents[1],
                out_dir=tmp_path,
                agent_cmd=[sys.executable, str(fake_agent)],
            )
            config = RuntimeConfig(max_runtime_seconds=2).public_dict()
            env = {
                "FIREWORKS_API_KEY": "test",
                "FIREWORKS_BASE_URL": "https://example.test",
                "ALLOWED_MODELS": "allowed-a",
                "LOCAL_MODEL_COMMAND": "fake-local",
            }
            with patch.dict(os.environ, env, clear=False):
                manager.start(json.dumps([{"task_id": "t1", "prompt": "Explain HTTP."}]), config)
                deadline = time.time() + 3
                snapshot = manager.snapshot()
                while snapshot["status"] in {"running", "cancelling"} and time.time() < deadline:
                    time.sleep(0.05)
                    snapshot = manager.snapshot()

        self.assertEqual(snapshot["status"], "succeeded")
        self.assertEqual(snapshot["results"][0]["answer"], "ok")
        self.assertEqual(snapshot["token_summary"]["total_tokens"], 3)
        self.assertTrue(snapshot["results_validation"]["valid"])

    def test_empty_answer_marks_run_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_agent = tmp_path / "fake_agent.py"
            fake_agent.write_text(
                "\n".join(
                    [
                        "import json, os, sys",
                        "tasks = json.load(open(os.environ['TASKS_FILE'], encoding='utf-8'))",
                        "payload = [{'task_id': t['task_id'], 'answer': ''} for t in tasks]",
                        "json.dump(payload, open(os.environ['RESULTS_FILE'], 'w', encoding='utf-8'))",
                        "print(\"done in 0.1s | tokens: 0 (prompt 0, completion 0) | by category: {}\", file=sys.stderr)",
                    ]
                ),
                encoding="utf-8",
            )
            manager = RunManager(
                root=Path(__file__).resolve().parents[1],
                out_dir=tmp_path,
                agent_cmd=[sys.executable, str(fake_agent)],
            )
            config = RuntimeConfig(max_runtime_seconds=2).public_dict()
            env = {
                "FIREWORKS_API_KEY": "test",
                "FIREWORKS_BASE_URL": "https://example.test",
                "ALLOWED_MODELS": "allowed-a",
                "LOCAL_MODEL_COMMAND": "fake-local",
            }
            with patch.dict(os.environ, env, clear=False):
                manager.start(json.dumps([{"task_id": "t1", "prompt": "Explain HTTP."}]), config)
                deadline = time.time() + 3
                snapshot = manager.snapshot()
                while snapshot["status"] in {"running", "cancelling"} and time.time() < deadline:
                    time.sleep(0.05)
                    snapshot = manager.snapshot()

        self.assertEqual(snapshot["status"], "failed")
        self.assertIn("empty answer", snapshot["error"])
        self.assertEqual(snapshot["results_validation"]["empty_task_ids"], ["t1"])


if __name__ == "__main__":
    unittest.main()
