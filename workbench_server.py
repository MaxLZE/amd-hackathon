"""Local web workbench for Frugal Router."""

from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from frugal_checks import build_run_report, load_json, parse_tasks_text, validate_tasks
from router_core import RuntimeConfig, classify, load_dotenv, load_runtime_config, resolve_models, save_runtime_config


ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"
OUT_DIR = ROOT / "out"
CONFIG_FILE = OUT_DIR / "workbench_config.json"
TASKS_FILE = OUT_DIR / "workbench_tasks.json"
RESULTS_FILE = OUT_DIR / "workbench_results.json"
SAMPLE_TASKS = ROOT / "tests" / "sample_tasks.json"


def now_ms() -> int:
    return int(time.time() * 1000)


def read_env() -> dict[str, str]:
    env = os.environ.copy()
    load_dotenv(ROOT / ".env", env)
    return env


def allowed_models(env: dict[str, str]) -> list[str]:
    return [model.strip() for model in env.get("ALLOWED_MODELS", "").split(",") if model.strip()]


def openai_package_available() -> bool:
    return importlib.util.find_spec("openai") is not None


def env_status() -> dict[str, Any]:
    env = read_env()
    models = allowed_models(env)
    runtime = load_runtime_config(CONFIG_FILE if CONFIG_FILE.exists() else None, env)
    resolved = resolve_models(models, runtime) if models else {}
    required = {
        "FIREWORKS_API_KEY": bool(env.get("FIREWORKS_API_KEY")),
        "FIREWORKS_BASE_URL": bool(env.get("FIREWORKS_BASE_URL")),
        "ALLOWED_MODELS": bool(models),
    }
    local_model_configured = bool(env.get("LOCAL_MODEL_COMMAND") or env.get("LOCAL_MODEL_PATH"))
    openai_installed = openai_package_available()
    fireworks_configured = all(required.values())
    return {
        "ready": local_model_configured or (fireworks_configured and openai_installed),
        "required": required,
        "openai_package_available": openai_installed,
        "model_count": len(models),
        "allowed_models": models,
        "base_url_configured": bool(env.get("FIREWORKS_BASE_URL")),
        "local_model_configured": local_model_configured,
        "resolved_models": resolved,
    }


def prompt_to_tasks(prompt_text: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    text = prompt_text.strip()
    if not text:
        return [], {
            "valid": False,
            "errors": ["enter a prompt or paste a tasks.json array"],
            "warnings": [],
            "count": 0,
            "duplicates": [],
            "categories": {},
        }

    if text.startswith("["):
        return parse_tasks_text(text)

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
            task_id = payload.get("task_id") if isinstance(payload.get("task_id"), str) else "prompt-1"
            return validate_tasks([{"task_id": task_id, "prompt": payload["prompt"]}])

    return validate_tasks([{"task_id": f"prompt-{now_ms()}", "prompt": text}])


def estimate_prediction_scale(tasks: list[dict[str, str]]) -> dict[str, Any]:
    categories: dict[str, int] = {}
    score = 0.0
    category_weights = {
        "factual": 1.0,
        "sentiment": 0.8,
        "summary": 1.2,
        "ner": 1.2,
        "math": 2.0,
        "logic": 2.4,
        "code_debug": 2.6,
        "codegen": 2.6,
    }
    total_chars = 0
    for task in tasks:
        prompt = task["prompt"]
        category = classify(prompt)
        categories[category] = categories.get(category, 0) + 1
        total_chars += len(prompt)
        score += category_weights.get(category, 1.0) + min(len(prompt) / 1200, 2.5)

    if score <= 2:
        scale = "small"
    elif score <= 7:
        scale = "medium"
    elif score <= 16:
        scale = "large"
    else:
        scale = "batch"

    return {
        "scale": scale,
        "score": round(score, 2),
        "task_count": len(tasks),
        "total_chars": total_chars,
        "categories": categories,
    }


def auto_runtime_config(tasks: list[dict[str, str]], env: dict[str, str] | None = None) -> tuple[RuntimeConfig, dict[str, Any]]:
    source_env = env or read_env()
    base = load_runtime_config(None, source_env)
    prediction = estimate_prediction_scale(tasks)
    scale = prediction["scale"]
    count = max(1, prediction["task_count"])
    heavy = any(category in prediction["categories"] for category in ("math", "logic", "code_debug", "codegen"))

    settings = {
        "small": (90, 1, 20, 0.85),
        "medium": (180, min(4, count), 30, 1.0),
        "large": (360, min(6, count), 45, 1.18),
        "batch": (510, min(8, count), 55, 1.25),
    }[scale]
    runtime_seconds, concurrency, timeout, token_multiplier = settings
    if heavy:
        timeout += 10
        token_multiplier = max(token_multiplier, 1.1)

    categories = json.loads(json.dumps(base.categories))
    for spec in categories.values():
        spec["max_tokens"] = max(48, int(spec["max_tokens"] * token_multiplier))

    local_configured = bool(source_env.get("LOCAL_MODEL_COMMAND") or source_env.get("LOCAL_MODEL_PATH"))
    strategy = (
        "local-first when LOCAL_MODEL_COMMAND is configured; Fireworks fallback records tokens"
        if local_configured
        else "Fireworks-only; no bundled local model configured"
    )
    prediction["strategy"] = strategy
    prediction["local_model_configured"] = local_configured

    runtime = RuntimeConfig(
        max_runtime_seconds=max(runtime_seconds, min(510, 60 + count * 12)),
        max_concurrency=concurrency,
        per_call_timeout=timeout,
        max_attempts=base.max_attempts,
        categories=categories,
        tier_preferences=base.tier_preferences,
        model_overrides=base.model_overrides,
    )
    return runtime, prediction


class RunManager:
    def __init__(
        self,
        root: Path = ROOT,
        out_dir: Path = OUT_DIR,
        agent_cmd: list[str] | None = None,
    ) -> None:
        self.root = root
        self.out_dir = out_dir
        self.tasks_file = out_dir / "workbench_tasks.json"
        self.results_file = out_dir / "workbench_results.json"
        self.config_file = out_dir / "workbench_config.json"
        self.agent_cmd = agent_cmd or [sys.executable, str(root / "agent.py")]
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.cancel_requested = False
        self.state: dict[str, Any] = self._empty_state()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "run_id": None,
            "status": "idle",
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "timed_out": False,
            "logs": [],
            "stdout": "",
            "stderr": "",
            "tasks_validation": None,
            "results_validation": None,
            "token_summary": None,
            "prediction": None,
            "results": [],
            "error": None,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def start(self, prompt_text: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        del config
        tasks, task_report = prompt_to_tasks(prompt_text)
        if not task_report["valid"]:
            raise ValueError("; ".join(task_report["errors"]))

        with self.lock:
            if self.process and self.process.poll() is None:
                raise RuntimeError("a run is already active")

            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.tasks_file.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
            self.results_file.write_text("[]", encoding="utf-8")

            env = read_env()
            runtime, prediction = auto_runtime_config(tasks, env)
            save_runtime_config(runtime.public_dict(), self.config_file)
            missing = [key for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS") if not env.get(key)]
            local_model_configured = bool(env.get("LOCAL_MODEL_COMMAND") or env.get("LOCAL_MODEL_PATH"))
            if missing and not local_model_configured:
                raise ValueError(f"missing env vars: {', '.join(missing)}")
            if not local_model_configured and not openai_package_available():
                raise ValueError("Python package 'openai' is not installed. Install dependencies with python3 -m pip install -r requirements.txt.")
            env.update(
                {
                    "TASKS_FILE": str(self.tasks_file),
                    "RESULTS_FILE": str(self.results_file),
                    "FRUGAL_CONFIG_FILE": str(self.config_file),
                }
            )

            self.cancel_requested = False
            run_id = now_ms()
            self.state = self._empty_state()
            self.state.update(
                {
                    "run_id": run_id,
                    "status": "running",
                    "started_at": run_id,
                    "tasks_validation": task_report,
                    "prediction": prediction,
                }
            )
            self.process = subprocess.Popen(
                self.agent_cmd,
                cwd=str(self.root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            threading.Thread(target=self._read_stream, args=("stdout", self.process.stdout), daemon=True).start()
            threading.Thread(target=self._read_stream, args=("stderr", self.process.stderr), daemon=True).start()
            threading.Thread(target=self._watch, args=(self.process, tasks, runtime), daemon=True).start()
        return self.snapshot()

    def current_config(self) -> RuntimeConfig:
        return load_runtime_config(self.config_file if self.config_file.exists() else None, read_env())

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            self.cancel_requested = True
            if self.process and self.process.poll() is None:
                self.state["status"] = "cancelling"
                self.process.terminate()
        return self.snapshot()

    def _append_log(self, stream: str, line: str) -> None:
        with self.lock:
            self.state[stream] += line
            self.state["logs"].append({"stream": stream, "line": line.rstrip("\n"), "time": now_ms()})
            self.state["logs"] = self.state["logs"][-600:]

    def _read_stream(self, stream_name: str, pipe: Any) -> None:
        if pipe is None:
            return
        try:
            for line in pipe:
                self._append_log(stream_name, line)
        finally:
            pipe.close()

    def _watch(self, process: subprocess.Popen[str], tasks: list[dict[str, str]], runtime: RuntimeConfig) -> None:
        timed_out = False
        try:
            exit_code = process.wait(timeout=runtime.max_runtime_seconds + 20)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            exit_code = process.wait()

        time.sleep(0.1)
        result_payload, result_errors = load_json(self.results_file)
        if result_errors:
            result_payload = []
            with self.lock:
                self.state["stderr"] += "\n".join(result_errors)

        with self.lock:
            stdout = self.state["stdout"]
            stderr = self.state["stderr"]

        report = build_run_report(tasks, result_payload, exit_code, stdout, stderr, timed_out)
        with self.lock:
            cancelled = self.cancel_requested and not timed_out
            result_validation = report["results_validation"]
            has_empty_answers = bool(result_validation.get("empty_task_ids"))
            success = exit_code == 0 and not timed_out and result_validation.get("valid") and not has_empty_answers
            if has_empty_answers:
                error = f"agent returned empty answer(s): {result_validation['empty_task_ids']}"
            elif exit_code == 0 and result_validation.get("valid"):
                error = None
            else:
                error = "agent run did not complete successfully"
            self.state.update(
                {
                    "status": "cancelled" if cancelled else ("succeeded" if success else "failed"),
                    "ended_at": now_ms(),
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "results": result_payload if isinstance(result_payload, list) else [],
                    "results_validation": result_validation,
                    "token_summary": report["token_summary"],
                    "error": None if cancelled else error,
                }
            )
            self.process = None


MANAGER = RunManager()


def current_config_payload() -> dict[str, Any]:
    return MANAGER.current_config().public_dict()


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def read_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length).decode("utf-8") if length else "{}"
    return json.loads(raw or "{}")


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "FrugalWorkbench/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/env":
            json_response(self, {"env": env_status(), "config": current_config_payload()})
        elif parsed.path == "/api/config":
            json_response(self, {"config": current_config_payload()})
        elif parsed.path == "/api/sample-tasks":
            text = SAMPLE_TASKS.read_text(encoding="utf-8")
            tasks, report = validate_tasks(json.loads(text))
            json_response(self, {"tasksText": text, "tasks": tasks, "validation": report})
        elif parsed.path == "/api/run/status":
            json_response(self, MANAGER.snapshot())
        elif parsed.path == "/api/results":
            payload, errors = load_json(RESULTS_FILE)
            json_response(self, {"results": payload or [], "errors": errors})
        elif parsed.path.startswith("/api/"):
            json_response(self, {"error": "not found"}, 404)
        else:
            self.serve_static(parsed.path)

    def do_POST(self) -> None:
        try:
            body = read_body(self)
            parsed = urlparse(self.path)
            if parsed.path == "/api/config":
                config = body.get("config", body)
                runtime = save_runtime_config(config, CONFIG_FILE)
                json_response(self, {"config": runtime.public_dict()})
            elif parsed.path == "/api/validate-tasks":
                tasks, report = parse_tasks_text(body.get("tasksText", ""))
                json_response(self, {"tasks": tasks, "validation": report})
            elif parsed.path == "/api/run":
                state = MANAGER.start(body.get("prompt") or body.get("tasksText", ""))
                json_response(self, state)
            elif parsed.path == "/api/run/cancel":
                json_response(self, MANAGER.cancel())
            else:
                json_response(self, {"error": "not found"}, 404)
        except json.JSONDecodeError as exc:
            json_response(self, {"error": f"invalid JSON body: {exc.msg}"}, 400)
        except (RuntimeError, ValueError) as exc:
            json_response(self, {"error": str(exc)}, 400)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path == "/" else unquote(request_path.lstrip("/"))
        path = (WEB_DIR / relative).resolve()
        if not path.is_relative_to(WEB_DIR.resolve()) or not path.exists() or path.is_dir():
            path = WEB_DIR / "index.html"
        content = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the local Frugal Router Workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    WEB_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    print(f"Frugal Router Workbench: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
