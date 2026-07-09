"""Vercel serverless entry point for the prompt workbench."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import mimetypes
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frugal_checks import build_run_report, load_json, parse_tasks_text, validate_tasks  # noqa: E402
from router_core import save_runtime_config  # noqa: E402
from workbench_server import auto_runtime_config, env_status, prompt_to_tasks, read_env  # noqa: E402


TMP_DIR = Path(tempfile.gettempdir()) / "frugal-router-vercel"
WEB_DIR = ROOT / "web"


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length).decode("utf-8") if length else "{}"
    return json.loads(raw or "{}")


def _static_response(handler: BaseHTTPRequestHandler, request_path: str) -> None:
    if request_path in {"", "/"}:
        relative = "index.html"
    else:
        relative = unquote(request_path.lstrip("/"))
        if relative.startswith("web/"):
            relative = relative[4:]
    path = (WEB_DIR / relative).resolve()
    if not path.is_relative_to(WEB_DIR.resolve()) or not path.exists() or path.is_dir():
        path = WEB_DIR / "index.html"
    content = path.read_bytes()
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


def _terminal_state(
    *,
    tasks: list[dict[str, str]],
    result_payload,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    timed_out: bool,
    prediction: dict,
    run_id: int,
) -> dict:
    report = build_run_report(tasks, result_payload, exit_code, stdout, stderr, timed_out)
    validation = report["results_validation"]
    has_empty_answers = bool(validation.get("empty_task_ids"))
    success = exit_code == 0 and not timed_out and validation.get("valid") and not has_empty_answers
    if has_empty_answers:
        error = f"agent returned empty answer(s): {validation['empty_task_ids']}"
    elif exit_code == 0 and validation.get("valid"):
        error = None
    else:
        error = "agent run did not complete successfully"
    return {
        "run_id": run_id,
        "status": "succeeded" if success else "failed",
        "started_at": run_id,
        "ended_at": run_id,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "logs": [],
        "stdout": stdout,
        "stderr": stderr,
        "tasks_validation": None,
        "results_validation": validation,
        "token_summary": report["token_summary"],
        "prediction": prediction,
        "results": result_payload if isinstance(result_payload, list) else [],
        "error": error,
    }


def run_prompt(prompt: str) -> tuple[dict, int]:
    tasks, task_report = prompt_to_tasks(prompt)
    if not task_report["valid"]:
        return {"error": "; ".join(task_report["errors"])}, 400

    env = read_env()
    missing = [key for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS") if not env.get(key)]
    local_model_configured = bool(env.get("LOCAL_MODEL_COMMAND") or env.get("LOCAL_MODEL_PATH"))
    if missing and not local_model_configured:
        return {"error": f"missing env vars: {', '.join(missing)}"}, 400

    run_id = int(os.environ.get("VERCEL_RUN_ID", "0")) or int(__import__("time").time() * 1000)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tasks_file = TMP_DIR / f"tasks_{run_id}.json"
    results_file = TMP_DIR / f"results_{run_id}.json"
    config_file = TMP_DIR / f"config_{run_id}.json"

    runtime, prediction = auto_runtime_config(tasks, env)
    tasks_file.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    results_file.write_text("[]", encoding="utf-8")
    save_runtime_config(runtime.public_dict(), config_file)

    child_env = env.copy()
    child_env.update(
        {
            "TASKS_FILE": str(tasks_file),
            "RESULTS_FILE": str(results_file),
            "FRUGAL_CONFIG_FILE": str(config_file),
            # Vercel vendors pip packages onto the function's sys.path only;
            # forward it so the agent subprocess can import `openai` too.
            "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
        }
    )
    timeout = min(float(os.environ.get("VERCEL_AGENT_TIMEOUT", "55")), runtime.max_runtime_seconds + 20)
    stdout = ""
    stderr = ""
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "agent.py")],
            cwd=str(ROOT),
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        exit_code = 124
        timed_out = True

    result_payload, result_errors = load_json(results_file)
    if result_errors:
        result_payload = []
        stderr = f"{stderr}\n" + "\n".join(result_errors)

    state = _terminal_state(
        tasks=tasks,
        result_payload=result_payload,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        prediction=prediction,
        run_id=run_id,
    )
    state["tasks_validation"] = task_report
    return state, 200


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - Vercel/http.server API
        path = urlparse(self.path).path
        if path == "/api/env":
            _json_response(self, {"env": env_status()})
        elif path == "/api/run/status":
            _json_response(self, {"status": "idle", "results": []})
        elif path == "/api/sample-tasks":
            text = (ROOT / "tests" / "sample_tasks.json").read_text(encoding="utf-8")
            tasks, report = validate_tasks(json.loads(text))
            _json_response(self, {"tasksText": text, "tasks": tasks, "validation": report})
        elif path == "/api/results":
            _json_response(self, {"results": [], "errors": []})
        elif path.startswith("/api/"):
            _json_response(self, {"error": "not found"}, 404)
        else:
            _static_response(self, path)

    def do_POST(self) -> None:  # noqa: N802 - Vercel/http.server API
        try:
            path = urlparse(self.path).path
            body = _read_json(self)
            if path == "/api/run":
                payload, status = run_prompt(body.get("prompt") or body.get("tasksText", ""))
                _json_response(self, payload, status)
            elif path == "/api/validate-tasks":
                tasks, report = parse_tasks_text(body.get("tasksText", ""))
                _json_response(self, {"tasks": tasks, "validation": report})
            elif path == "/api/run/cancel":
                _json_response(self, {"status": "cancelled", "results": []})
            else:
                _json_response(self, {"error": "not found"}, 404)
        except json.JSONDecodeError as exc:
            _json_response(self, {"error": f"invalid JSON body: {exc.msg}"}, 400)
        except Exception as exc:  # noqa: BLE001 - return JSON instead of dropping fetch
            _json_response(self, {"error": f"server error: {exc}"}, 500)
