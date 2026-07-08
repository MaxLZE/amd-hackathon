"""Local test harness for the Frugal Router agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from frugal_checks import build_run_report, load_json, validate_tasks
from router_core import load_dotenv


ROOT = Path(__file__).parent
DEFAULT_TASKS = ROOT / "tests" / "sample_tasks.json"
DEFAULT_OUT_DIR = ROOT / "out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent.py locally and validate results.")
    parser.add_argument("tasks_file", nargs="?", help="Legacy positional task file path.")
    parser.add_argument("--tasks", help="Task JSON file path. Overrides the positional path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for results.json and reports.")
    parser.add_argument("--config", help="Optional runtime config JSON file for routing and limits.")
    parser.add_argument("--timeout", type=float, default=600, help="Subprocess timeout in seconds.")
    parser.add_argument(
        "--json-report",
        nargs="?",
        const="-",
        help="Write a structured run report to this path, or stdout when omitted.",
    )
    return parser.parse_args()


def require_env() -> list[str]:
    if os.environ.get("LOCAL_MODEL_PATH"):
        return []
    missing = []
    for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
        if var not in os.environ or not os.environ[var].strip():
            missing.append(var)
    return missing


def print_block(label: str, text: str) -> None:
    if text.strip():
        print(f"\n--- {label} ---")
        print(text.rstrip())


def write_report(report: dict, target: str | None) -> None:
    if not target:
        return
    encoded = json.dumps(report, indent=2)
    if target == "-":
        print("\n--- json report ---")
        print(encoded)
        return
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    print(f"report written: {path}")


def main() -> int:
    args = parse_args()
    tasks_path = Path(args.tasks or args.tasks_file or DEFAULT_TASKS).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    results_path = out_dir / "results.json"
    config_path = Path(args.config).expanduser() if args.config else None

    load_dotenv(ROOT / ".env")
    missing_env = require_env()
    if missing_env:
        print(f"missing env vars: {', '.join(missing_env)}")
        print("copy .env.example to .env and fill in local values")
        return 1

    task_payload, task_load_errors = load_json(tasks_path)
    if task_load_errors:
        for error in task_load_errors:
            print(f"ERROR {error}")
        return 1
    tasks, task_report = validate_tasks(task_payload)
    if not task_report["valid"]:
        for error in task_report["errors"]:
            print(f"ERROR {error}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"TASKS_FILE": str(tasks_path), "RESULTS_FILE": str(results_path)})
    if config_path:
        env["FRUGAL_CONFIG_FILE"] = str(config_path)

    stdout = ""
    stderr = ""
    exit_code: int | None = None
    timed_out = False
    try:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "agent.py")],
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
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

    print_block("agent stdout", stdout)
    print_block("agent stderr", stderr)
    print(f"\nagent exit code: {exit_code}")

    result_payload, result_load_errors = load_json(results_path)
    if result_load_errors:
        result_payload = []
        stderr = f"{stderr}\n" + "\n".join(result_load_errors)

    report = build_run_report(tasks, result_payload, exit_code, stdout, stderr, timed_out)
    report["tasks_validation"] = task_report
    report["tasks_file"] = str(tasks_path)
    report["results_file"] = str(results_path)
    if config_path:
        report["config_file"] = str(config_path)

    result_report = report["results_validation"]
    for error in result_report["errors"]:
        print(f"ERROR {error}")
    for warning in task_report["warnings"] + result_report["warnings"]:
        print(f"WARN {warning}")

    empty = result_report["empty_task_ids"]
    print(f"schema OK: {result_report['count']} results, {len(empty)} empty answers {empty or ''}")
    print(f"deterministic checks: {len(result_report['warnings'])} warning(s)")

    token_summary = report["token_summary"]
    if token_summary["found"]:
        print(
            "tokens: "
            f"{token_summary['total_tokens']} "
            f"(prompt {token_summary['prompt_tokens']}, completion {token_summary['completion_tokens']})"
        )

    if isinstance(result_payload, list):
        by_id = {
            item.get("task_id"): item.get("answer", "")
            for item in result_payload
            if isinstance(item, dict) and isinstance(item.get("task_id"), str)
        }
        print("\n--- answers ---")
        for task in tasks:
            answer = by_id.get(task["task_id"], "").replace("\n", " ")
            print(f"\n[{task['task_id']}] {answer[:300]}")

    write_report(report, args.json_report)
    if timed_out:
        return 124
    if exit_code:
        return int(exit_code)
    return 0 if result_report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
