"""Validation and reporting helpers for local Frugal Router runs."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from router_core import classify


SUMMARY_RE = re.compile(
    r"done in (?P<seconds>[\d.]+)s \| tokens: (?P<total>\d+) "
    r"\(prompt (?P<prompt>\d+), completion (?P<completion>\d+)\) \| "
    r"by category: (?P<by_category>\{.*\})"
)


def load_json(path: Path) -> tuple[Any | None, list[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except FileNotFoundError:
        return None, [f"{path} does not exist"]
    except json.JSONDecodeError as exc:
        return None, [f"{path} is invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}"]


def parse_tasks_text(text: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return [], {
            "valid": False,
            "errors": [f"tasks JSON is invalid: {exc.msg} at line {exc.lineno}, column {exc.colno}"],
            "warnings": [],
            "count": 0,
            "duplicates": [],
            "categories": {},
        }
    return validate_tasks(payload)


def validate_tasks(payload: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    tasks: list[dict[str, str]] = []
    seen: set[str] = set()
    duplicates: list[str] = []

    if not isinstance(payload, list):
        return [], {
            "valid": False,
            "errors": ["tasks.json must be a JSON array"],
            "warnings": [],
            "count": 0,
            "duplicates": [],
            "categories": {},
        }

    for index, task in enumerate(payload):
        if not isinstance(task, dict):
            errors.append(f"task {index} must be an object")
            continue
        task_id = task.get("task_id")
        prompt = task.get("prompt")
        if not isinstance(task_id, str) or not task_id.strip():
            errors.append(f"task {index} has a missing or non-string task_id")
            continue
        if task_id in seen:
            duplicates.append(task_id)
            errors.append(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"{task_id}: prompt must be a non-empty string")
            continue
        extra_keys = sorted(set(task) - {"task_id", "prompt"})
        if extra_keys:
            warnings.append(f"{task_id}: extra keys ignored locally: {', '.join(extra_keys)}")
        tasks.append({"task_id": task_id, "prompt": prompt})

    categories: dict[str, int] = {}
    for task in tasks:
        category = classify(task["prompt"])
        categories[category] = categories.get(category, 0) + 1

    return tasks, {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "count": len(tasks),
        "duplicates": duplicates,
        "categories": categories,
    }


def validate_results(payload: Any, tasks: list[dict[str, str]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    by_id: dict[str, str] = {}
    duplicate_result_ids: list[str] = []

    if not isinstance(payload, list):
        return {
            "valid": False,
            "errors": ["results.json must be a JSON array"],
            "warnings": [],
            "count": 0,
            "missing_task_ids": [task["task_id"] for task in tasks],
            "extra_task_ids": [],
            "duplicate_result_ids": [],
            "empty_task_ids": [],
        }

    for index, result in enumerate(payload):
        if not isinstance(result, dict):
            errors.append(f"result {index} must be an object")
            continue
        keys = set(result)
        if keys != {"task_id", "answer"}:
            errors.append(f"result {index} must contain exactly task_id and answer")
        task_id = result.get("task_id")
        answer = result.get("answer")
        if not isinstance(task_id, str) or not task_id:
            errors.append(f"result {index} has a missing or non-string task_id")
            continue
        if not isinstance(answer, str):
            errors.append(f"{task_id}: answer must be a string")
            continue
        if task_id in by_id:
            duplicate_result_ids.append(task_id)
            errors.append(f"duplicate result task_id: {task_id}")
        by_id[task_id] = answer

    task_ids = [task["task_id"] for task in tasks]
    task_id_set = set(task_ids)
    result_id_set = set(by_id)
    missing = [task_id for task_id in task_ids if task_id not in result_id_set]
    extra = sorted(result_id_set - task_id_set)
    empty = [task_id for task_id in task_ids if not by_id.get(task_id, "").strip()]
    if missing:
        errors.append(f"missing task_ids: {missing}")
    if extra:
        errors.append(f"extra task_ids: {extra}")
    if empty:
        warnings.append(f"empty answers: {empty}")
    warnings.extend(deterministic_warnings(tasks, by_id))

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "count": len(payload),
        "missing_task_ids": missing,
        "extra_task_ids": extra,
        "duplicate_result_ids": duplicate_result_ids,
        "empty_task_ids": empty,
    }


def deterministic_warnings(tasks: list[dict[str, str]], answers: dict[str, str]) -> list[str]:
    warnings: list[str] = []
    for task in tasks:
        task_id = task["task_id"]
        prompt = task["prompt"]
        answer = answers.get(task_id, "")
        if not answer.strip():
            continue
        if "python" in prompt.lower() and re.search(r"\bdef |```", answer):
            code = "\n".join(re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)) or answer
            try:
                compile(code, task_id, "exec")
            except SyntaxError as exc:
                warnings.append(f"{task_id}: python answer has a SyntaxError ({exc.msg})")
        limit = re.search(r"(?:at most|maximum of|no more than)\s+(\d+)\s+words", prompt, re.I)
        if limit and len(answer.split()) > int(limit.group(1)):
            warnings.append(f"{task_id}: answer is {len(answer.split())} words, limit {limit.group(1)}")
        if re.search(r"[a-zA-Z,]$", answer.rstrip()) and not answer.rstrip().endswith("```"):
            warnings.append(f"{task_id}: answer may be truncated (ends mid-sentence)")
    return warnings


def parse_agent_summary(stderr: str) -> dict[str, Any]:
    match = SUMMARY_RE.search(stderr)
    if not match:
        return {
            "found": False,
            "seconds": None,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "by_category": {},
        }
    by_category: dict[str, int]
    try:
        parsed = ast.literal_eval(match.group("by_category"))
        by_category = parsed if isinstance(parsed, dict) else {}
    except (SyntaxError, ValueError):
        by_category = {}
    return {
        "found": True,
        "seconds": float(match.group("seconds")),
        "total_tokens": int(match.group("total")),
        "prompt_tokens": int(match.group("prompt")),
        "completion_tokens": int(match.group("completion")),
        "by_category": by_category,
    }


def build_run_report(
    tasks: list[dict[str, str]],
    results: Any,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    timed_out: bool = False,
) -> dict[str, Any]:
    return {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "results_validation": validate_results(results, tasks),
        "token_summary": parse_agent_summary(stderr),
    }
