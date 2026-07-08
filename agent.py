"""Frugal Router — token-minimal general-purpose agent for the Fireworks track.

Reads /input/tasks.json, classifies each task locally (zero API tokens),
routes it to the cheapest suitable allowed model with a terse category
prompt, and writes /output/results.json. Optimised for the scoring rule:
pass the accuracy gate, then rank ascending by total tokens.
"""

import asyncio
import json
import os
import random
import shlex
import sys
import time
from typing import Any

from router_core import clean_answer, classify, load_runtime_config, resolve_models

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASKS_FILE = os.environ.get("TASKS_FILE", "/input/tasks.json")
RESULTS_FILE = os.environ.get("RESULTS_FILE", "/output/results.json")
CONFIG = load_runtime_config()

# Leave headroom inside the 10-minute harness limit for startup + file I/O.
MAX_RUNTIME_SECONDS = CONFIG.max_runtime_seconds
MAX_CONCURRENCY = CONFIG.max_concurrency
# 30s balances fast failure detection against double-billing: a timed-out
# request may still complete server-side and its tokens still get recorded.
PER_CALL_TIMEOUT = CONFIG.per_call_timeout
MAX_ATTEMPTS = CONFIG.max_attempts

CATEGORIES = CONFIG.categories
LOCAL_MODEL_COMMAND = os.environ.get("LOCAL_MODEL_COMMAND", "").strip()
LOCAL_MODEL_TIMEOUT = float(os.environ.get("LOCAL_MODEL_TIMEOUT", "8"))
LOCAL_MODEL_CATEGORIES = {
    c.strip()
    for c in os.environ.get("LOCAL_MODEL_CATEGORIES", "factual,sentiment,summary,ner").split(",")
    if c.strip()
}


# ---------------------------------------------------------------------------
# Async workers
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.by_category: dict[str, int] = {}

    def add(self, category: str, usage) -> None:
        if not usage:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.by_category[category] = self.by_category.get(category, 0) + pt + ct


async def try_local_answer(task_id: str, prompt: str, category: str, spec: dict, deadline: float) -> str | None:
    if not LOCAL_MODEL_COMMAND or category not in LOCAL_MODEL_CATEGORIES:
        return None
    if deadline - time.monotonic() < 3:
        return None

    payload = {
        "task_id": task_id,
        "prompt": prompt,
        "category": category,
        "instruction": spec["instruction"],
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            *shlex.split(LOCAL_MODEL_COMMAND),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(json.dumps(payload).encode("utf-8")),
            timeout=min(LOCAL_MODEL_TIMEOUT, max(1, deadline - time.monotonic())),
        )
    except Exception as exc:  # noqa: BLE001 - local model is an optional zero-token path
        print(f"[{task_id}] local model failed: {exc!r}", file=sys.stderr)
        return None

    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        print(f"[{task_id}] local model exited {proc.returncode}: {detail}", file=sys.stderr)
        return None

    text = stdout.decode("utf-8", errors="replace").strip()
    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            text = parsed.get("answer", "") if isinstance(parsed, dict) else text
        except json.JSONDecodeError:
            pass
    answer = clean_answer(text)
    if answer:
        print(f"[{task_id}] answered by local model", file=sys.stderr)
    return answer or None


async def answer_task(
    client: Any,
    sem: asyncio.Semaphore,
    task: dict,
    models: dict[str, str],
    fallback_model: str,
    deadline: float,
    stats: Stats,
) -> tuple[str, str]:
    task_id = task.get("task_id", "")
    prompt = task.get("prompt", "")
    category = classify(prompt)
    spec = CATEGORIES[category]
    model = models[spec["tier"]]
    content = f"{prompt}\n\n{spec['instruction']}"
    max_tokens = spec["max_tokens"]

    async with sem:
        local_answer = await try_local_answer(task_id, prompt, category, spec, deadline)
        if local_answer:
            return task_id, local_answer
        if client is None:
            print(f"[{task_id}] no Fireworks fallback configured", file=sys.stderr)
            return task_id, ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            remaining = deadline - time.monotonic()
            if remaining < 5:
                return task_id, ""
            # Last attempt swaps to the fallback model in case the primary
            # endpoint is the thing failing.
            use_model = fallback_model if (attempt == MAX_ATTEMPTS and fallback_model != model) else model
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=use_model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=max_tokens,
                        temperature=0,
                    ),
                    timeout=min(PER_CALL_TIMEOUT, remaining),
                )
                stats.add(category, getattr(resp, "usage", None))
                choice = resp.choices[0]
                text = clean_answer(choice.message.content)
                # Truncated answers read as wrong to the judge — one retry
                # with a doubled cap if time allows.
                if choice.finish_reason == "length" and max_tokens < 1600 and deadline - time.monotonic() > 30:
                    max_tokens *= 2
                    continue
                if text:
                    return task_id, text
            except Exception as exc:  # noqa: BLE001 - retry on any transport/API error
                print(f"[{task_id}] attempt {attempt} failed: {exc!r}", file=sys.stderr)
            await asyncio.sleep(min(2**attempt + random.random(), 8))
    return task_id, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_results(results: dict[str, str], task_ids: list[str]) -> None:
    payload = [{"task_id": tid, "answer": results.get(tid, "")} for tid in task_ids]
    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, RESULTS_FILE)


async def main() -> int:
    start = time.monotonic()
    deadline = start + MAX_RUNTIME_SECONDS

    with open(TASKS_FILE, encoding="utf-8") as f:
        tasks = json.load(f)
    task_ids = [t.get("task_id", "") for t in tasks]

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    fireworks_ready = bool(api_key and base_url and allowed)
    if not fireworks_ready and not LOCAL_MODEL_COMMAND:
        print("Fireworks env vars are missing and LOCAL_MODEL_COMMAND is not configured", file=sys.stderr)
        write_results({}, task_ids)
        return 1
    if not allowed and not LOCAL_MODEL_COMMAND:
        print("ALLOWED_MODELS is empty", file=sys.stderr)
        write_results({}, task_ids)
        return 1

    model_pool = allowed or ["local-only"]
    models = resolve_models(model_pool, CONFIG)
    fallback_model = model_pool[0]
    print(f"models per tier: {models}", file=sys.stderr)

    client = None
    if fireworks_ready:
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError:
            print("Python package 'openai' is not installed; run python3 -m pip install -r requirements.txt", file=sys.stderr)
            write_results({}, task_ids)
            return 1
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    stats = Stats()

    results: dict[str, str] = {}
    try:
        coros = [answer_task(client, sem, t, models, fallback_model, deadline, stats) for t in tasks]
        for tid, answer in await asyncio.gather(*coros):
            results[tid] = answer
    finally:
        write_results(results, task_ids)
        if client is not None:
            await client.close()

    total = stats.prompt_tokens + stats.completion_tokens
    empty_task_ids = [task_id for task_id in task_ids if not results.get(task_id, "").strip()]
    print(
        f"done in {time.monotonic() - start:.1f}s | tokens: {total} "
        f"(prompt {stats.prompt_tokens}, completion {stats.completion_tokens}) | "
        f"by category: {stats.by_category}",
        file=sys.stderr,
    )
    if empty_task_ids:
        print(f"empty answers: {empty_task_ids}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
