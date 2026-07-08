"""Frugal Router — token-minimal general-purpose agent for the Fireworks track.

Reads /input/tasks.json, classifies each task locally (zero API tokens),
answers easy categories with a bundled local model when possible (zero
Fireworks tokens), and routes the rest to the cheapest suitable allowed
model with a terse category prompt. Writes /output/results.json.
Optimised for the scoring rule: pass the accuracy gate, then rank
ascending by total Fireworks tokens.
"""

import asyncio
import json
import os
import random
import sys
import time
from typing import Any

from fireworks_client import FireworksClient
from local_engine import LocalEngine
from router_core import clean_answer, classify, load_runtime_config, parse_allowed_models, resolve_models

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
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "").strip()
LOCAL_MODEL_CATEGORIES = {
    c.strip()
    for c in os.environ.get("LOCAL_MODEL_CATEGORIES", "factual,sentiment,summary,ner").split(",")
    if c.strip()
}
# Time kept free for Fireworks fallbacks when deciding whether a local
# generation still fits in the budget.
FIREWORKS_RESERVE_SECONDS = float(os.environ.get("FIREWORKS_RESERVE_SECONDS", "60"))


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------

class ModelPool:
    """Tier→model mapping over ALLOWED_MODELS; models that 404 get dropped."""

    def __init__(self, allowed: list[str]):
        self.allowed = list(allowed)
        self.tiers = resolve_models(self.allowed, CONFIG)

    def model_for(self, tier: str) -> str:
        return self.tiers.get(tier) or self.fallback()

    def fallback(self) -> str:
        return self.tiers.get("easy") or self.allowed[0]

    def disqualify(self, model: str) -> bool:
        """Drop a model that the API rejected; keep at least one in the pool."""
        if model not in self.allowed or len(self.allowed) <= 1:
            return False
        self.allowed.remove(model)
        self.tiers = resolve_models(self.allowed, CONFIG)
        print(f"model disqualified: {model}; tiers now {self.tiers}", file=sys.stderr)
        return True


def is_model_error(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 404:
        return True
    text = str(exc).lower()
    return "model_not_found" in text or ("model" in text and ("not found" in text or "does not exist" in text))


class Stats:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.local_answers = 0
        self.by_category: dict[str, int] = {}

    def add(self, category: str, usage) -> None:
        if not usage:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.by_category[category] = self.by_category.get(category, 0) + pt + ct


# ---------------------------------------------------------------------------
# Async workers
# ---------------------------------------------------------------------------

async def answer_task(
    client: Any,
    sem: asyncio.Semaphore,
    engine: LocalEngine | None,
    task: dict,
    pool: ModelPool | None,
    deadline: float,
    stats: Stats,
) -> tuple[str, str]:
    task_id = task["task_id"]
    prompt = task["prompt"]
    category = classify(prompt)
    spec = CATEGORIES.get(category) or CATEGORIES["factual"]
    content = f"{prompt}\n\n{spec['instruction']}"
    max_tokens = spec["max_tokens"]
    best = ""  # best answer seen so far — never return "" if we paid for text

    # Local-first path: zero Fireworks tokens. The engine skips itself when
    # the remaining time budget (minus the Fireworks reserve) is too small.
    if engine is not None and category in LOCAL_MODEL_CATEGORIES:
        reserve = FIREWORKS_RESERVE_SECONDS if client is not None else 5.0
        local = await engine.generate(content, max_tokens, deadline - reserve)
        if local is not None:
            text, finish = local
            answer = clean_answer(text)
            if answer and finish == "stop":
                stats.local_answers += 1
                print(f"[{task_id}] answered locally ({category})", file=sys.stderr)
                return task_id, answer
            best = answer
            if answer:
                print(f"[{task_id}] local answer suspect (finish={finish}); falling back", file=sys.stderr)

    if client is None or pool is None:
        if not best:
            print(f"[{task_id}] no Fireworks fallback configured", file=sys.stderr)
        return task_id, best

    async with sem:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            remaining = deadline - time.monotonic()
            if remaining < 5:
                return task_id, best
            model = pool.model_for(spec["tier"])
            # Last attempt swaps to the fallback model in case the primary
            # endpoint is the thing failing.
            use_model = pool.fallback() if (attempt == MAX_ATTEMPTS and pool.fallback() != model) else model
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
                if len(text) > len(best):
                    best = text
                # Truncated answers read as wrong to the judge — one retry
                # with a doubled cap if time allows; the truncated text is
                # kept as `best` in case the retry fails too.
                if choice.finish_reason == "length" and max_tokens < 1600 and deadline - time.monotonic() > 30:
                    max_tokens *= 2
                    continue
                if text:
                    return task_id, text
            except Exception as exc:  # noqa: BLE001 - retry on any transport/API error
                print(f"[{task_id}] attempt {attempt} failed: {exc!r}", file=sys.stderr)
                # A rejected model will fail every retry identically — drop it
                # from the pool and retry immediately with the replacement.
                if is_model_error(exc) and pool.disqualify(use_model):
                    continue
            await asyncio.sleep(min(2**attempt + random.random(), 8))
    return task_id, best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def normalize_tasks(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("tasks.json must be a JSON array")
    tasks: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            print(f"skipping task {index}: not an object", file=sys.stderr)
            continue
        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            print(f"skipping task {index}: missing task_id", file=sys.stderr)
            continue
        prompt = item.get("prompt")
        tasks.append({"task_id": task_id, "prompt": prompt if isinstance(prompt, str) else ""})
    return tasks


def write_results(results: dict[str, str], task_ids: list[str]) -> None:
    seen: set[str] = set()
    ordered = [tid for tid in task_ids if not (tid in seen or seen.add(tid))]
    payload = [{"task_id": tid, "answer": results.get(tid, "")} for tid in ordered]
    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, RESULTS_FILE)


async def main() -> int:
    start = time.monotonic()
    deadline = start + MAX_RUNTIME_SECONDS

    try:
        with open(TASKS_FILE, encoding="utf-8") as f:
            tasks = normalize_tasks(json.load(f))
    except Exception as exc:  # noqa: BLE001 - unreadable input is catastrophic
        print(f"failed to read {TASKS_FILE}: {exc!r}", file=sys.stderr)
        write_results({}, [])
        return 1
    task_ids = [t["task_id"] for t in tasks]

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "")
    allowed = parse_allowed_models(os.environ.get("ALLOWED_MODELS", ""))
    fireworks_ready = bool(api_key and base_url and allowed)

    engine = LocalEngine.create(LOCAL_MODEL_PATH)

    if not fireworks_ready and engine is None:
        print("no Fireworks configuration and no local model — cannot answer anything", file=sys.stderr)
        write_results({}, task_ids)
        return 1

    client = None
    pool = None
    if fireworks_ready:
        try:
            from openai import AsyncOpenAI
        except ModuleNotFoundError:
            print("Python package 'openai' is not installed; using stdlib Fireworks client", file=sys.stderr)
            client = FireworksClient(api_key=api_key, base_url=base_url)
        else:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)
        pool = ModelPool(allowed)
        print(f"models per tier: {pool.tiers}", file=sys.stderr)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    stats = Stats()

    results: dict[str, str] = {}
    try:
        coros = [answer_task(client, sem, engine, t, pool, deadline, stats) for t in tasks]
        outcomes = await asyncio.gather(*coros, return_exceptions=True)
        for task, outcome in zip(tasks, outcomes):
            if isinstance(outcome, BaseException):
                print(f"[{task['task_id']}] unhandled error: {outcome!r}", file=sys.stderr)
                results.setdefault(task["task_id"], "")
            else:
                tid, answer = outcome
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
    print(f"local answers: {stats.local_answers}/{len(task_ids)}", file=sys.stderr)
    if empty_task_ids:
        # Empty answers lose those tasks at the accuracy gate but must NOT
        # fail the container: a non-zero exit turns a partial score into
        # RUNTIME_ERROR for the whole submission.
        print(f"empty answers: {empty_task_ids}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
