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
import re
import sys
import time

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TASKS_FILE = os.environ.get("TASKS_FILE", "/input/tasks.json")
RESULTS_FILE = os.environ.get("RESULTS_FILE", "/output/results.json")

# Leave headroom inside the 10-minute harness limit for startup + file I/O.
MAX_RUNTIME_SECONDS = float(os.environ.get("MAX_RUNTIME_SECONDS", "510"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "10"))
# 30s balances fast failure detection against double-billing: a timed-out
# request may still complete server-side and its tokens still get recorded.
PER_CALL_TIMEOUT = float(os.environ.get("PER_CALL_TIMEOUT", "30"))
MAX_ATTEMPTS = 3

# Per-category behaviour. Instructions are appended to the task prompt as a
# single user message (no system message — some models ignore or re-bill it,
# and every token counts). max_tokens caps runaway verbosity; if a response
# is cut off we retry once with double the cap so the judge sees a complete
# answer.
CATEGORIES = {
    "factual": {
        "instruction": "Answer concisely and accurately in 2-4 sentences.",
        "max_tokens": 220,
        "tier": "easy",
    },
    "math": {
        "instruction": "Solve step by step but keep each step brief. End with 'Answer: <result>'.",
        "max_tokens": 350,
        "tier": "reason",
    },
    "sentiment": {
        "instruction": "State the sentiment label, then justify it in one sentence.",
        "max_tokens": 80,
        "tier": "easy",
    },
    "summary": {
        "instruction": "Follow the requested length and format exactly. Output only the summary.",
        "max_tokens": 180,
        "tier": "easy",
    },
    "ner": {
        "instruction": "Extract the entities with their types. Output only the labelled list, nothing else.",
        "max_tokens": 200,
        "tier": "easy",
    },
    "code_debug": {
        "instruction": "If the code has a bug, state it in one or two sentences, then give the corrected code. No extra commentary.",
        "max_tokens": 600,
        "tier": "code",
    },
    "logic": {
        "instruction": "Reason through the constraints briefly, then clearly state the final answer.",
        "max_tokens": 450,
        "tier": "reason",
    },
    "codegen": {
        "instruction": "Output only the code with a minimal docstring. No explanation before or after.",
        "max_tokens": 600,
        "tier": "code",
    },
}

# Preferred model name fragments per tier, best first. Resolved against the
# runtime ALLOWED_MODELS list (IDs may carry an accounts/.../models/ prefix).
# minimax-m3 is a reasoning model — its thinking tokens bill as completion
# tokens, so it is deliberately last everywhere.
TIER_PREFERENCES = {
    "easy": ["gemma-4-26b-a4b-it", "gemma-4-31b-it", "gemma-4-31b-it-nvfp4", "kimi-k2p7-code"],
    "reason": ["gemma-4-31b-it", "gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it", "kimi-k2p7-code"],
    "code": ["kimi-k2p7-code", "gemma-4-31b-it", "gemma-4-31b-it-nvfp4", "gemma-4-26b-a4b-it"],
}

# ---------------------------------------------------------------------------
# Zero-token task classifier
# ---------------------------------------------------------------------------

CODE_SNIPPET_RE = re.compile(
    r"```|\bdef \w+\(|\bfunction\s+\w*\(|=>\s*{|\breturn\b.*;|#include\s*<|\bpublic\s+(static|class)\b"
)
BUG_WORDS_RE = re.compile(r"\b(bug|debug|fix|error|wrong|incorrect|broken|fault|doesn'?t work|fails?)\b")
CODEGEN_RE = re.compile(
    r"\b(write|implement|create|build)\b.{0,40}"
    r"\b(function|method|class|script|program|code|helper|utility|routine|algorithm|api|endpoint)\b"
)
MATH_RE = re.compile(
    r"\b(calculate|compute|how (much|many)|percent|percentage|total cost|average|"
    r"sum of|profit|interest|discount|per (hour|day|week|month|year)|projection)\b|%"
)
LOGIC_RE = re.compile(
    r"\b(puzzle|riddle|deduce|deduction|constraints?|clues?|exactly one|"
    r"who (is|has|owns|lives|sits)|seated|seating|arrange|logically|must be true|"
    r"each (has?|have|owns?)? ?a different|sits? (immediately )?(to the )?(left|right|next)|"
    r"either end|in a row)\b"
)
NER_RE = re.compile(
    r"\bentit(y|ies)\b|named entity|\bextract\b.{0,80}\b(person|people|organi[sz]ation|location|date)s?\b"
)


def classify(prompt: str) -> str:
    p = prompt.lower()
    if "sentiment" in p or re.search(r"\b(positive|negative|neutral)\b.{0,30}\bclassif", p):
        return "sentiment"
    if re.search(r"\bsummari[sz]e|\bsummary\b|\bcondense\b|tl;?dr", p):
        return "summary"
    if NER_RE.search(p):
        return "ner"
    if CODE_SNIPPET_RE.search(prompt):
        # Explicit codegen phrasing wins even when bug-adjacent words appear
        # ("write a function that raises an error if ..."); bare code or
        # bug wording without a spec means debugging.
        if CODEGEN_RE.search(p):
            return "codegen"
        return "code_debug"
    if CODEGEN_RE.search(p):
        return "codegen"
    if LOGIC_RE.search(p):
        return "logic"
    numbers = len(re.findall(r"\d[\d,.]*", prompt))
    if MATH_RE.search(p) or numbers >= 4:
        return "math"
    return "factual"


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def resolve_models(allowed: list[str]) -> dict[str, str]:
    """Map each tier to a concrete allowed model ID."""

    def find(fragment: str) -> str | None:
        # Exact / suffix match first so 'gemma-4-31b-it' does not grab the
        # '-nvfp4' variant; substring match as a fallback.
        for m in allowed:
            if m == fragment or m.endswith("/" + fragment):
                return m
        for m in allowed:
            if fragment in m:
                return m
        return None

    resolved = {}
    for tier, prefs in TIER_PREFERENCES.items():
        model = next((found for p in prefs if (found := find(p))), None)
        resolved[tier] = model or allowed[0]
    return resolved


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def clean_answer(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text or "").strip()


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


async def answer_task(
    client: AsyncOpenAI,
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

    api_key = os.environ["FIREWORKS_API_KEY"]
    base_url = os.environ["FIREWORKS_BASE_URL"]
    allowed = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    if not allowed:
        print("ALLOWED_MODELS is empty", file=sys.stderr)
        return 1

    with open(TASKS_FILE, encoding="utf-8") as f:
        tasks = json.load(f)
    task_ids = [t.get("task_id", "") for t in tasks]

    models = resolve_models(allowed)
    fallback_model = allowed[0]
    print(f"models per tier: {models}", file=sys.stderr)

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
        await client.close()

    total = stats.prompt_tokens + stats.completion_tokens
    print(
        f"done in {time.monotonic() - start:.1f}s | tokens: {total} "
        f"(prompt {stats.prompt_tokens}, completion {stats.completion_tokens}) | "
        f"by category: {stats.by_category}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
