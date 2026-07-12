"""Generate a synthetic distillation dataset for the local model via Fireworks.

Distills onto the task DISTRIBUTION (category styles, the router's terse
output contract), never onto eval samples: prompts are teacher-generated
from topic seeds, deduped, and decontaminated against tests/sample_tasks.json.
Teacher answers use the exact production category instructions so the
student learns the same output format the router requests at grading time.

Pilot 250/category; auto-scales to 1250/category when the pilot is clean
(filter pass rate >= 98%, zero decontamination hits). Dev-time tokens are
billed to the dev account and never touch the submission score.

Usage:  python3 bench/make_distill_set.py [--pilot-only] [--per-category 250]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_heal  # noqa: E402
from fireworks_client import FireworksClient, ReasoningEffort, create_chat  # noqa: E402
from router_core import DEFAULT_CATEGORIES, load_dotenv  # noqa: E402

TEACHER = "accounts/fireworks/models/kimi-k2p6"
PROMPT_BATCH = 25
# Categories whose prompts embed long passages/code overflow the generation
# cap at 25/batch (truncated JSON array -> parse failure -> starved category,
# seen as summary keeping 50/250 in the first pilot). Smaller batches fit.
CATEGORY_BATCH = {"summary": 8, "ner": 10, "code_debug": 12}
PROMPT_GEN_MAX_TOKENS = 4000
PILOT_PER_CAT = 250
FULL_PER_CAT = 1250
SCALE_PASS_RATE = 0.98
EVAL_FRACTION = 0.05
# Reasoning stays ON for categories where teacher correctness needs it;
# easy categories use effort "none" (cheaper, same quality).
REASONING_CATEGORIES = {"math", "logic", "code_debug", "codegen"}
ANSWER_MAX_TOKENS = {"math": 1200, "logic": 1200, "code_debug": 1400, "codegen": 1400}

TOPICS = {
    "factual": ["photosynthesis", "HTTP and the web", "inflation", "plate tectonics", "vaccines", "black holes",
                "the water cycle", "supply and demand", "DNS", "batteries", "antibiotics", "GPS", "the UN",
                "machine learning", "monetary policy", "the immune system", "semiconductors", "ocean currents"],
    "math": ["percentage discounts", "compound interest", "unit conversion", "rates and speeds", "profit margins",
             "averages and weighted means", "ratios in recipes", "tax calculations", "loan payments",
             "area and volume", "probability of dice/cards", "work-rate problems", "currency exchange"],
    "sentiment": ["product reviews", "restaurant reviews", "tweets about a service", "movie reviews",
                  "customer support transcripts", "app store feedback", "hotel reviews", "neutral announcements"],
    "summary": ["news articles", "meeting notes", "scientific abstracts", "product announcements",
                "historical passages", "policy memos", "sports recaps", "technical blog posts"],
    "ner": ["business news", "sports reports", "political news", "science announcements", "entertainment news",
            "financial reports", "travel articles", "press releases"],
    "code_debug": ["off-by-one errors", "mutable default arguments", "wrong comparison operators",
                   "scope/closure bugs", "string vs int confusion", "incorrect loop bounds", "None handling",
                   "wrong dict key access", "shadowed variables", "bad recursion base cases"],
    "logic": ["seating arrangements", "who-owns-what puzzles", "scheduling constraints", "truth-tellers and liars",
              "ordering/ranking puzzles", "family relationships", "allocation puzzles"],
    "codegen": ["string manipulation", "list/dict processing", "date handling", "simple parsers",
                "math utilities", "file-path helpers", "validation functions", "simple classes", "sorting/filtering"],
}

PROMPT_GEN_TEMPLATE = (
    "Generate {n} diverse, self-contained '{category}' tasks a user might ask an AI assistant. "
    "Theme: {topic}. Style guidance: {style}. Vary difficulty, phrasing, and length. "
    "Each task must be fully answerable from its own text (include any needed passage/code/numbers inline). "
    "Return ONLY a JSON array of {n} strings."
)
STYLE = {
    "factual": "questions asking to explain a concept, definition, or how something works",
    "math": "word problems with concrete numbers requiring calculation",
    "sentiment": "include a short realistic text and ask for its sentiment classification",
    "summary": "include a 100-200 word passage and ask for a summary of a specific length/format",
    "ner": "include 2-4 sentences of realistic prose and ask to extract named entities with types "
           "(person, organization, location, date)",
    "code_debug": "include a short Python function (5-15 lines) containing exactly one realistic bug and ask to find and fix it",
    "logic": "constraint puzzles with 3-5 entities and a single determinable answer",
    "codegen": "ask to write a small Python function with a precise spec and example input/output",
}


NARRATION_RE = re.compile(r"^(the user wants|we need to|let'?s |okay|first,? (i|we) )", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```")


def conform_answer(category: str, text: str) -> str | None:
    """Enforce the terse output contract on teacher answers, or reject.

    The first student fine-tune learned the teacher's narration preambles
    ("The user wants a Python function...") verbatim, blew the category
    token caps at inference, and regressed vs the stock model (11/24
    escalations). Training targets must look exactly like what the router
    wants emitted.
    """
    text = text.strip()
    if category in {"codegen", "code_debug"} and NARRATION_RE.match(text):
        fence = CODE_FENCE_RE.search(text)
        if not fence:
            return None
        text = text[fence.start():].strip()  # drop narration before the code
    elif NARRATION_RE.match(text):
        return None
    if category == "math" and "answer:" not in text.lower():
        return None  # instruction demands a final "Answer: <result>" line
    spec = DEFAULT_CATEGORIES[category]
    if len(text) // 3 > spec["max_tokens"] * 1.5:
        return None  # too verbose to fit the router's cap at inference
    return text or None


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def contaminated(prompt: str, eval_norms: list[str]) -> bool:
    n = normalize(prompt)
    return any(e[:80] in n or n[:80] in e for e in eval_norms)


def parse_json_array(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [p.strip() for p in parsed if isinstance(p, str) and len(p.strip()) > 20]


async def call_teacher(client, effort, sem, *, content: str, max_tokens: int, temperature: float, stats: dict) -> str:
    async with sem:
        for attempt in range(4):
            try:
                resp = await asyncio.wait_for(
                    create_chat(client, effort, model=TEACHER,
                                messages=[{"role": "user", "content": content}],
                                max_tokens=max_tokens, temperature=temperature),
                    timeout=90,
                )
                usage = getattr(resp, "usage", None)
                stats["tokens"] += (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001 - backoff then retry
                if attempt == 3:
                    print(f"teacher call failed for good: {exc!r}", file=sys.stderr)
                    return ""
                await asyncio.sleep(2 ** attempt + random.random())
    return ""


async def generate_category(client, sem, category: str, target: int, eval_norms: list[str],
                            seen: set[str], stats: dict) -> list[dict]:
    effort_none = ReasoningEffort()
    effort = None if category in REASONING_CATEGORIES else effort_none
    spec = DEFAULT_CATEGORIES[category]
    examples: list[dict] = []
    topics = TOPICS[category]
    cstats = stats["categories"].setdefault(category, {"generated": 0, "dedup_dropped": 0,
                                                       "contaminated": 0, "filter_failed": 0, "kept": 0})
    batch = CATEGORY_BATCH.get(category, PROMPT_BATCH)
    round_idx = 0
    while cstats["kept"] < target and round_idx < (target // batch + 6):
        topic = topics[round_idx % len(topics)]
        round_idx += 1
        gen = await call_teacher(client, effort_none, sem,
                                 content=PROMPT_GEN_TEMPLATE.format(n=batch, category=category,
                                                                    topic=topic, style=STYLE[category]),
                                 max_tokens=PROMPT_GEN_MAX_TOKENS, temperature=0.9, stats=stats)
        prompts = parse_json_array(gen)
        cstats["generated"] += len(prompts)

        fresh = []
        for prompt in prompts:
            key = hashlib.sha1(normalize(prompt)[:200].encode()).hexdigest()
            if key in seen:
                cstats["dedup_dropped"] += 1
                continue
            if contaminated(prompt, eval_norms):
                cstats["contaminated"] += 1
                continue
            seen.add(key)
            fresh.append(prompt)
        fresh = fresh[: target - cstats["kept"]]

        async def answer(prompt: str) -> None:
            content = f"{prompt}\n\n{spec['instruction']}"
            text = await call_teacher(client, effort, sem, content=content,
                                      max_tokens=ANSWER_MAX_TOKENS.get(category, spec["max_tokens"]),
                                      temperature=0, stats=stats)
            text = conform_answer(category, text.strip() and text)
            if not text or self_heal.has_hard(self_heal.verify(prompt, text, category)):
                cstats["filter_failed"] += 1
                return
            examples.append({"messages": [{"role": "user", "content": content},
                                          {"role": "assistant", "content": text}],
                             "category": category})
            cstats["kept"] += 1

        await asyncio.gather(*[answer(p) for p in fresh])
        print(f"[{category}] kept {cstats['kept']}/{target} (tokens so far: {stats['tokens']:,})", file=sys.stderr)
    return examples


async def run(args: argparse.Namespace) -> int:
    env = load_dotenv(ROOT / ".env", os.environ.copy())
    if not (env.get("FIREWORKS_API_KEY") and env.get("FIREWORKS_BASE_URL")):
        print("set FIREWORKS_API_KEY and FIREWORKS_BASE_URL first", file=sys.stderr)
        return 1
    eval_tasks = json.loads((ROOT / "tests" / "sample_tasks.json").read_text(encoding="utf-8"))
    eval_norms = [normalize(t["prompt"]) for t in eval_tasks]

    client = FireworksClient(api_key=env["FIREWORKS_API_KEY"], base_url=env["FIREWORKS_BASE_URL"])
    sem = asyncio.Semaphore(args.concurrency)
    stats: dict = {"tokens": 0, "categories": {}}
    seen: set[str] = set()
    random.seed(7)

    async def build(per_cat: int) -> list[dict]:
        results = await asyncio.gather(
            *[generate_category(client, sem, cat, per_cat, eval_norms, seen, stats) for cat in TOPICS]
        )
        return [ex for group in results for ex in group]

    examples = await build(args.per_category)

    kept = sum(c["kept"] for c in stats["categories"].values())
    attempted = kept + sum(c["filter_failed"] for c in stats["categories"].values())
    pass_rate = kept / attempted if attempted else 0.0
    contamination = sum(c["contaminated"] for c in stats["categories"].values())
    print(f"pilot: kept {kept}, answer pass rate {pass_rate:.3f}, contamination hits {contamination}", file=sys.stderr)

    scaled = False
    if not args.pilot_only and pass_rate >= SCALE_PASS_RATE and contamination == 0:
        scaled = True
        print("pilot clean -> scaling to full size", file=sys.stderr)
        # kept-counts persist per category, so pass the absolute target:
        # each category tops up from pilot level to full_per_category.
        examples += await build(args.full_per_category)
    elif not args.pilot_only:
        print("pilot NOT clean -> skipping auto-scale; inspect stats and rerun", file=sys.stderr)

    random.shuffle(examples)
    n_eval = max(1, int(len(examples) * EVAL_FRACTION))
    out_dir = ROOT / "out" / "distill"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("eval", examples[:n_eval]), ("train", examples[n_eval:])):
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats.update({"total": len(examples), "train": len(examples) - n_eval, "eval": n_eval,
                  "answer_pass_rate": round(pass_rate, 4), "scaled_to_full": scaled})
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    await client.close()
    print(json.dumps({k: v for k, v in stats.items() if k != "categories"}, indent=2))
    print(f"dataset written to {out_dir}/train.jsonl and eval.jsonl")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-category", type=int, default=PILOT_PER_CAT)
    p.add_argument("--full-per-category", type=int, default=FULL_PER_CAT)
    p.add_argument("--pilot-only", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
