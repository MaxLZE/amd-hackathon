"""Judged accuracy of a local GGUF model on the production local path.

Runs tasks through the exact agent.py local pipeline (classify -> category
instruction -> cap -> clean_answer -> self_heal suspect logic), then has a
Fireworks judge compare non-suspect answers against premium reference
answers (from a bench/benchmark.py report).

Usage:
  python3 bench/local_eval.py --model ~/models/local.gguf \
      --refs out/benchmark_report_effort.json [--categories factual,ner] [--out report.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_heal  # noqa: E402
from fireworks_client import FireworksClient, ReasoningEffort, create_chat  # noqa: E402
from local_engine import LocalEngine  # noqa: E402
from router_core import classify, clean_answer, condense_prompt, load_dotenv, load_runtime_config  # noqa: E402

JUDGE_PROMPT = (
    "You are grading a budget model against a premium model.\n"
    "Task:\n{prompt}\n\nPremium answer:\n{reference}\n\nBudget answer:\n{candidate}\n\n"
    "Does the budget answer contain the same essential content and the same "
    "final result/conclusion as the premium answer? Ignore differences in "
    "length, tone, and formatting. Reply with exactly one word: YES or NO."
)


async def run(args: argparse.Namespace) -> int:
    env = load_dotenv(ROOT / ".env", os.environ.copy())
    config = load_runtime_config(None, env)
    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    report = json.loads(Path(args.refs).read_text(encoding="utf-8"))
    references = {r["task_id"]: r["reference"]["answer"] for r in report["tasks"] if r["reference"].get("ok")}
    wanted = {c.strip() for c in args.categories.split(",") if c.strip()} if args.categories else None

    os.environ["LOCAL_MODEL_THREADS"] = str(args.threads)
    engine = LocalEngine.create(os.path.expanduser(args.model))
    if engine is None:
        print("could not load local model", file=sys.stderr)
        return 1

    rows = []
    deadline = asyncio.get_event_loop().time() + 10**6
    for t in tasks:
        category = classify(t["prompt"])
        if (wanted and category not in wanted) or t["task_id"] not in references:
            continue
        spec = config.categories.get(category) or config.categories["factual"]
        content = f"{condense_prompt(t['prompt'])}\n\n{spec['instruction']}"
        text, finish = await engine.generate(content, spec["max_tokens"], deadline)
        answer = clean_answer(text)
        issues = self_heal.verify(t["prompt"], answer, category)
        fixed = self_heal.local_fix(t["prompt"], answer, issues)
        if fixed is not None:
            answer = fixed
            issues = self_heal.verify(t["prompt"], answer, category)
        suspect = finish != "stop" or self_heal.has_hard(issues)
        rows.append({"task_id": t["task_id"], "category": category, "prompt": t["prompt"],
                     "answer": answer, "finish": finish, "suspect": suspect, "match": None})

    client = FireworksClient(api_key=env["FIREWORKS_API_KEY"], base_url=env["FIREWORKS_BASE_URL"])
    effort = ReasoningEffort()
    judge = args.judge if "/" in args.judge else f"accounts/fireworks/models/{args.judge}"
    sem = asyncio.Semaphore(args.concurrency)

    async def judge_row(row: dict) -> None:
        if row["suspect"]:
            return  # agent.py escalates these to Fireworks anyway
        content = JUDGE_PROMPT.format(prompt=row["prompt"], reference=references[row["task_id"]], candidate=row["answer"])
        async with sem:
            resp = await create_chat(client, effort, model=judge,
                                     messages=[{"role": "user", "content": content}], max_tokens=4, temperature=0)
        word = (resp.choices[0].message.content or "").strip().upper()
        row["match"] = word.startswith("YES") if word.startswith(("YES", "NO")) else None

    await asyncio.gather(*[judge_row(r) for r in rows])
    await client.close()

    by_cat: dict[str, dict[str, int]] = {}
    print(f"{'task':10} {'cat':10} {'suspect':7} match  answer[:60]")
    for r in rows:
        mark = {True: "YES", False: "NO", None: "-"}[r["match"]]
        print(f"{r['task_id']:10} {r['category']:10} {str(r['suspect']):7} {mark:5}  {r['answer'][:60]!r}")
        c = by_cat.setdefault(r["category"], {"tasks": 0, "escalated": 0, "yes": 0, "judged": 0})
        c["tasks"] += 1
        if r["suspect"]:
            c["escalated"] += 1
        elif r["match"] is not None:
            c["judged"] += 1
            c["yes"] += int(r["match"])
    print()
    for cat, c in sorted(by_cat.items()):
        print(f"{cat}: {c['tasks']} tasks, {c['escalated']} escalated, judged accuracy {c['yes']}/{c['judged']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"model": args.model, "by_category": by_cat, "rows": rows}, indent=2), encoding="utf-8")
        print(f"report written to {args.out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="~/models/local.gguf")
    p.add_argument("--tasks", default=str(ROOT / "tests" / "sample_tasks.json"))
    p.add_argument("--refs", default=str(ROOT / "out" / "benchmark_report_effort.json"))
    p.add_argument("--judge", default="kimi-k2p6")
    p.add_argument("--categories", default="", help="comma list; empty = all")
    p.add_argument("--threads", type=int, default=8, help="quality is thread-independent; more = faster")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="")
    return asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
