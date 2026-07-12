"""Benchmark the Frugal Router against a high-tier reference model.

Measures, per task and in aggregate:
  1. token usage        - prompt/completion tokens from API usage fields
  2. reply speed        - wall-clock latency and completion tokens/sec
  3. content accuracy   - LLM-judged: does the routed answer convey the same
                          essential content/result as the high-tier answer?

The router side reuses the exact production code path (classify -> tier ->
resolve_models -> terse instruction + max_tokens cap). The reference side
sends the raw prompt to one high-tier model with a generous cap.

Usage (from the repo root):
  python3 bench/benchmark.py                         # tests/sample_tasks.json
  python3 bench/benchmark.py --tasks path/to.json --reference kimi-k2p6 \
      --judge glm-5p2 --concurrency 4 --out out/benchmark_report.json

Env: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS (same as agent.py).
Only the stdlib client is used, so no extra dependencies are required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import self_heal  # noqa: E402
from fireworks_client import FireworksClient, ReasoningEffort, create_chat  # noqa: E402
from frugal_checks import deterministic_warnings  # noqa: E402
from router_core import (  # noqa: E402
    classify,
    clean_answer,
    condense_prompt,
    load_dotenv,
    load_runtime_config,
    parse_allowed_models,
    resolve_models,
)

REFERENCE_MAX_TOKENS = 1600
JUDGE_PROMPT = (
    "You are grading a budget model against a premium model.\n"
    "Task:\n{prompt}\n\n"
    "Premium answer:\n{reference}\n\n"
    "Budget answer:\n{candidate}\n\n"
    "Does the budget answer contain the same essential content and the same "
    "final result/conclusion as the premium answer? Ignore differences in "
    "length, tone, and formatting. Reply with exactly one word: YES or NO."
)


def find_model(allowed: list[str], fragment: str) -> str | None:
    for model in allowed:
        if model == fragment or model.endswith("/" + fragment):
            return model
    return next((model for model in allowed if fragment in model), None)


async def timed_call(client: FireworksClient, effort: ReasoningEffort | None = None, **kwargs) -> dict[str, Any]:
    started = time.monotonic()
    try:
        resp = await create_chat(client, effort, **kwargs)
    except Exception as exc:  # noqa: BLE001 - record the failure, keep benching
        return {"ok": False, "error": repr(exc), "latency_s": round(time.monotonic() - started, 2)}
    latency = time.monotonic() - started
    usage = getattr(resp, "usage", None)
    completion = getattr(usage, "completion_tokens", 0) or 0
    return {
        "ok": True,
        "answer": clean_answer(resp.choices[0].message.content),
        "finish_reason": resp.choices[0].finish_reason,
        "latency_s": round(latency, 2),
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": completion,
        "tokens_per_s": round(completion / latency, 1) if latency > 0 else None,
    }


async def bench_task(
    client: FireworksClient,
    sem: asyncio.Semaphore,
    task: dict[str, str],
    tiers: dict[str, str],
    categories: dict[str, dict[str, Any]],
    reference_model: str,
    heal: bool = False,
    effort: ReasoningEffort | None = None,
) -> dict[str, Any]:
    prompt = task["prompt"]
    category = classify(prompt)
    spec = categories.get(category) or categories["factual"]
    router_model = tiers.get(spec["tier"]) or next(iter(tiers.values()))

    async with sem:
        routed = await timed_call(
            client,
            effort,
            model=router_model,
            messages=[{"role": "user", "content": f"{condense_prompt(prompt)}\n\n{spec['instruction']}"}],
            max_tokens=spec["max_tokens"],
            temperature=0,
        )
    routed["heal"] = None
    if heal and routed.get("ok"):
        issues = self_heal.verify(prompt, routed["answer"], category)
        fixed = self_heal.local_fix(prompt, routed["answer"], issues) if issues else None
        if fixed is not None:
            routed["answer"], routed["heal"] = fixed, "local-fix"
            issues = self_heal.verify(prompt, fixed, category)
        if self_heal.has_hard(issues):
            content = self_heal.repair_prompt(prompt, routed["answer"], issues)
            async with sem:
                repair = await timed_call(
                    client,
                    effort,
                    model=router_model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=spec["max_tokens"],
                    temperature=0,
                )
            if repair.get("ok"):
                # repair cost counts against the router side
                routed["prompt_tokens"] += repair["prompt_tokens"]
                routed["completion_tokens"] += repair["completion_tokens"]
                routed["latency_s"] = round(routed["latency_s"] + repair["latency_s"], 2)
                if repair["answer"] and not self_heal.has_hard(self_heal.verify(prompt, repair["answer"], category)):
                    routed["answer"], routed["heal"] = repair["answer"], "model-repair"
    async with sem:
        reference = await timed_call(
            client,
            model=reference_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=REFERENCE_MAX_TOKENS,
            temperature=0,
        )
    return {
        "task_id": task["task_id"],
        "category": category,
        "router_model": router_model,
        "routed": routed,
        "reference": reference,
    }


async def judge_task(
    client: FireworksClient,
    sem: asyncio.Semaphore,
    judge_model: str,
    row: dict[str, Any],
    prompt: str,
    effort: ReasoningEffort | None = None,
) -> None:
    routed, reference = row["routed"], row["reference"]
    if not (routed.get("ok") and reference.get("ok")):
        row["match"] = None
        return
    content = JUDGE_PROMPT.format(prompt=prompt, reference=reference["answer"], candidate=routed["answer"])
    async with sem:
        # Without effort suppression the judge burns its 4-token cap on
        # hidden reasoning and returns an empty/garbage verdict.
        verdict = await timed_call(client, effort, model=judge_model, messages=[{"role": "user", "content": content}], max_tokens=4, temperature=0)
    if not verdict.get("ok"):
        row["match"] = None
        return
    word = verdict["answer"].strip().upper()
    row["match"] = True if word.startswith("YES") else False if word.startswith("NO") else None


def aggregate(rows: list[dict[str, Any]], side: str) -> dict[str, Any]:
    ok = [row[side] for row in rows if row[side].get("ok")]
    if not ok:
        return {"calls_ok": 0, "calls_failed": len(rows)}
    latencies = [r["latency_s"] for r in ok]
    return {
        "calls_ok": len(ok),
        "calls_failed": len(rows) - len(ok),
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in ok),
        "prompt_tokens": sum(r["prompt_tokens"] for r in ok),
        "completion_tokens": sum(r["completion_tokens"] for r in ok),
        "avg_latency_s": round(statistics.mean(latencies), 2),
        "p50_latency_s": round(statistics.median(latencies), 2),
        "max_latency_s": max(latencies),
        "avg_tokens_per_s": round(statistics.mean([r["tokens_per_s"] for r in ok if r["tokens_per_s"]]), 1),
    }


async def run(args: argparse.Namespace) -> int:
    env = load_dotenv(ROOT / ".env", os.environ.copy())
    api_key, base_url = env.get("FIREWORKS_API_KEY", ""), env.get("FIREWORKS_BASE_URL", "")
    allowed = parse_allowed_models(env.get("ALLOWED_MODELS", ""))
    if not (api_key and base_url and allowed):
        print("set FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS first", file=sys.stderr)
        return 1

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = [t for t in tasks if isinstance(t, dict) and t.get("task_id") and t.get("prompt")]
    config = load_runtime_config(None, env)
    tiers = resolve_models(allowed, config)

    reference_model = find_model(allowed, args.reference) if args.reference else None
    if reference_model is None:
        # default: the code-tier pick, usually the priciest capable model in the pool
        reference_model = tiers.get("code") or allowed[-1]
    judge_model = (find_model(allowed, args.judge) if args.judge else None) or reference_model

    print(f"tasks: {len(tasks)} | tiers: {tiers}", file=sys.stderr)
    print(f"reference: {reference_model} | judge: {judge_model}", file=sys.stderr)

    client = FireworksClient(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(args.concurrency)
    effort = None if args.no_effort_suppression else ReasoningEffort()
    rows = await asyncio.gather(
        *[bench_task(client, sem, t, tiers, config.categories, reference_model, heal=args.self_heal, effort=effort) for t in tasks]
    )
    await asyncio.gather(*[judge_task(client, sem, judge_model, row, task["prompt"], effort=effort) for row, task in zip(rows, tasks)])

    answers = {row["task_id"]: row["routed"].get("answer", "") for row in rows}
    warnings = deterministic_warnings(tasks, answers)

    judged = [row for row in rows if row.get("match") is not None]
    matches = sum(1 for row in judged if row["match"])
    router_agg, reference_agg = aggregate(rows, "routed"), aggregate(rows, "reference")
    heal_counts: dict[str, int] = {}
    for row in rows:
        kind = row["routed"].get("heal")
        if kind:
            heal_counts[kind] = heal_counts.get(kind, 0) + 1

    report = {
        "task_count": len(tasks),
        "reference_model": reference_model,
        "judge_model": judge_model,
        "self_heal": args.self_heal,
        "heal_counts": heal_counts,
        "router": router_agg,
        "reference": reference_agg,
        "accuracy": {
            "judged": len(judged),
            "matched": matches,
            "match_rate": round(matches / len(judged), 3) if judged else None,
            "deterministic_warnings": warnings,
        },
        "token_savings_vs_reference": (
            round(1 - router_agg["total_tokens"] / reference_agg["total_tokens"], 3)
            if router_agg.get("total_tokens") and reference_agg.get("total_tokens")
            else None
        ),
        "tasks": rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'task':10} {'cat':10} {'model':28} {'tok':>6} {'ref tok':>8} {'lat s':>6} {'ref s':>6}  {'match':5}  heal")
    for row in rows:
        routed, reference = row["routed"], row["reference"]
        rt = routed.get("prompt_tokens", 0) + routed.get("completion_tokens", 0)
        ft = reference.get("prompt_tokens", 0) + reference.get("completion_tokens", 0)
        mark = {True: "YES", False: "NO", None: "-"}[row.get("match")]
        model_short = row["router_model"].rsplit("/", 1)[-1]
        print(
            f"{row['task_id']:10} {row['category']:10} {model_short:28} {rt:>6} {ft:>8} "
            f"{routed.get('latency_s', 0):>6} {reference.get('latency_s', 0):>6}  {mark:5}  {routed.get('heal') or '-'}"
        )

    print(f"\nrouter:    {router_agg}")
    print(f"reference: {reference_agg}")
    print(f"accuracy:  {matches}/{len(judged)} judged equivalent"
          f" ({report['accuracy']['match_rate']}) | deterministic warnings: {len(warnings)}")
    if args.self_heal:
        print(f"self-heal: {heal_counts or 'nothing to repair'}")
    if report["token_savings_vs_reference"] is not None:
        print(f"token savings vs reference: {report['token_savings_vs_reference'] * 100:.1f}%")
    print(f"report written to {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default=str(ROOT / "tests" / "sample_tasks.json"))
    parser.add_argument("--reference", default="", help="high-tier model (fragment of an ALLOWED_MODELS entry)")
    parser.add_argument("--judge", default="", help="judge model; defaults to the reference model")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--self-heal", action="store_true", help="verify answers and repair hard failures (self_heal.py)")
    parser.add_argument("--no-effort-suppression", action="store_true", help="skip reasoning_effort suppression (measure the old behavior)")
    parser.add_argument("--out", default=str(ROOT / "out" / "benchmark_report.json"))
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
