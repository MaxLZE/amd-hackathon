# Token-reduction strategy — Track 1

Goal: minimize **total Fireworks tokens** (prompt + completion, as recorded by
the judging proxy) while staying above the LLM-judge accuracy gate.

## What actually counts

- Every Fireworks call is billed on both sides: retries re-pay the prompt,
  repair prompts re-pay the prompt *plus* the previous answer.
- **Local inference counts as zero.** The single biggest lever is moving work
  onto the bundled Qwen2.5-3B Q4 (already baked into the image at
  `/app/models/local.gguf`, ~1.9 GB, fits the 4 GB / 2 vCPU grading VM).
- Reasoning models bill their thinking tokens. Never route to one
  (`minimax-m3`) without probing `usage.completion_tokens` first.
- Answer caching across runs is **explicitly banned** ("do not hardcode or
  cache answers; evaluation uses unseen prompt variants"). Exit code, valid
  JSON, and the 10-minute cap are hard gates.

## Root cause of the 31.6% submission (fixed 2026-07-12)

Every allowed model bills **hidden reasoning tokens** (60–100 for a
one-sentence answer). With the tight per-category `max_tokens` caps the
model hit the cap *mid-reasoning*, so the visible content was truncated
chain-of-thought garbage — the judge failed those answers. On the 2-vCPU
grading VM the local model mostly skipped itself for lack of time, so
nearly every task took this broken Fireworks path.

Fix: `reasoning_effort: "none"` on every call (`fireworks_client.create_chat`),
with a per-model downgrade ladder none → low → omit for models that 400
("none" works on GLM/Kimi/DeepSeek; gpt-oss takes "low"). Result on the
24-task suite: **24/24 judged equivalent, 3,736 total tokens** (77.9% below
reference). Double win: reasoning was also the dominant completion-token cost.
Note: pre-fix accuracy numbers (including the 87.5% below) are unreliable —
the bench judge itself was truncating its 4-token verdict on hidden reasoning.

Local Qwen2.5-3B judged eval (easy categories): factual 3/3, sentiment 3/3,
summary 3/3, **ner 0/3** — NER removed from `LOCAL_MODEL_CATEGORIES`.

## Strategy ladder, ranked by leverage

Measured baseline (24-task mock suite): router 1,634 tokens @ 87.5% match;
high-tier-everything 2,053. Estimates below are directional, to be re-measured
with `bench/benchmark.py` against the real API.

### 1. Local-first routing (est. −30–45%, the headline move)

Serve easy categories (factual, sentiment, summary, NER) entirely from the
local model — those were 610/1,634 tokens (37%) of the mock baseline. Already
wired (`LOCAL_MODEL_CATEGORIES`); the work is *verification*, not plumbing:
a 3B model passes the gate on these only if every local answer is checked
before acceptance.

- Status: shipped. `self_heal.verify()` now rejects hard-broken local answers
  and escalates them to Fireworks (verify-then-escalate).
- Next: measure the local pass rate per category with the real judge; every
  category above ~90% local pass moves into `LOCAL_MODEL_CATEGORIES`.

### 2. Local verification & repair (est. −5–10%, plus gate insurance)

Anything the local model or plain Python can *check or fix* for free:

- **Code**: syntax-compile (shipped) + sandboxed execution (shipped —
  `self_heal.exec_check`): python answers run in an isolated subprocess;
  crashes and violations of literal examples stated in the prompt
  ("double(2) should return 4", doctest lines) are hard failures. Symbolic
  examples (`f(s) returns True if…`) are never asserted — no false positives.
- **Math**: worked steps like "80 * 0.75 = 60" are recomputed with an
  AST-whitelisted evaluator (shipped — `self_heal.arithmetic_errors`);
  rounding-tolerant, variable assignments ignored. A wrong step is a hard
  failure that names the exact error in the repair prompt.
- **Repairs**: the self-heal ladder already prefers the local engine; on the
  grading VM the +331-token repair cost seen in the mock A/B drops to ~0.

### 3. Cap and instruction tuning (est. −5–15% of remaining API tokens)

Completion tokens dominate API cost. Per category:

- Tighten `max_tokens` to just above the p95 observed completion length
  (benchmark report gives this per category). Self-heal's truncation retry
  makes tight caps safe — a rare truncation costs one retry, a loose cap
  costs every task.
- Shave instructions further (they're re-sent on every call): e.g.
  "State the sentiment label, then justify it in one sentence." →
  "Label + one-line reason." Every instruction word is paid N times.

### 4. Context shortening (small, apply with care)

Prompt text is billed as-is, but most of it is the task itself and cannot be
dropped without risking intent. Safe cuts only:

- Collapse runs of whitespace/blank lines in task prompts (shipped —
  `router_core.condense_prompt`, applied to every outbound prompt and repair;
  lossless: indentation inside code is preserved).
- Never echo the task back, no system message, no few-shot examples, single
  user message (already the design).
- Do **not** summarize long passages before sending — summarizing costs a
  call, and a lossy passage risks the gate. Long-passage tasks are exactly
  the ones to send to the local model instead.

### 5. Regex router hardening (zero tokens, protects the budget)

Misclassification is expensive in both directions: an easy task routed to the
code tier wastes premium tokens; a code task routed to the easy tier fails and
pays for a retry. The classifier is free, so invest here:

- Grow `router_core.classify()` patterns from benchmark misroutes.
- Ambiguity rule: when no pattern is confident, prefer the *cheapest* tier —
  self-heal escalates the failures, so the downside is bounded.

### 6. Model selection within ALLOWED_MODELS

- Launch day: resolve each tier to the cheapest model that passes the gate on
  a category sample (one benchmark run; submissions are rate-limited to
  10/hour, so tune via `bench/benchmark.py`, not by submitting).
- The disqualify-on-404 logic keeps a wrong guess from burning retries.

### 7. What we deliberately do NOT do

| Idea | Why not |
|---|---|
| Answer caching / memoization across runs | Banned by the rules; prompts are unseen variants |
| Semantic near-duplicate answer reuse | Same rule; judged as caching |
| Parallel sampling / best-of-n on Fireworks | Multiplies tokens for marginal accuracy |
| Few-shot examples | Prompt tokens × every task; terse instructions win |
| Reasoning models for logic/math | Thinking tokens are billed; a verified 31B non-reasoner is cheaper |
| Agentic multi-turn loops | Each turn re-pays the context |

Within-run exact-duplicate prompts may be answered once and copied (same
input, same run — not caching), but expect the harness to never send exact
duplicates; don't build around it.

## Runtime budget (the real constraint on local-first)

2 vCPU ⇒ roughly 8–12 tok/s local decode (the engine calibrates the real rate
at startup). 10-minute cap, ~60 s reserved for Fireworks fallbacks:

- ~540 s of local decode ≈ 4,500–6,500 local tokens ≈ 25–40 easy-category
  answers. `LocalEngine` already skips itself when the remaining budget is
  too small, so a large task file degrades gracefully to API calls instead
  of timing out (exit-code-0 gate is never at risk).
- Priority order when time is short: shortest-output categories first
  (sentiment > NER > factual > summary) — most tokens saved per local second.

## Measurement loop

1. `python bench/benchmark.py --self-heal` with the real key (per-category
   tokens, latency, judge match rate, `out/benchmark_report.json`).
2. Move categories local / tighten caps based on the report.
3. Re-run; submit only when the local A/B is stable (10 submissions/hour).
