# Frugal Router — Track 1 General-Purpose AI Agent

Token-minimal agent for the Fireworks AI track. Strategy: classify each task
locally with regex (zero API tokens), then make exactly one call per task with
a terse category-specific prompt, a tight `max_tokens` cap, and the cheapest
suitable allowed model. Scoring passes an accuracy gate, then ranks ascending
by total tokens — so no agentic loops, no few-shot examples, no LLM classification.

## Routing

| Categories | Model | Why |
|---|---|---|
| sentiment, NER, summary, factual | `gemma-4-26b-a4b-it` | cheap MoE, follows brevity instructions |
| math, logic | `gemma-4-31b-it` | strongest non-reasoning option |
| code debug, codegen | `kimi-k2p7-code` | code specialist |
| (escalation only) | `minimax-m3` | assumed reasoning model; thinking tokens likely count toward the score — verify via benchmark before using |

Models are resolved at runtime against `ALLOWED_MODELS` by suffix/substring
match; falls back to the first allowed model.

## Local testing

```
python -m pip install -r requirements.txt
python run_local.py       # runs tests/sample_tasks.json, validates schema, prints token totals
```

## Vercel workbench

The Vercel entry point is [api/index.py](api/index.py), with routing in
[vercel.json](vercel.json). Static UI files are served from `web/`, and
`/api/run` executes the prompt synchronously inside the serverless function
using `/tmp` files.

Configure these Vercel environment variables:

```
FIREWORKS_API_KEY=...
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
ALLOWED_MODELS=accounts/fireworks/models/glm-5p2,accounts/fireworks/models/kimi-k2p6
```

Benchmark checklist before submitting:
1. Token totals per category (agent logs them to stderr) — tune `max_tokens` caps.
2. M3 probe: point one math task at `minimax-m3` (temporarily edit `TIER_PREFERENCES`)
   and compare `usage.completion_tokens` against `gemma-4-31b-it` to confirm the
   reasoning-token assumption before ever routing to it.
3. Zero warnings from the deterministic checks (code syntax, summary length, truncation).

## Build & push (judging VM is linux/amd64)

Preferred: push the repo to GitHub — [.github/workflows/docker.yml](.github/workflows/docker.yml)
builds linux/amd64 and pushes to `ghcr.io/<user>/frugal-router:latest` automatically.
**After the first push, set the GHCR package visibility to Public** (packages are
private by default and the harness cannot pull private images).

Or locally with Docker installed:

```
docker buildx build --platform linux/amd64 -t ghcr.io/<user>/frugal-router:latest --push .
```

## Container smoke test

```
docker run --rm -v "%cd%/tests:/input" -v "%cd%/out:/output" ^
  -e FIREWORKS_API_KEY=... -e FIREWORKS_BASE_URL=... -e ALLOWED_MODELS=... ^
  --entrypoint python ghcr.io/<user>/frugal-router:latest agent.py
```

(mount `tests/sample_tasks.json` as `/input/tasks.json` or set `TASKS_FILE`)

## Tuning knobs

- `CATEGORIES` dict in [agent.py](agent.py): per-category instruction + `max_tokens`
- `TIER_PREFERENCES`: which model each category tier prefers
- `MAX_RUNTIME_SECONDS` (default 510), `MAX_CONCURRENCY` (default 10)
