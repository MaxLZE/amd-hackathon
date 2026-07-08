# Frugal Router — Track 1 General-Purpose AI Agent

Token-minimal agent for the Fireworks AI track. Strategy: classify each task
locally with regex (zero API tokens), optionally answer eligible tasks with a
bundled local model, then fall back to one Fireworks call with a terse
category-specific prompt and a tight `max_tokens` cap. Scoring passes an
accuracy gate, then ranks ascending by Fireworks tokens.

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
cp .env.example .env      # fill in your local Fireworks values
python3 -m pip install -r requirements.txt
python3 run_local.py      # runs tests/sample_tasks.json, validates schema, prints token totals
```

Useful harness flags:

```
python3 run_local.py --tasks tests/sample_tasks.json --out-dir out --json-report out/report.json
python3 run_local.py --config out/workbench_config.json --timeout 600
```

## Local workbench

The browser workbench is a prompt bar. Paste one prompt or a Track 1
`tasks.json` array; the backend estimates scale, runtime limits, concurrency,
token caps, and routing automatically. Secrets stay server-side.

```
python3 workbench_server.py
```

Open `http://127.0.0.1:8765`.

## Docker workbench

The workbench has its own image so the hackathon submission `Dockerfile` stays
minimal.

```
docker-compose up --build -d
```

Open `http://127.0.0.1:8765`. Runtime secrets are read from `.env`; they are not
copied into the image. Results and generated workbench config are written to
`out/`.

If Compose is unavailable, use plain Docker:

```
docker build -f Dockerfile.workbench -t frugal-router-workbench:local .
docker run --rm --name frugal-router-workbench --env-file .env -p 8765:8765 -v "$PWD/out:/app/out" frugal-router-workbench:local
```

On WSL, enable Docker Desktop integration for this distro or make
`/var/run/docker.sock` available before running these commands.

Benchmark checklist before submitting:
1. Token totals per category (agent logs them to stderr) — tune `max_tokens` caps.
2. M3 probe: point one math task at `minimax-m3` (temporarily edit `DEFAULT_TIER_PREFERENCES`)
   and compare `usage.completion_tokens` against `gemma-4-31b-it` to confirm the
   reasoning-token assumption before ever routing to it.
3. Zero warnings from the deterministic checks (code syntax, summary length, truncation).

## Local model strategy

Official clarification: local model answers count fully toward accuracy and use
zero Fireworks tokens. That is the best possible token score when accuracy holds.
The judging environment has 4 GB RAM and 2 vCPU. A 2B-3B 4-bit model is the
practical target; a 7B 4-bit model leaves little room for the agent. No Ollama
or model runtime is preinstalled, so model weights and runtime must be bundled
inside the Docker image while staying under the 10 GB compressed image limit.

This repo supports an optional local-first hook:

```
LOCAL_MODEL_COMMAND=python3 local_model_runner.py
LOCAL_MODEL_CATEGORIES=factual,sentiment,summary,ner
```

The command receives JSON on stdin with `task_id`, `prompt`, `category`, and
`instruction`, and should return either plain answer text or `{"answer": "..."}`.
If it fails or returns an empty answer, the agent falls back to Fireworks when
Fireworks env vars are configured.

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

- `DEFAULT_CATEGORIES` in [router_core.py](router_core.py): per-category instruction + `max_tokens`
- backend prediction scale in [workbench_server.py](workbench_server.py): local UI runtime, concurrency, and token cap estimates
- `out/workbench_config.json`: generated local routing/runtime config from the prompt workbench
- `FRUGAL_MODEL_EASY`, `FRUGAL_MODEL_REASON`, `FRUGAL_MODEL_CODE`: exact allowed-model overrides
- `MAX_RUNTIME_SECONDS` (default 510), `MAX_CONCURRENCY` (default 10)
