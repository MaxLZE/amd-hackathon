"""Throughput benchmark of candidate local GGUF models under 2-core pinning.

Simulates the 2-vCPU grading VM: each model is measured in a subprocess
pinned with `taskset -c 0,1` and n_threads=2 (LocalEngine, the production
inference path). Reports load time, prefill/decode tok/s, peak RSS, and a
projected full-run wall clock so model size can be chosen by time budget.

Host cores are faster than grading vCPUs — treat numbers as *relative*,
anchored to the Qwen2.5-3B baseline.

Usage:  python3 bench/local_bench.py [--models qwen25-3b,qwen25-1.5b] [--out out/local_bench_report.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_DIR = Path(os.path.expanduser("~/models"))
# Grading-run projection assumptions: ~100 tasks, terse ~60-token answers,
# ~200 prompt tokens each, 450s usable budget, 1.5x safety margin.
PROJECT_TASKS = 100
PROJECT_OUT_TOKENS = 60
PROJECT_PROMPT_TOKENS = 200
BUDGET_SECONDS = 450
SAFETY = 1.5

# First URL that downloads wins; later entries are fallbacks (quant repos
# come and go). Missing everywhere -> candidate is skipped with a note.
CANDIDATES: dict[str, list[str]] = {
    "qwen25-3b": [],  # expected to already exist as ~/models/local.gguf
    "qwen25-1.5b": [
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    ],
    "gemma4-4b": [
        "https://huggingface.co/ggml-org/gemma-4-4b-it-GGUF/resolve/main/gemma-4-4b-it-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/google_gemma-4-4b-it-GGUF/resolve/main/google_gemma-4-4b-it-Q4_K_M.gguf",
    ],
    "gemma4-2b": [
        "https://huggingface.co/ggml-org/gemma-4-2b-it-GGUF/resolve/main/gemma-4-2b-it-Q4_K_M.gguf",
        "https://huggingface.co/bartowski/google_gemma-4-2b-it-GGUF/resolve/main/google_gemma-4-2b-it-Q4_K_M.gguf",
    ],
    "llama32-3b": [
        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ],
}

PREFILL_TEXT = "Summarize the following paragraph in one sentence. " + ("The quick brown fox jumps over the lazy dog. " * 40)


def model_path(name: str) -> Path:
    if name == "qwen25-3b":
        return MODELS_DIR / "local.gguf"
    return MODELS_DIR / f"{name}.q4_k_m.gguf"


def ensure_model(name: str) -> Path | None:
    path = model_path(name)
    if path.exists():
        return path
    for url in CANDIDATES.get(name, []):
        part = path.with_suffix(".part")
        print(f"[{name}] downloading {url}", file=sys.stderr)
        proc = subprocess.run(["curl", "-sL", "--fail", "--retry", "3", "-o", str(part), url])
        if proc.returncode == 0 and part.exists() and part.stat().st_size > 100 * 1024 * 1024:
            part.rename(path)
            return path
        part.unlink(missing_ok=True)
        print(f"[{name}] not available at that URL", file=sys.stderr)
    return None


async def measure(path: str) -> dict:
    """Child mode: load the model with 2 threads and measure. Prints JSON."""
    from local_engine import LocalEngine

    os.environ["LOCAL_MODEL_THREADS"] = "2"
    os.environ.pop("LOCAL_DECODE_TPS", None)
    started = time.monotonic()
    engine = LocalEngine.create(path)
    if engine is None:
        return {"ok": False, "error": "load failed"}
    load_s = time.monotonic() - started
    deadline = time.monotonic() + 10**6

    # Prefill: long prompt, tiny completion; decode: short prompt, long completion.
    prefill_tps, decode_tps = [], []
    for _ in range(3):
        t0 = time.monotonic()
        await engine.generate(PREFILL_TEXT, 4, deadline)
        prefill_tps.append(len(PREFILL_TEXT) / 3 / max(time.monotonic() - t0, 0.01))
    for _ in range(3):
        t0 = time.monotonic()
        result = await engine.generate("Count from 1 to 60 as words, comma separated.", 128, deadline)
        elapsed = max(time.monotonic() - t0, 0.01)
        text = result[0] if result else ""
        decode_tps.append(max(len(text) // 3, 1) / elapsed)

    hwm_gb = 0.0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM"):
            hwm_gb = round(int(line.split()[1]) / 1024 / 1024, 2)
    return {
        "ok": True,
        "load_s": round(load_s, 1),
        "prefill_tps": round(statistics.median(prefill_tps), 1),
        "decode_tps": round(statistics.median(decode_tps), 1),
        "vm_hwm_gb": hwm_gb,
    }


def project(decode_tps: float, prefill_tps: float) -> float:
    per_task = PROJECT_OUT_TOKENS / max(decode_tps, 0.1) + PROJECT_PROMPT_TOKENS / max(prefill_tps, 1)
    return round(PROJECT_TASKS * per_task * SAFETY, 0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", default=",".join(CANDIDATES))
    p.add_argument("--out", default=str(ROOT / "out" / "local_bench_report.json"))
    p.add_argument("--measure", help="internal: child mode, measure one gguf path")
    args = p.parse_args()

    if args.measure:
        print(json.dumps(asyncio.run(measure(args.measure))))
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        path = ensure_model(name)
        if path is None:
            results.append({"name": name, "ok": False, "error": "unavailable"})
            continue
        proc = subprocess.run(
            ["taskset", "-c", "0,1", sys.executable, __file__, "--measure", str(path)],
            capture_output=True, text=True,
        )
        line = (proc.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = {"ok": False, "error": proc.stderr[-300:]}
        row.update({"name": name, "path": str(path), "size_gb": round(path.stat().st_size / 1e9, 2)})
        if row.get("ok"):
            row["projected_run_s"] = project(row["decode_tps"], row["prefill_tps"])
            row["fits_budget"] = row["projected_run_s"] <= BUDGET_SECONDS
        results.append(row)
        print(f"[{name}] {row}", file=sys.stderr)

    report = {
        "assumptions": {"tasks": PROJECT_TASKS, "out_tokens": PROJECT_OUT_TOKENS,
                        "prompt_tokens": PROJECT_PROMPT_TOKENS, "budget_s": BUDGET_SECONDS, "safety": SAFETY,
                        "note": "host cores faster than grading vCPUs; compare relatively, anchor on qwen25-3b"},
        "models": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'model':14} {'size':>6} {'load s':>7} {'prefill':>8} {'decode':>7} {'RSS GB':>7} {'proj s':>7} fits")
    for r in results:
        if not r.get("ok"):
            print(f"{r['name']:14} unavailable ({r.get('error', '')[:60]})")
            continue
        print(f"{r['name']:14} {r['size_gb']:>6} {r['load_s']:>7} {r['prefill_tps']:>8} "
              f"{r['decode_tps']:>7} {r['vm_hwm_gb']:>7} {r['projected_run_s']:>7} {r['fits_budget']}")
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
