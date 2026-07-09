"""In-process llama.cpp engine for zero-token local answers.

Loads the GGUF model once at startup and serializes generations behind an
asyncio lock: the grading VM has 2 vCPU, so concurrent generations would
only thrash the cache. Every answer produced here costs zero Fireworks
tokens.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time

# Conservative 2-vCPU estimates for a ~3B Q4 model; decode speed is refined
# from observed throughput as generations complete.
PREFILL_TOKENS_PER_SECOND = 80.0
DECODE_TOKENS_PER_SECOND = 4.0


class LocalEngine:
    def __init__(self, llm) -> None:
        self._llm = llm
        self._lock = asyncio.Lock()
        self._decode_tps = float(os.environ.get("LOCAL_DECODE_TPS", DECODE_TOKENS_PER_SECOND))

    @classmethod
    def create(cls, model_path: str) -> "LocalEngine | None":
        """Load the model, or return None (with a stderr note) if unavailable."""
        if not model_path:
            return None
        if not os.path.exists(model_path):
            print(f"local model not found at {model_path}", file=sys.stderr)
            return None
        try:
            from llama_cpp import Llama
        except Exception as exc:  # noqa: BLE001 - local model is optional
            print(f"llama_cpp unavailable: {exc!r}", file=sys.stderr)
            return None
        try:
            started = time.monotonic()
            llm = Llama(
                model_path=model_path,
                n_ctx=int(os.environ.get("LOCAL_MODEL_CTX", "4096")),
                n_threads=int(os.environ.get("LOCAL_MODEL_THREADS", "2")),
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"failed to load local model: {exc!r}", file=sys.stderr)
            return None
        print(
            f"local model loaded in {time.monotonic() - started:.1f}s: {model_path}",
            file=sys.stderr,
        )
        engine = cls(llm)
        engine._calibrate()
        return engine

    def _calibrate(self) -> None:
        """Measure real decode speed with a tiny generation so the first
        budget decision is honest — the default estimate assumes a fast CPU
        and a throttled/slow host could otherwise start a generation that
        blows the runtime budget before the estimate self-corrects."""
        if os.environ.get("LOCAL_DECODE_TPS"):
            return  # explicit override wins
        try:
            # The model is mmap'd: the very first inference touches every
            # weight page for the first time, so timing it measures cold
            # disk I/O (seconds) rather than decode speed. Warm the page
            # cache with an untimed call before the timed ones.
            self._generate_sync("Hi", 4)
            samples = []
            for _ in range(3):
                started = time.monotonic()
                resp = self._generate_sync("Say OK.", 8)
                elapsed = max(time.monotonic() - started, 0.01)
                tokens = (resp.get("usage") or {}).get("completion_tokens", 0)
                if tokens:
                    samples.append(tokens / elapsed)
            if samples:
                # Median of 3 discards a one-off disk/scheduler hiccup that a
                # single sample can't distinguish from genuinely slow hardware;
                # 20% safety margin on top of that.
                self._decode_tps = max(0.1, statistics.median(samples) * 0.8)
                print(f"local decode calibrated: {statistics.median(samples):.2f} tok/s (samples={[round(s, 2) for s in samples]})", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - keep the default estimate
            print(f"local calibration failed: {exc!r}", file=sys.stderr)

    def estimate_seconds(self, content: str, max_tokens: int) -> float:
        prompt_tokens = max(1, len(content) // 3)
        return prompt_tokens / PREFILL_TOKENS_PER_SECOND + max_tokens / self._decode_tps

    async def generate(self, content: str, max_tokens: int, time_limit: float) -> tuple[str, str] | None:
        """Return (text, finish_reason), or None when skipped for lack of time.

        `time_limit` is an absolute time.monotonic() value the generation must
        finish before (deadline minus the Fireworks reserve).
        """
        async with self._lock:
            # Re-check after waiting for the lock: generations queued ahead of
            # us may have eaten the remaining budget.
            if time.monotonic() + self.estimate_seconds(content, max_tokens) > time_limit:
                return None
            started = time.monotonic()
            try:
                resp = await asyncio.to_thread(self._generate_sync, content, max_tokens)
            except Exception as exc:  # noqa: BLE001 - fall back to Fireworks
                print(f"local generation failed: {exc!r}", file=sys.stderr)
                return ("", "error")
            elapsed = max(time.monotonic() - started, 0.1)
            completion = (resp.get("usage") or {}).get("completion_tokens", 0)
            if completion:
                observed = completion / elapsed
                self._decode_tps = max(0.1, 0.7 * self._decode_tps + 0.3 * observed)
            choice = resp["choices"][0]
            text = (choice.get("message") or {}).get("content") or ""
            return (text, choice.get("finish_reason") or "stop")

    def _generate_sync(self, content: str, max_tokens: int) -> dict:
        return self._llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0,
        )
