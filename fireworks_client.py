"""Small OpenAI-compatible chat client for Fireworks-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.error
import urllib.request
from types import SimpleNamespace
from urllib.parse import urljoin

# Strongest reasoning suppression first. The allowed models bill hidden
# reasoning tokens on every call (60-100 tokens for a one-sentence answer)
# and, worse, tight max_tokens caps truncate mid-reasoning so the visible
# content is chain-of-thought garbage. "none" eliminates it entirely on the
# GLM/Kimi/DeepSeek pool; gpt-oss rejects "none" but accepts "low"; the final
# None entry drops the parameter for models that reject it outright.
REASONING_EFFORT_LADDER: tuple[str | None, ...] = ("none", "low", None)


class ReasoningEffort:
    """Per-model position on REASONING_EFFORT_LADDER, downgraded on 400s."""

    def __init__(self, start: str | None = "none") -> None:
        self._start = REASONING_EFFORT_LADDER.index(start) if start in REASONING_EFFORT_LADDER else len(REASONING_EFFORT_LADDER) - 1
        self._idx: dict[str, int] = {}

    def current(self, model: str) -> str | None:
        return REASONING_EFFORT_LADDER[self._idx.get(model, self._start)]

    def downgrade(self, model: str) -> bool:
        idx = self._idx.get(model, self._start)
        if idx + 1 >= len(REASONING_EFFORT_LADDER):
            return False
        self._idx[model] = idx + 1
        print(
            f"reasoning_effort downgraded to {REASONING_EFFORT_LADDER[idx + 1]!r} for {model}",
            file=sys.stderr,
        )
        return True


def _is_bad_request(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        # 404 model_not_found bodies also carry "invalid_request_error", so
        # the status code is the only trustworthy signal when present.
        return status == 400
    text = str(exc).lower()
    return "invalid_request_error" in text and "model_not_found" not in text and "not found" not in text


async def create_chat(client, effort: ReasoningEffort | None, **payload):
    """chat.completions.create with reasoning suppression and 400 downgrade.

    Works with both FireworksClient and AsyncOpenAI: the effort level rides
    in extra_body, which AsyncOpenAI merges natively and FireworksClient
    merges in _create_sync.
    """
    while True:
        level = effort.current(payload["model"]) if effort else None
        kwargs = dict(payload)
        if level:
            kwargs["extra_body"] = {"reasoning_effort": level}
        try:
            return await client.chat.completions.create(**kwargs)
        except Exception as exc:
            # A 400 with the effort param set is almost certainly the model
            # rejecting that effort level; downgrading is capped at two steps
            # per model, after which the original error propagates.
            if level and effort and _is_bad_request(exc) and effort.downgrade(payload["model"]):
                continue
            raise


class FireworksHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


class FireworksClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self.chat = SimpleNamespace(completions=_Completions(self))

    async def close(self) -> None:
        return None


class _Completions:
    def __init__(self, client: FireworksClient) -> None:
        self._client = client

    async def create(self, **payload):
        return await asyncio.to_thread(self._create_sync, payload)

    def _create_sync(self, payload: dict):
        url = urljoin(self._client.base_url, "chat/completions")
        extra = payload.pop("extra_body", None) or {}
        payload = {**payload, **extra}
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._client.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request) as response:  # noqa: S310 - user-provided Fireworks base URL is required
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FireworksHTTPError(exc.code, detail) from exc

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=(choice.get("message") or {}).get("content", "")),
                    finish_reason=choice.get("finish_reason"),
                )
                for choice in data.get("choices", [])
            ],
            usage=SimpleNamespace(**(data.get("usage") or {})),
        )
