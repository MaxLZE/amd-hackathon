"""Small OpenAI-compatible chat client for Fireworks-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from types import SimpleNamespace
from urllib.parse import urljoin


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
