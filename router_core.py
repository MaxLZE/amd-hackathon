"""Shared routing, configuration, and classification helpers."""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CATEGORIES: dict[str, dict[str, Any]] = {
    "factual": {
        "instruction": "Answer concisely and accurately in 2-4 sentences.",
        "max_tokens": 220,
        "tier": "easy",
    },
    "math": {
        "instruction": "Solve step by step but keep each step brief. End with 'Answer: <result>'.",
        "max_tokens": 350,
        "tier": "reason",
    },
    "sentiment": {
        "instruction": "State the sentiment label, then justify it in one sentence.",
        "max_tokens": 80,
        "tier": "easy",
    },
    "summary": {
        "instruction": "Follow the requested length and format exactly. Output only the summary.",
        "max_tokens": 180,
        "tier": "easy",
    },
    "ner": {
        "instruction": "Extract the entities with their types. Output only the labelled list, nothing else.",
        "max_tokens": 200,
        "tier": "easy",
    },
    "code_debug": {
        "instruction": "If the code has a bug, state it in one or two sentences, then give the corrected code. No extra commentary.",
        "max_tokens": 600,
        "tier": "code",
    },
    "logic": {
        "instruction": "Reason through the constraints briefly, then clearly state the final answer.",
        "max_tokens": 450,
        "tier": "reason",
    },
    "codegen": {
        "instruction": "Output only the code with a minimal docstring. No explanation before or after.",
        "max_tokens": 600,
        "tier": "code",
    },
}

DEFAULT_TIER_PREFERENCES: dict[str, list[str]] = {
    "easy": ["glm-5p2", "glm-5p1", "kimi-k2p6", "gpt-oss-120b", "gemma-4-26b-a4b-it", "gemma-4-31b-it"],
    "reason": ["deepseek-v4-pro", "glm-5p2", "gpt-oss-120b", "gemma-4-31b-it", "gemma-4-31b-it-nvfp4"],
    "code": ["kimi-k2p6", "kimi-k2p5", "kimi-k2p7-code", "deepseek-v4-pro", "glm-5p2"],
}


@dataclass
class RuntimeConfig:
    max_runtime_seconds: float = 510
    max_concurrency: int = 10
    per_call_timeout: float = 30
    max_attempts: int = 3
    categories: dict[str, dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CATEGORIES))
    tier_preferences: dict[str, list[str]] = field(default_factory=lambda: copy.deepcopy(DEFAULT_TIER_PREFERENCES))
    model_overrides: dict[str, str] = field(default_factory=dict)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _as_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_dotenv(path: Path, env: dict[str, str] | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs without overwriting existing environment values."""

    target = env if env is not None else os.environ
    if not path.exists():
        return target
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip("'\"")
        target.setdefault(key.strip(), value)
    return target


def load_runtime_config(path: Path | None = None, env: dict[str, str] | None = None) -> RuntimeConfig:
    """Load workbench/agent configuration from JSON plus env overrides."""

    source_env = env if env is not None else os.environ
    default = RuntimeConfig().public_dict()
    config_path = path or (Path(source_env["FRUGAL_CONFIG_FILE"]) if source_env.get("FRUGAL_CONFIG_FILE") else None)
    if config_path and config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("config file must contain a JSON object")
        default = _deep_merge(default, loaded)

    config = RuntimeConfig(
        max_runtime_seconds=_as_float(source_env.get("MAX_RUNTIME_SECONDS"), float(default["max_runtime_seconds"])),
        max_concurrency=_as_int(source_env.get("MAX_CONCURRENCY"), int(default["max_concurrency"])),
        per_call_timeout=_as_float(source_env.get("PER_CALL_TIMEOUT"), float(default["per_call_timeout"])),
        max_attempts=_as_int(source_env.get("MAX_ATTEMPTS"), int(default["max_attempts"])),
        categories=default["categories"],
        tier_preferences=default["tier_preferences"],
        model_overrides=default.get("model_overrides", {}),
    )
    for tier in ("easy", "reason", "code"):
        env_key = f"FRUGAL_MODEL_{tier.upper()}"
        if source_env.get(env_key):
            config.model_overrides[tier] = source_env[env_key]
    return config


def save_runtime_config(config: dict[str, Any], path: Path) -> RuntimeConfig:
    merged = _deep_merge(RuntimeConfig().public_dict(), config)
    runtime = RuntimeConfig(
        max_runtime_seconds=float(merged["max_runtime_seconds"]),
        max_concurrency=int(merged["max_concurrency"]),
        per_call_timeout=float(merged["per_call_timeout"]),
        max_attempts=int(merged.get("max_attempts", 3)),
        categories=merged["categories"],
        tier_preferences=merged["tier_preferences"],
        model_overrides=merged.get("model_overrides", {}),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime.public_dict(), indent=2), encoding="utf-8")
    return runtime


CODE_SNIPPET_RE = re.compile(
    r"```|\bdef \w+\(|\bfunction\s+\w*\(|=>\s*{|\breturn\b.*;|#include\s*<|\bpublic\s+(static|class)\b"
)
CODEGEN_RE = re.compile(
    r"\b(write|implement|create|build)\b.{0,40}"
    r"\b(function|method|class|script|program|code|helper|utility|routine|algorithm|api|endpoint)\b"
)
MATH_RE = re.compile(
    r"\b(calculate|compute|how (much|many)|percent|percentage|total cost|average|"
    r"sum of|profit|interest|discount|per (hour|day|week|month|year)|projection)\b|%"
)
LOGIC_RE = re.compile(
    r"\b(puzzle|riddle|deduce|deduction|constraints?|clues?|exactly one|"
    r"who (is|has|owns|lives|sits)|seated|seating|arrange|logically|must be true|"
    r"each (has?|have|owns?)? ?a different|sits? (immediately )?(to the )?(left|right|next)|"
    r"either end|in a row)\b"
)
NER_RE = re.compile(
    r"\bentit(y|ies)\b|named entity|\bextract\b.{0,80}\b(person|people|organi[sz]ation|location|date)s?\b"
)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def classify(prompt: str) -> str:
    p = prompt.lower()
    if "sentiment" in p or re.search(r"\b(positive|negative|neutral)\b.{0,30}\bclassif", p):
        return "sentiment"
    if re.search(r"\bsummari[sz]e|\bsummary\b|\bcondense\b|tl;?dr", p):
        return "summary"
    if NER_RE.search(p):
        return "ner"
    if CODE_SNIPPET_RE.search(prompt):
        if CODEGEN_RE.search(p):
            return "codegen"
        return "code_debug"
    if CODEGEN_RE.search(p):
        return "codegen"
    if LOGIC_RE.search(p):
        return "logic"
    numbers = len(re.findall(r"\d[\d,.]*", prompt))
    if MATH_RE.search(p) or numbers >= 4:
        return "math"
    return "factual"


def resolve_models(allowed: list[str], config: RuntimeConfig | None = None) -> dict[str, str]:
    """Map each tier to a concrete allowed model ID."""

    runtime = config or RuntimeConfig()

    def find(fragment: str) -> str | None:
        for model in allowed:
            if model == fragment or model.endswith("/" + fragment):
                return model
        for model in allowed:
            if fragment in model:
                return model
        return None

    resolved = {}
    for tier, prefs in runtime.tier_preferences.items():
        override = runtime.model_overrides.get(tier)
        if override and override in allowed:
            resolved[tier] = override
            continue
        model = next((found for pref in prefs if (found := find(pref))), None)
        if model is None and allowed:
            print(
                f"warning: no tier preference matched ALLOWED_MODELS for tier '{tier}'; "
                f"falling back to {allowed[0]} — update DEFAULT_TIER_PREFERENCES with the launch-day model list",
                file=sys.stderr,
            )
        resolved[tier] = model or (allowed[0] if allowed else "")
    return resolved


def clean_answer(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text or "")
    # Salvage answers from reasoning output that was cut mid-stream: keep
    # whatever follows a closing tag, and never emit a bare "<think>" tag.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        head, _, tail = text.partition("<think>")
        text = head.strip() or tail
    return text.strip()


def condense_prompt(text: str) -> str:
    """Lossless whitespace trim: drop trailing spaces and collapse blank-line
    runs. Leading indentation is preserved — task prompts can contain code."""
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip()


def parse_allowed_models(value: str) -> list[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(model).strip() for model in parsed if str(model).strip()]
    return [model.strip().strip("'\"") for model in raw.split(",") if model.strip().strip("'\"")]
