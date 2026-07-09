"""Zero/low-token answer verification and repair (self-healing).

Scoring is accuracy-gate first, then ascending tokens: a malformed answer can
cost a whole task at the gate, while a targeted repair costs a few hundred
tokens at most — but blind re-asking erodes the token budget. The ladder here
is strictly cheapest-first:

  1. verify()      deterministic checks, zero tokens
  2. local_fix()   mechanical fixes (word-limit trims), zero tokens
  3. repair        one terse model call, only for HARD issues; callers should
                   prefer the bundled local model (zero Fireworks tokens)
                   before spending API tokens

Soft issues (heuristic truncation, style) are never worth a repair call.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

from router_core import condense_prompt

HARD = "hard"
SOFT = "soft"

WORD_LIMIT_RE = re.compile(r"(?:at most|maximum of|no more than)\s+(\d+)\s+words", re.I)
CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")

# "f(2) should return 4" / "f(2) returns 4" / ">>> f(2)\n4" — only literal
# arguments and literal expected values become executable assertions.
SHOULD_RETURN_RE = re.compile(
    r"\b(\w+\([^\n)]*\))\s+(?:should\s+return|must\s+return|returns|==)\s+"
    r"(True|False|None|-?\d[\d,.]*|\"[^\"]*\"|'[^']*'|\[[^\]\n]*\]|\([^)\n]*\)|\{[^}\n]*\})"
)
DOCTEST_RE = re.compile(r">>>\s*([^\n]+)\n\s*([^\s>][^\n]*)")

# "80 * 0.75 = 60" style worked steps; letters never match, so variable
# assignments and code are excluded by construction.
ARITH_CANDIDATE_RE = re.compile(r"([0-9][0-9,.\s()+*/×÷^-]*?)\s*=\s*\$?(-?[\d,]+(?:\.\d+)?)")

EXEC_CHECK_ENABLED = os.environ.get("SELF_HEAL_EXEC", "1").strip().lower() not in {"0", "false", "off"}
EXEC_TIMEOUT_SECONDS = float(os.environ.get("SELF_HEAL_EXEC_TIMEOUT", "4"))
_EXEC_CACHE: dict[tuple[int, int], str | None] = {}


def _python_code(prompt: str, answer: str) -> str | None:
    """Extract python code from an answer when the task asked for python."""
    if "python" not in prompt.lower() or not re.search(r"\bdef |```", answer):
        return None
    blocks = CODE_BLOCK_RE.findall(answer)
    return "\n".join(blocks) if blocks else answer


def _safe_arith(expr: str) -> float | None:
    """Evaluate a pure-arithmetic expression; None if anything else appears."""
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed):
            return None
        if isinstance(child, ast.Constant) and not isinstance(child.value, (int, float)):
            return None
    try:
        return float(eval(compile(node, "<arith>", "eval"), {"__builtins__": {}}))  # noqa: S307 - AST-whitelisted
    except Exception:  # noqa: BLE001 - overflow/zero-division in model text
        return None


def arithmetic_errors(answer: str) -> list[str]:
    """Recompute worked steps like '80 * 0.75 = 60'. Free accuracy check for math.

    Forgiving on rounding: only flags when the stated value is wrong even at
    its own precision AND off by more than 0.5%.
    """
    errors: list[str] = []
    for match in ARITH_CANDIDATE_RE.finditer(answer):
        raw_expr, raw_value = match.group(1), match.group(2)
        expr = (
            raw_expr.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "").strip()
        )
        if not re.search(r"[+*/-]", expr.strip("() .")):
            continue  # "80 = 80" is a restatement, not a computation
        computed = _safe_arith(expr)
        if computed is None:
            continue
        stated_text = raw_value.replace(",", "")
        stated = float(stated_text)
        decimals = len(stated_text.split(".")[1]) if "." in stated_text else 0
        rounded_ok = round(computed, decimals) == stated
        relative_off = abs(computed - stated) > 0.005 * max(1.0, abs(computed))
        if not rounded_ok and relative_off:
            errors.append(f"arithmetic error: {raw_expr.strip()} = {raw_value} (computed {computed:g})")
        if len(errors) >= 3:
            break
    return errors


def _literal_call_asserts(prompt: str, code: str) -> list[str]:
    """Turn examples stated in the prompt into assertions — only when the call
    uses literal arguments and the function is actually defined in the code."""
    asserts: list[str] = []
    pairs = list(SHOULD_RETURN_RE.findall(prompt))
    for expr, expected in DOCTEST_RE.findall(prompt):
        pairs.append((expr.strip(), expected.strip()))
    for expr, expected in pairs:
        try:
            call = ast.parse(expr.strip(), mode="eval").body
            ast.literal_eval(expected)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        try:
            for arg in call.args:
                ast.literal_eval(arg)
        except (ValueError, SyntaxError):
            continue  # symbolic args like f(s) — nothing to execute
        if call.keywords:
            continue
        if re.search(rf"\bdef\s+{re.escape(call.func.id)}\s*\(", code):
            asserts.append(f"assert ({expr.strip()}) == ({expected.strip()})")
    return asserts


def exec_check(prompt: str, answer: str) -> str | None:
    """Run python answers in an isolated subprocess; zero tokens.

    Catches code that compiles but crashes, and answers that violate literal
    examples stated in the prompt. Returns a description or None when clean.
    """
    if not EXEC_CHECK_ENABLED:
        return None
    code = _python_code(prompt, answer)
    if code is None:
        return None
    key = (hash(prompt), hash(answer))
    if key in _EXEC_CACHE:
        return _EXEC_CACHE[key]
    script = code + "\n" + "\n".join(_literal_call_asserts(prompt, code)) + "\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        result = "python code did not finish executing (possible infinite loop)"
    except Exception:  # noqa: BLE001 - no subprocess available: skip, never block
        result = None
    else:
        if proc.returncode == 0:
            result = None
        else:
            lines = (proc.stderr or "").strip().splitlines()
            detail = lines[-1] if lines else f"exit code {proc.returncode}"
            result = f"python code fails when executed: {detail}"
    if len(_EXEC_CACHE) > 256:
        _EXEC_CACHE.clear()
    _EXEC_CACHE[key] = result
    return result


def verify(prompt: str, answer: str, category: str | None = None) -> list[tuple[str, str]]:
    """Deterministic, zero-token checks. Returns (severity, description) pairs."""
    if not answer.strip():
        return [(HARD, "answer is empty")]
    issues: list[tuple[str, str]] = []
    code = _python_code(prompt, answer)
    if code is not None:
        try:
            compile(code, "<answer>", "exec")
        except SyntaxError as exc:
            issues.append((HARD, f"python code has a SyntaxError: {exc.msg} (line {exc.lineno})"))
        else:
            failure = exec_check(prompt, answer)
            if failure:
                issues.append((HARD, failure))
    if category == "math":
        issues.extend((HARD, error) for error in arithmetic_errors(answer))
    limit = WORD_LIMIT_RE.search(prompt)
    words = len(answer.split())
    if limit and words > int(limit.group(1)):
        issues.append((SOFT, f"answer is {words} words, over the {limit.group(1)}-word limit"))
    if re.search(r"[a-zA-Z,]$", answer.rstrip()) and not answer.rstrip().endswith("```"):
        issues.append((SOFT, "answer may be truncated (ends mid-sentence)"))
    return issues


def has_hard(issues: list[tuple[str, str]]) -> bool:
    return any(severity == HARD for severity, _ in issues)


def trim_to_word_limit(answer: str, limit: int) -> str:
    """Cut to the word limit, preferring a sentence boundary past half the limit."""
    words = answer.split()
    if len(words) <= limit:
        return answer
    clipped = " ".join(words[:limit])
    ends = [match.end() for match in SENTENCE_END_RE.finditer(clipped)]
    if ends and len(clipped[: ends[-1]].split()) >= limit // 2:
        return clipped[: ends[-1]].strip()
    return clipped.rstrip(",;: ")


def local_fix(prompt: str, answer: str, issues: list[tuple[str, str]]) -> str | None:
    """Mechanical zero-token repair; None when nothing applies."""
    limit = WORD_LIMIT_RE.search(prompt)
    if limit and any("word limit" in description for _, description in issues):
        fixed = trim_to_word_limit(answer, int(limit.group(1)))
        if fixed != answer:
            return fixed
    return None


def repair_prompt(prompt: str, answer: str, issues: list[tuple[str, str]]) -> str:
    """Terse one-shot repair request; includes only the failed checks."""
    problems = "; ".join(description for _, description in issues)
    return (
        f"{condense_prompt(prompt)}\n\n"
        f"Your previous answer failed checks: {problems}.\n"
        f"Previous answer:\n{answer}\n\n"
        f"Output only the corrected answer, nothing else."
    )
