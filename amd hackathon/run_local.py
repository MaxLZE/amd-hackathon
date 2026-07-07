"""Local test harness for the Frugal Router agent.

Loads .env, runs agent.py against tests/sample_tasks.json, validates the
output schema, and prints each answer plus the token summary the agent
logs to stderr. Usage:  python run_local.py [tasks_file]
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TASKS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "sample_tasks.json"
OUT_DIR = ROOT / "out"
RESULTS = OUT_DIR / "results.json"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    load_dotenv(ROOT / ".env")
    for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
        if var not in os.environ:
            print(f"missing {var} — copy .env.example to .env and fill it in")
            return 1

    OUT_DIR.mkdir(exist_ok=True)
    env = os.environ | {"TASKS_FILE": str(TASKS), "RESULTS_FILE": str(RESULTS)}

    proc = subprocess.run([sys.executable, str(ROOT / "agent.py")], env=env)
    print(f"\nagent exit code: {proc.returncode}")
    if proc.returncode != 0:
        return proc.returncode

    # --- schema validation -------------------------------------------------
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert isinstance(results, list), "results.json must be a JSON array"
    by_id = {}
    for r in results:
        assert set(r) == {"task_id", "answer"}, f"bad keys: {set(r)}"
        assert isinstance(r["answer"], str), f"answer must be a string: {r['task_id']}"
        by_id[r["task_id"]] = r["answer"]
    missing = [t["task_id"] for t in tasks if t["task_id"] not in by_id]
    assert not missing, f"missing task_ids: {missing}"
    empty = [tid for tid, a in by_id.items() if not a.strip()]

    print(f"schema OK: {len(results)} results, {len(empty)} empty answers {empty or ''}")

    # --- zero-token deterministic checks (warnings only) --------------------
    warnings = []
    for t in tasks:
        tid, prompt, answer = t["task_id"], t["prompt"], by_id[t["task_id"]]
        if not answer.strip():
            continue  # already reported as empty
        if "python" in prompt.lower() and re.search(r"\bdef |```", answer):
            code = "\n".join(re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)) or answer
            try:
                compile(code, tid, "exec")
            except SyntaxError as exc:
                warnings.append(f"{tid}: python answer has a SyntaxError ({exc.msg})")
        limit = re.search(r"(?:at most|maximum of|no more than)\s+(\d+)\s+words", prompt, re.I)
        if limit and len(answer.split()) > int(limit.group(1)):
            warnings.append(f"{tid}: answer is {len(answer.split())} words, limit {limit.group(1)}")
        if re.search(r"[a-zA-Z,]$", answer.rstrip()) and not answer.rstrip().endswith("```"):
            warnings.append(f"{tid}: answer may be truncated (ends mid-sentence)")
    for w in warnings:
        print(f"WARN {w}")
    print(f"deterministic checks: {len(warnings)} warning(s)")

    print("\n--- answers ---")
    for t in tasks:
        answer = by_id[t["task_id"]].replace("\n", " ")
        print(f"\n[{t['task_id']}] {answer[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
