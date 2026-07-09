"""Unit tests for the zero/low-token verification and repair helpers."""

from __future__ import annotations

import unittest

from router_core import condense_prompt
from self_heal import (
    HARD,
    SOFT,
    arithmetic_errors,
    exec_check,
    has_hard,
    local_fix,
    repair_prompt,
    trim_to_word_limit,
    verify,
)


class VerifyTests(unittest.TestCase):
    def test_clean_answer_passes(self) -> None:
        self.assertEqual(verify("What is 2+2?", "The answer is 4."), [])

    def test_empty_answer_is_hard(self) -> None:
        issues = verify("Anything", "   ")
        self.assertTrue(has_hard(issues))
        self.assertIn("empty", issues[0][1])

    def test_python_syntax_error_is_hard(self) -> None:
        answer = '```python\ndef evens(nums)\n    return [n for n in nums if n % 2 == 0]\n```'
        issues = verify("Write a Python function returning even numbers.", answer)
        self.assertTrue(has_hard(issues))
        self.assertTrue(any("SyntaxError" in msg for _, msg in issues))

    def test_valid_python_passes(self) -> None:
        answer = '```python\ndef evens(nums):\n    return [n for n in nums if n % 2 == 0]\n```'
        issues = verify("Write a Python function returning even numbers.", answer)
        self.assertFalse(has_hard(issues))

    def test_non_python_code_is_not_compiled(self) -> None:
        answer = "```\nfunction f(x) { return x; }\n```"
        issues = verify("Write a JavaScript function.", answer)
        self.assertFalse(has_hard(issues))

    def test_word_limit_violation_is_soft(self) -> None:
        answer = "one two three four five six seven eight nine ten."
        issues = verify("Condense this to at most 5 words: ...", answer)
        self.assertEqual([severity for severity, _ in issues], [SOFT])

    def test_mid_sentence_ending_is_soft(self) -> None:
        issues = verify("Explain inflation.", "Inflation is a sustained increase in")
        self.assertEqual([severity for severity, _ in issues], [SOFT])


class RepairTests(unittest.TestCase):
    def test_trim_prefers_sentence_boundary(self) -> None:
        self.assertEqual(trim_to_word_limit("One two three four. Five six seven eight nine ten", 6), "One two three four.")

    def test_trim_hard_cuts_without_boundary(self) -> None:
        self.assertEqual(trim_to_word_limit("a b c d e f g h", 3), "a b c")

    def test_local_fix_trims_word_limit(self) -> None:
        prompt = "Summarise in at most 4 words: ..."
        answer = "Alpha beta gamma delta epsilon zeta."
        fixed = local_fix(prompt, answer, verify(prompt, answer))
        self.assertIsNotNone(fixed)
        self.assertLessEqual(len(fixed.split()), 4)

    def test_local_fix_none_when_nothing_applies(self) -> None:
        prompt = "Write a Python function."
        answer = "def f(:\n    pass"
        self.assertIsNone(local_fix(prompt, answer, verify(prompt, answer)))

    def test_repair_prompt_names_the_failures(self) -> None:
        prompt = "Write a Python function returning even numbers."
        answer = "def evens(nums)\n    return []"
        issues = verify(prompt, answer)
        text = repair_prompt(prompt, answer, issues)
        self.assertIn("failed checks", text)
        self.assertIn("SyntaxError", text)
        self.assertIn(answer, text)

    def test_hard_constant_used_for_empty(self) -> None:
        self.assertEqual(verify("q", "")[0][0], HARD)


class ArithmeticTests(unittest.TestCase):
    def test_wrong_step_is_flagged(self) -> None:
        errors = arithmetic_errors("80 * 0.75 = 65; then 65 * 1.10 = 71.50. Answer: $71.50")
        self.assertTrue(any("80 * 0.75 = 65" in e for e in errors))

    def test_correct_steps_pass(self) -> None:
        self.assertEqual(arithmetic_errors("80 * 0.75 = 60; 60 * 1.10 = 66. Answer: $66.00"), [])

    def test_rounding_is_tolerated(self) -> None:
        self.assertEqual(arithmetic_errors("2 / 3 = 0.67 and 2 * 1.15**3 = 3.04"), [])

    def test_variable_assignments_ignored(self) -> None:
        self.assertEqual(arithmetic_errors("Let x = 5 and rate = 0.15"), [])

    def test_restatements_ignored(self) -> None:
        self.assertEqual(arithmetic_errors("The total = 140 pages"), [])

    def test_verify_flags_math_only_for_math_category(self) -> None:
        answer = "12 * 5 = 61. Answer: 61."
        self.assertTrue(has_hard(verify("How many pages?", answer, category="math")))
        self.assertFalse(has_hard(verify("How many pages?", answer)))


class ExecCheckTests(unittest.TestCase):
    def test_crashing_code_is_caught(self) -> None:
        answer = "```python\nimport math\nvalue = 1 / 0\n```"
        failure = exec_check("Write a Python snippet.", answer)
        self.assertIsNotNone(failure)
        self.assertIn("ZeroDivisionError", failure)

    def test_valid_function_passes(self) -> None:
        answer = "```python\ndef double(x):\n    return x * 2\n```"
        self.assertIsNone(exec_check("Write a Python function double(x).", answer))

    def test_literal_example_from_prompt_is_asserted(self) -> None:
        prompt = "Write a Python function double(x). double(2) should return 4."
        wrong = "```python\ndef double(x):\n    return x * 3\n```"
        right = "```python\ndef double(x):\n    return x * 2\n```"
        self.assertIsNotNone(exec_check(prompt, wrong))
        self.assertIsNone(exec_check(prompt, right))

    def test_symbolic_example_is_not_asserted(self) -> None:
        prompt = "Implement is_palindrome(s) in Python; is_palindrome(s) returns True if s is a palindrome."
        answer = "```python\ndef is_palindrome(s):\n    t = ''.join(c.lower() for c in s if c.isalnum())\n    return t == t[::-1]\n```"
        self.assertIsNone(exec_check(prompt, answer))

    def test_verify_reports_exec_failure_as_hard(self) -> None:
        answer = "```python\nraise RuntimeError('boom')\n```"
        self.assertTrue(has_hard(verify("Write a Python snippet.", answer)))


class CondenseTests(unittest.TestCase):
    def test_blank_runs_collapse_and_trailing_spaces_drop(self) -> None:
        self.assertEqual(condense_prompt("a  \n\n\n\nb\n"), "a\n\nb")

    def test_indentation_preserved(self) -> None:
        code = "Fix this:\n\ndef f():\n    return 1\n"
        self.assertIn("    return 1", condense_prompt(code))


if __name__ == "__main__":
    unittest.main()
