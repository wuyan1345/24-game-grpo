from __future__ import annotations

import ast
import math
import re
from collections import Counter
from dataclasses import dataclass


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
ALLOWED_CHARS_PATTERN = re.compile(r"^[\d\s\+\-\*/\(\)\.]+$")


@dataclass
class VerificationResult:
    has_answer_tag: bool
    answer_text: str
    valid_chars: bool
    valid_syntax: bool
    used_numbers_match: bool
    evaluates_to_24: bool
    value: float | None
    error: str | None = None

    @property
    def is_valid_expression(self) -> bool:
        return self.valid_chars and self.valid_syntax and self.used_numbers_match

    @property
    def is_correct(self) -> bool:
        return self.is_valid_expression and self.evaluates_to_24


class SafeEvalVisitor(ast.NodeVisitor):
    allowed_binary_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def visit(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self.allowed_unary_ops):
            value = self.visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, self.allowed_binary_ops):
            left = self.visit(node.left)
            right = self.visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if math.isclose(right, 0.0, abs_tol=1e-12) and isinstance(node.op, ast.Div):
                raise ZeroDivisionError("division by zero")
            return left / right
        raise ValueError(f"unsupported syntax: {ast.dump(node)}")


def extract_answer_text(completion: str) -> tuple[bool, str]:
    match = ANSWER_PATTERN.search(completion)
    if not match:
        return False, ""
    return True, match.group(1).strip()


def extract_numbers(expression: str) -> list[int]:
    return [int(token) for token in re.findall(r"\d+", expression)]


def normalize_answer_text(answer_text: str) -> str:
    normalized = answer_text.strip()
    if normalized.count("=") == 1:
        normalized = normalized.split("=", maxsplit=1)[0].strip()
    return normalized


def verify_completion(completion: str, numbers: list[int], tolerance: float = 1e-6) -> VerificationResult:
    has_answer_tag, answer_text = extract_answer_text(completion)
    if not has_answer_tag:
        return VerificationResult(False, "", False, False, False, False, None, "missing_answer_tag")

    answer_text = normalize_answer_text(answer_text)
    valid_chars = bool(ALLOWED_CHARS_PATTERN.fullmatch(answer_text))
    if not valid_chars:
        return VerificationResult(True, answer_text, False, False, False, False, None, "invalid_chars")

    used_numbers_match = Counter(extract_numbers(answer_text)) == Counter(numbers)
    try:
        tree = ast.parse(answer_text, mode="eval")
        value = SafeEvalVisitor().visit(tree)
        valid_syntax = True
    except Exception as exc:  # noqa: BLE001
        return VerificationResult(
            True,
            answer_text,
            valid_chars,
            False,
            used_numbers_match,
            False,
            None,
            exc.__class__.__name__,
        )

    evaluates_to_24 = math.isclose(value, 24.0, abs_tol=tolerance)
    return VerificationResult(
        True,
        answer_text,
        valid_chars,
        valid_syntax,
        used_numbers_match,
        evaluates_to_24,
        value,
        None,
    )
