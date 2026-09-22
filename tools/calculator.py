"""
tools/calculator.py
====================
Safe(r) arithmetic evaluator. Every expression is first checked by the
PDA bracket validator (toc/pda_bracket_validator.py) -- this is the real
use of the PDA module, not decoration: it rejects malformed nesting
BEFORE the expression reaches any eval-adjacent code.

Uses Python's `ast` module to evaluate only a whitelisted set of numeric
operations -- never a raw eval() of user input.
"""

import ast
import operator
from dataclasses import dataclass
from typing import Optional

from toc.pda_bracket_validator import validate_brackets, ValidationResult

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@dataclass
class CalcResult:
    ok: bool
    value: Optional[float] = None
    error: Optional[str] = None


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def calculate(expression: str) -> CalcResult:
    # Step 1: PDA validation of bracket nesting -- fails fast on malformed input
    bracket_check = validate_brackets(expression)
    if not bracket_check.ok:
        return CalcResult(
            ok=False,
            error=f"Unbalanced brackets ({bracket_check.result.name}) at position {bracket_check.position}",
        )

    # Step 2: parse + restricted eval via AST whitelist (no raw eval())
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
        return CalcResult(ok=True, value=value)
    except Exception as e:
        return CalcResult(ok=False, error=str(e))


if __name__ == "__main__":
    tests = ["2 + 3 * 4", "(2 + 3) * 4", "(2 + 3 * 4", "__import__('os')"]
    for t in tests:
        print(t, "->", calculate(t))
