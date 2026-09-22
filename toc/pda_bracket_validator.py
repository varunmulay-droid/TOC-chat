"""
toc/pda_bracket_validator.py
=============================
Pushdown Automaton, given ONE concrete, well-scoped job: validate that
brackets/parens/braces in an expression are properly nested and balanced
BEFORE handing the expression to the calculator or python execution tool.

This replaces the vague "PDA for nested structures" framing with something
that (a) is a genuine PDA -- a stack-based acceptor for a language that a
plain DFA cannot recognize (balanced brackets require unbounded memory of
nesting depth), and (b) does real, useful work: it stops malformed
expressions from ever reaching eval()/exec()-adjacent tool code, which is
also a small security/robustness win independent of the TOC framing.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ValidationResult(Enum):
    VALID = auto()
    UNBALANCED_UNCLOSED = auto()   # opened but never closed
    UNBALANCED_UNOPENED = auto()   # closed without a matching open
    MISMATCHED_TYPE = auto()       # e.g. "(]" -- wrong closer for the opener


_PAIRS = {")": "(", "]": "[", "}": "{"}
_OPENERS = set(_PAIRS.values())
_CLOSERS = set(_PAIRS.keys())


@dataclass
class PDAResult:
    result: ValidationResult
    position: Optional[int] = None  # index of the offending character, if any

    @property
    def ok(self) -> bool:
        return self.result == ValidationResult.VALID


def validate_brackets(expr: str) -> PDAResult:
    """Stack-based (PDA) acceptor for the balanced-bracket language.

    A DFA cannot decide this language in general because it would need
    unbounded memory to track arbitrary nesting depth -- that's precisely
    why a pushdown automaton (finite control + a stack) is the right,
    non-decorative tool here.
    """
    stack: list[str] = []

    for i, ch in enumerate(expr):
        if ch in _OPENERS:
            stack.append(ch)
        elif ch in _CLOSERS:
            if not stack:
                return PDAResult(ValidationResult.UNBALANCED_UNOPENED, position=i)
            top = stack.pop()
            if top != _PAIRS[ch]:
                return PDAResult(ValidationResult.MISMATCHED_TYPE, position=i)

    if stack:
        # Report the position of the earliest unclosed opener
        return PDAResult(ValidationResult.UNBALANCED_UNCLOSED, position=expr.index(stack[0]))

    return PDAResult(ValidationResult.VALID)


if __name__ == "__main__":
    tests = [
        "12 * (4 + 1)",
        "(2 + 3",
        "2 + 3)",
        "(2 + [3 * 4)]",
        "[(1+2) * (3+4)]",
    ]
    for t in tests:
        print(f"{t!r:25} -> {validate_brackets(t)}")
