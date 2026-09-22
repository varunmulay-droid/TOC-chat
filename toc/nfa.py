"""
toc/nfa.py
==========
Nondeterministic Finite Automaton used as a FAST, CHEAP intent pre-scanner.

Purpose: before paying for a MiniLM embedding + similarity search, run a
cheap pattern scan that can flag "this is obviously a calculator query" or
"this is obviously a slash-command" or "this looks like a memory-write
request". It runs several candidate patterns "in parallel" (true NFA
non-determinism) and returns ALL matching intents -- it never hard-rejects
a query. If nothing matches, the caller falls through to semantic routing.

This module intentionally does NOT try to understand natural language. It
only flags strong, unambiguous surface patterns. Ambiguous input produces
an empty match set, which is the correct, safe outcome.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class FastIntent(Enum):
    SLASH_COMMAND = auto()      # e.g. "/calc 2+3", "/remember: ..."
    MATH_EXPRESSION = auto()    # bare arithmetic, e.g. "12 * (4 + 1)"
    DOC_QUESTION = auto()       # "summarize this document", "what does the pdf say"
    MEMORY_RECALL = auto()      # "what did I tell you about..."
    NONE = auto()


@dataclass
class NFAMatch:
    intent: FastIntent
    matched_text: str


# Each pattern is one "parallel path" in the NFA. A query can match
# multiple paths simultaneously (that's the non-determinism) -- the
# router decides priority, this module just reports what fired.
_PATTERNS = [
    (FastIntent.SLASH_COMMAND, re.compile(r"^\s*/\w+")),
    (FastIntent.MATH_EXPRESSION, re.compile(r"^\s*[\d\s\+\-\*/\.\(\)%\^]+\s*$")),
    (FastIntent.DOC_QUESTION, re.compile(
        r"\b(summarize|summarise|according to|in the (document|pdf|file)|what does the (doc|pdf|file) say)\b",
        re.IGNORECASE,
    )),
    (FastIntent.MEMORY_RECALL, re.compile(
        r"\b(what did i (say|tell you)|do you remember|earlier i (said|mentioned))\b",
        re.IGNORECASE,
    )),
]


def scan(text: str) -> List[NFAMatch]:
    """Run all candidate patterns against `text`, non-deterministically.

    Returns every match found (possibly zero, possibly several). An empty
    list means "no strong surface signal" -- the caller should fall through
    to semantic (MiniLM) routing rather than treating this as an error.
    """
    matches: List[NFAMatch] = []
    stripped = text.strip()
    if not stripped:
        return matches

    for intent, pattern in _PATTERNS:
        m = pattern.search(stripped)
        if m:
            matches.append(NFAMatch(intent=intent, matched_text=m.group(0)))

    return matches


if __name__ == "__main__":
    tests = [
        "/calc 2 + 2",
        "12 * (4 + 1)",
        "Can you summarize this document for me?",
        "What did I tell you about my last project?",
        "Hey, how's it going today?",
    ]
    for t in tests:
        print(f"{t!r:55} -> {scan(t)}")
