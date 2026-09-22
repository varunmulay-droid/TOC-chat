"""
toc/cfg.py
==========
A small Context-Free Grammar parser for EXPLICIT command syntax only.

Grammar (informal EBNF):

    command      ::= "/" command_name [ ":" ] WS argument
    command_name ::= "calc" | "remember" | "forget" | "search" | "run"
    argument     ::= any remaining text (delegated to the tool itself)

Deliberately narrow scope: this parser ONLY fires on text that already
matched FastIntent.SLASH_COMMAND in the NFA pre-scan (see nfa.py). It never
runs against, and never blocks, natural language chat. This is the fix for
the failure mode where a brittle grammar parser rejects valid queries that
don't match expected syntax -- free-form chat simply never reaches this
parser at all.
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class CommandName(Enum):
    CALC = "calc"
    REMEMBER = "remember"
    FORGET = "forget"
    SEARCH = "search"
    RUN = "run"


@dataclass
class ParsedCommand:
    name: CommandName
    argument: str
    raw: str


class CFGParseError(Exception):
    """Raised only for text that looks like a command but is malformed.
    Never raised for plain chat -- see module docstring."""
    pass


_COMMAND_RE = re.compile(
    r"^\s*/(?P<name>calc|remember|forget|search|run)\s*:?\s*(?P<arg>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def parse(text: str) -> Optional[ParsedCommand]:
    """Parse an explicit slash-command.

    Returns None if `text` doesn't look like a command at all (caller
    should treat it as plain chat). Raises CFGParseError only when the
    text clearly starts a command but is malformed (e.g. unknown command
    name, empty argument) -- this narrow, explicit failure is safe because
    it only ever fires on text the user deliberately prefixed with "/".
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    m = _COMMAND_RE.match(stripped)
    if not m:
        raise CFGParseError(f"Text starts with '/' but doesn't match known command syntax: {text!r}")

    name_str = m.group("name").lower()
    arg = m.group("arg").strip()

    if not arg:
        raise CFGParseError(f"Command '/{name_str}' requires an argument, got none.")

    return ParsedCommand(name=CommandName(name_str), argument=arg, raw=text)


if __name__ == "__main__":
    ok_tests = ["/calc 2 + 2", "/remember: my birthday is in June", "/search vector databases"]
    bad_tests = ["/unknown foo", "/calc", "just chatting normally"]

    for t in ok_tests:
        print(t, "->", parse(t))

    for t in bad_tests:
        try:
            result = parse(t)
            print(t, "-> (no command detected)" if result is None else result)
        except CFGParseError as e:
            print(t, "-> parse error (expected):", e)
