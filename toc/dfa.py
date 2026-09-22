"""
toc/dfa.py
==========
Deterministic Finite Automaton used as the conversation STATE MANAGER.

This is not decorative — every turn of the conversation actually moves
through these states, and the router (see tools/router.py) queries the
current state to decide what's legal to do next.

States:
    IDLE            -> waiting for user input
    AWAITING_INPUT   -> input received, about to classify it
    ROUTING         -> intent classification in progress
    TOOL_CALL       -> a tool (calculator / python) is executing
    RAG_LOOKUP      -> a FAISS document retrieval is in progress
    MEMORY_LOOKUP   -> a FAISS memory retrieval is in progress
    RESPONDING      -> LLM is generating the final response
    ERROR           -> something failed; recoverable, transitions back to IDLE

Transitions are defined explicitly. An illegal transition raises
InvalidTransitionError instead of silently doing the wrong thing --
this is the whole point of using a real DFA instead of an ad hoc
if/else chain: illegal states are structurally impossible.
"""

from enum import Enum, auto
from typing import Dict, Set


class ConvState(Enum):
    IDLE = auto()
    AWAITING_INPUT = auto()
    ROUTING = auto()
    TOOL_CALL = auto()
    RAG_LOOKUP = auto()
    MEMORY_LOOKUP = auto()
    RESPONDING = auto()
    ERROR = auto()


class InvalidTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""
    pass


# Transition table: current_state -> set of legal next states
TRANSITIONS: Dict[ConvState, Set[ConvState]] = {
    ConvState.IDLE: {ConvState.AWAITING_INPUT},
    ConvState.AWAITING_INPUT: {ConvState.ROUTING, ConvState.ERROR},
    ConvState.ROUTING: {
        ConvState.TOOL_CALL,
        ConvState.RAG_LOOKUP,
        ConvState.MEMORY_LOOKUP,
        ConvState.RESPONDING,  # plain chat, no retrieval needed
        ConvState.ERROR,
    },
    ConvState.TOOL_CALL: {ConvState.RESPONDING, ConvState.ERROR},
    ConvState.RAG_LOOKUP: {ConvState.RESPONDING, ConvState.ERROR},
    ConvState.MEMORY_LOOKUP: {ConvState.RESPONDING, ConvState.RAG_LOOKUP, ConvState.ERROR},
    ConvState.RESPONDING: {ConvState.IDLE, ConvState.ERROR},
    ConvState.ERROR: {ConvState.IDLE},
}


class ConversationDFA:
    """A tiny, explicit DFA driving one conversation turn at a time."""

    def __init__(self):
        self.state: ConvState = ConvState.IDLE
        self.history: list[ConvState] = [self.state]

    def can_transition(self, next_state: ConvState) -> bool:
        return next_state in TRANSITIONS.get(self.state, set())

    def transition(self, next_state: ConvState) -> ConvState:
        if not self.can_transition(next_state):
            raise InvalidTransitionError(
                f"Illegal transition: {self.state.name} -> {next_state.name}. "
                f"Legal targets from {self.state.name}: "
                f"{[s.name for s in TRANSITIONS.get(self.state, set())]}"
            )
        self.state = next_state
        self.history.append(next_state)
        return self.state

    def reset(self):
        self.state = ConvState.IDLE
        self.history = [self.state]

    def trace(self) -> str:
        """Human-readable trace of this turn's state path, useful for debugging/demo."""
        return " -> ".join(s.name for s in self.history)


if __name__ == "__main__":
    # Quick self-test / demo of a full turn
    dfa = ConversationDFA()
    dfa.transition(ConvState.AWAITING_INPUT)
    dfa.transition(ConvState.ROUTING)
    dfa.transition(ConvState.RAG_LOOKUP)
    dfa.transition(ConvState.RESPONDING)
    dfa.transition(ConvState.IDLE)
    print("Trace:", dfa.trace())

    try:
        dfa.transition(ConvState.RESPONDING)  # illegal from IDLE
    except InvalidTransitionError as e:
        print("Correctly rejected illegal transition:", e)
