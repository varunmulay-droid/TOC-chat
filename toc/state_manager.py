"""
toc/state_manager.py
=====================
Thin session wrapper around ConversationDFA (see dfa.py). Keeps one DFA
instance per session_id so multiple concurrent Gradio users don't share
state.
"""

from typing import Dict
from toc.dfa import ConversationDFA


class StateManager:
    def __init__(self):
        self._sessions: Dict[str, ConversationDFA] = {}

    def get(self, session_id: str) -> ConversationDFA:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationDFA()
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].reset()

    def drop(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# Module-level singleton used by app.py / router.py
GLOBAL_STATE_MANAGER = StateManager()
