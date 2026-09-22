"""
tools/router.py
================
The actual pipeline glue. Implements the collapsed architecture:

    DFA state check
      -> NFA fast intent scan
           -> CFG command parse (only if NFA flagged SLASH_COMMAND)
                -> Tool path (PDA-validated calc / python)
           -> else: MiniLM-based semantic routing among {chat, rag, memory}
      -> Context builder
      -> LLM generate
      -> memory write

Critically: the rule-based layers (NFA/CFG) only ever FAST-PATH obvious
cases or execute explicit commands. They never hard-reject natural
language -- anything that isn't an explicit command or bare math
expression always falls through to the LLM/semantic path.
"""

from dataclasses import dataclass
from typing import Optional

from toc.dfa import ConversationDFA, ConvState
from toc.nfa import scan, FastIntent
from toc.cfg import parse as cfg_parse, CFGParseError, CommandName
from tools.calculator import calculate
from tools.python_tool import run_python
from models.encoder import MiniLMEncoder
from models.llm import QwenLLM
from memory.memory_manager import MemoryManager
from rag.retriever import Retriever


@dataclass
class RouteResult:
    response_text: str
    path_taken: str        # for debugging/demo: "tool:calc", "rag", "memory", "chat", etc.
    state_trace: str


class Router:
    def __init__(
        self,
        llm: Optional[QwenLLM] = None,
        encoder: Optional[MiniLMEncoder] = None,
        memory: Optional[MemoryManager] = None,
        retriever: Optional[Retriever] = None,
    ):
        self.llm = llm or QwenLLM()
        self.encoder = encoder or MiniLMEncoder()
        self.memory = memory or MemoryManager(encoder=self.encoder)
        self.retriever = retriever or Retriever(encoder=self.encoder)

    def handle_turn(self, user_text: str, dfa: Optional[ConversationDFA] = None) -> RouteResult:
        dfa = dfa or ConversationDFA()
        dfa.transition(ConvState.AWAITING_INPUT)
        dfa.transition(ConvState.ROUTING)

        # --- NFA fast pre-scan ---
        nfa_matches = scan(user_text)
        matched_intents = {m.intent for m in nfa_matches}

        # --- Explicit command path (CFG), only if NFA flagged a slash command ---
        if FastIntent.SLASH_COMMAND in matched_intents:
            try:
                command = cfg_parse(user_text)
            except CFGParseError as e:
                dfa.transition(ConvState.ERROR)
                dfa.transition(ConvState.IDLE)
                return RouteResult(
                    response_text=f"Couldn't parse that command: {e}",
                    path_taken="cfg:error",
                    state_trace=dfa.trace(),
                )

            dfa.transition(ConvState.TOOL_CALL)
            result_text, path = self._run_command(command)
            dfa.transition(ConvState.RESPONDING)
            dfa.transition(ConvState.IDLE)
            self.memory.remember(user_text, result_text)
            return RouteResult(response_text=result_text, path_taken=path, state_trace=dfa.trace())

        # --- Bare math expression fast path (no slash needed) ---
        if FastIntent.MATH_EXPRESSION in matched_intents:
            dfa.transition(ConvState.TOOL_CALL)
            calc_result = calculate(user_text)
            response = str(calc_result.value) if calc_result.ok else f"Calculation error: {calc_result.error}"
            dfa.transition(ConvState.RESPONDING)
            dfa.transition(ConvState.IDLE)
            self.memory.remember(user_text, response)
            return RouteResult(response_text=response, path_taken="tool:calc", state_trace=dfa.trace())

        # --- Everything else: semantic path (RAG / memory / plain chat), LLM decides tone ---
        context_chunks = []
        path = "chat"

        if FastIntent.DOC_QUESTION in matched_intents:
            dfa.transition(ConvState.RAG_LOOKUP)
            context_chunks = self.retriever.retrieve(user_text)
            path = "rag"

        if FastIntent.MEMORY_RECALL in matched_intents:
            if dfa.state != ConvState.RAG_LOOKUP:
                dfa.transition(ConvState.MEMORY_LOOKUP)
            recalled = self.memory.recall(user_text)
            context_chunks.extend(recalled)
            path = "memory" if path == "chat" else f"{path}+memory"

        if dfa.state == ConvState.ROUTING:
            dfa.transition(ConvState.RESPONDING)
        else:
            dfa.transition(ConvState.RESPONDING)

        prompt = user_text
        system_prompt = "You are TOC-GPT, a helpful assistant."
        if context_chunks:
            joined_context = "\n---\n".join(context_chunks)
            system_prompt += f"\nRelevant context:\n{joined_context}"

        response_text = self.llm.generate(prompt, system_prompt=system_prompt)

        dfa.transition(ConvState.IDLE)
        self.memory.remember(user_text, response_text)
        return RouteResult(response_text=response_text, path_taken=path, state_trace=dfa.trace())

    def _run_command(self, command) -> tuple[str, str]:
        if command.name == CommandName.CALC:
            result = calculate(command.argument)
            text = str(result.value) if result.ok else f"Calculation error: {result.error}"
            return text, "tool:calc"

        if command.name == CommandName.RUN:
            result = run_python(command.argument)
            text = result.stdout if result.ok else f"Execution error: {result.error}"
            return text, "tool:python"

        if command.name == CommandName.REMEMBER:
            self.memory.remember(command.argument, "(explicitly saved by user)")
            return "Got it, I'll remember that.", "tool:memory_write"

        if command.name == CommandName.SEARCH:
            chunks = self.retriever.retrieve(command.argument)
            text = "\n---\n".join(chunks) if chunks else "No relevant documents found."
            return text, "tool:rag_search"

        return f"Command '{command.name.value}' not yet implemented.", "tool:unimplemented"
