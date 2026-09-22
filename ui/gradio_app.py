"""
ui/gradio_app.py
=================
Gradio chat UI wired to the Router. Shows the TOC state trace and the
path taken (chat / rag / memory / tool:calc / etc.) alongside the
response -- useful for both debugging and demoing the TOC layer working.
"""

import uuid
import gradio as gr

from tools.router import Router
from toc.state_manager import GLOBAL_STATE_MANAGER

router = Router()


def respond(message: str, history, session_id: str):
    dfa = GLOBAL_STATE_MANAGER.get(session_id)
    result = router.handle_turn(message, dfa=dfa)
    debug_line = f"\n\n*[path: {result.path_taken} | states: {result.state_trace}]*"
    return result.response_text + debug_line


def build_app() -> gr.Blocks:
    with gr.Blocks(title="TOC-GPT") as demo:
        gr.Markdown(
            "# TOC-GPT\n"
            "A Theory-of-Computation-guided mini conversational AI. "
            "Try plain chat, a bare math expression like `12 * (4+1)`, "
            "or explicit commands: `/calc`, `/remember:`, `/search`, `/run`."
        )
        session_id = gr.State(str(uuid.uuid4()))
        gr.ChatInterface(fn=respond, additional_inputs=[session_id])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
