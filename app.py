"""
app.py
======
Top-level entrypoint. Hugging Face Spaces looks for `app.py` by default
(with Gradio SDK), and this also works for local `python app.py` runs.
"""

from ui.gradio_app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch()
