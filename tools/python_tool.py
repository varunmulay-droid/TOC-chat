"""
tools/python_tool.py
=====================
Restricted Python execution for simple user code snippets (data
manipulation, quick scripts) -- NOT a full sandbox, but restricts
builtins and enforces a wall-clock timeout. For a public-facing demo,
treat this as "best-effort restriction", not a security boundary; if
this ships publicly, prefer running it in an actually isolated
subprocess/container.
"""

import io
import contextlib
import signal
from dataclasses import dataclass
from typing import Optional

_SAFE_BUILTINS = {
    "print": print, "range": range, "len": len, "sum": sum, "min": min,
    "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "str": str, "int": int, "float": float, "bool": bool,
}


@dataclass
class PythonExecResult:
    ok: bool
    stdout: str = ""
    error: Optional[str] = None


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Execution timed out")


def run_python(code: str, timeout_seconds: int = 5) -> PythonExecResult:
    stdout_buffer = io.StringIO()
    restricted_globals = {"__builtins__": _SAFE_BUILTINS}

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, restricted_globals, {})
        return PythonExecResult(ok=True, stdout=stdout_buffer.getvalue())
    except _TimeoutError as e:
        return PythonExecResult(ok=False, error=str(e))
    except Exception as e:
        return PythonExecResult(ok=False, stdout=stdout_buffer.getvalue(), error=str(e))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    print(run_python("print('hello from restricted python')\nprint(sum(range(10)))"))
    print(run_python("import os"))  # should fail: os not in safe builtins/importable
    print(run_python("while True: pass", timeout_seconds=2))  # should time out
