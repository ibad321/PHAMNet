"""Small helpers: console/file logging and fault-tolerant calls."""
from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Optional


class Reporter:
    """Prints messages and keeps them so that they can be written to ``REPORT.txt``."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, *args) -> None:
        text = " ".join(str(a) for a in args)
        print(text, flush=True)
        self.lines.append(text)

    def header(self, title: str) -> None:
        self("\n" + "=" * 72 + "\n## " + title + "\n" + "=" * 72)

    def write(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("\n".join(self.lines))


def safe_call(report: Reporter, title: str, fn: Callable, *args, **kwargs) -> Optional[object]:
    """Run ``fn`` and log, instead of raising, when an optional analysis step fails."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - optional steps must not abort a training session
        report(f"[{title}] skipped: {type(exc).__name__}: {exc}")
        report.lines.append(traceback.format_exc(limit=4))
        return None


def python_version() -> str:
    return sys.version.split()[0]
