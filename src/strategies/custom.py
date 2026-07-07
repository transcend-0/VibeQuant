"""LLM-authored custom strategy: Python source, not a fixed template.

`params["source"]` is executed directly (compile + exec, no AST allowlist,
no sandbox) to obtain a `signal(closes, position)` function matching the
same contract as every other template in this package. This is an
accepted, explicit risk decision (2026-07-07): VibeQuant is a single-user,
127.0.0.1-only local tool, and the user chose direct exec over building a
static-analysis gate or process sandbox. The known exposure this leaves
open: `src/research/ingest.py` feeds arbitrary PDF/URL/text content into
the LLM, so adversarial content in an ingested source could in principle
prompt-inject the model into emitting malicious Python here, which would
then run with the full privileges of the local process (file/network
access, no restrictions). Do not expose this server beyond localhost
without revisiting this.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import StrategyTemplate, register

DEFAULT_SOURCE = """def signal(closes, position):
    return None
"""


def build(params: Dict[str, Any]) -> Callable[[List[float], float], Optional[float]]:
    source = str(params.get("source") or DEFAULT_SOURCE)
    namespace: Dict[str, Any] = {}
    exec(source, namespace)  # noqa: S102 -- intentionally unsandboxed, see module docstring
    fn = namespace.get("signal")
    if not callable(fn):
        raise KeyError(
            "custom strategy source must define a callable named "
            "'signal(closes, position)'"
        )
    return fn


register(
    StrategyTemplate(
        name="custom",
        summary_en="LLM-authored custom signal (Python source, no fixed template).",
        summary_zh="LLM 自定义信号（Python 源码，不受限于内置模板）。",
        defaults={"source": DEFAULT_SOURCE, "warmup": 0},
        build=build,
        warmup=lambda p: int(p.get("warmup", 0)),
    )
)
