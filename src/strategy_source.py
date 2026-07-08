"""Shape checks for LLM-authored `strategy.params` (source + expressions).

Used by both `intent.py` (fresh parse) and `research/refine.py` (revision)
so a malformed or hallucinated strategy is caught before it reaches the
user, and — during LLM generation — fed back into `query_structured`'s
retry loop so the model can self-correct instead of the caller silently
accepting something broken. This is a syntax/shape check only (`compile`,
never `exec`): the LLM's code is not run until the user reviews and
confirms the plan.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from .adapters.akquant_factor import known_expression_functions

_CLASS_RE = re.compile(r"class\s+\w+\s*\(\s*(Strategy|BaseStrategy)\s*\)")
_ALLOWED_EXPR_TOKENS = {f.lower() for f in known_expression_functions()} | {
    "open", "high", "low", "close", "volume",
}
_EXPR_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _expression_valid(raw: str) -> bool:
    """Same check as research/llm_ideas._expression_valid, duplicated here
    (not imported) to avoid a circular import: llm_ideas also depends on
    this module for strategy-source validation."""
    body = raw.split("=", 1)[-1]
    return all(
        t.lower() in _ALLOWED_EXPR_TOKENS for t in _EXPR_TOKEN_RE.findall(body)
    )


class StrategySourceError(ValueError):
    pass


def validate_strategy_params(params: Dict[str, Any]) -> None:
    """Raise StrategySourceError if strategy.params is not usable."""
    params = params or {}  # an LLM reply with "params: null" leaves this None
    source = str(params.get("source") or "").strip()
    if not source:
        raise StrategySourceError("strategy.params.source is empty")
    try:
        compile(source, "<strategy_source>", "exec")
    except SyntaxError as exc:
        raise StrategySourceError(f"strategy.params.source has a syntax error: {exc}") from exc
    if not _CLASS_RE.search(source):
        raise StrategySourceError(
            "strategy.params.source must define a class inheriting from "
            "Strategy/BaseStrategy, e.g. 'class Strategy(BaseStrategy): "
            "def on_bar(self, bar): ...'"
        )
    expressions = params.get("expressions")
    if expressions:
        bad = [e for e in expressions if not _expression_valid(str(e))]
        if bad:
            raise StrategySourceError(
                f"strategy.params.expressions contains invalid factor "
                f"expressions (not \"Name = Expr\" using the allowed "
                f"functions): {bad}"
            )
