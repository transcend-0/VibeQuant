"""Tool registry: named, single-purpose steps the runner executes.

Each tool is `fn(ctx: RunContext) -> None`, reading and writing the
shared context. Tools never import akquant directly — engine access
goes through src.adapters.
"""

from __future__ import annotations

from typing import Callable, Dict

from .context import RunContext

ToolFn = Callable[[RunContext], None]

REGISTRY: Dict[str, ToolFn] = {}


def register(name: str) -> Callable[[ToolFn], ToolFn]:
    def deco(fn: ToolFn) -> ToolFn:
        REGISTRY[name] = fn
        return fn

    return deco


def get_tool(name: str) -> ToolFn:
    if name not in REGISTRY:
        raise KeyError(f"unknown tool {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


from . import steps  # noqa: E402,F401  (registers the built-in tools)
