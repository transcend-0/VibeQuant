"""Shared test fixtures.

parse_prompt / refine_task / auto_optimize are LLM-backed with no rule
fallback (see src/intent.py, src/research/refine.py). Tests exercise the
plumbing (JSON -> YAML -> TaskSpec -> validation, error handling) against
a scripted fake client rather than a real model call.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeLLMClient:
    """Stand-in for src.llm.LLMClient with a scripted reply."""

    model_name = "fake-llm"

    def __init__(self, responder):
        self._responder = responder  # callable(user_prompt, system_prompt) -> str

    def query(self, user_prompt, system_prompt=None, return_reasoning_content=False):
        reply = self._responder(user_prompt, system_prompt)
        if return_reasoning_content:
            return reply, ""
        return reply


# Modules that call get_client() directly (imported by name, so the lookup
# must be patched on each module, not on src.llm).
_GET_CLIENT_SITES = (
    "src.intent",
    "src.research.refine",
    "src.research.auto_optimize",
)


@pytest.fixture
def fake_llm(monkeypatch):
    """fake_llm(responder) installs a FakeLLMClient everywhere get_client()
    is looked up. responder(user_prompt, system_prompt) -> reply string.
    fake_llm(None) simulates "LLM not configured" (get_client() -> None)."""

    def _install(responder):
        client = None if responder is None else FakeLLMClient(responder)
        for modpath in _GET_CLIENT_SITES:
            mod = importlib.import_module(modpath)
            monkeypatch.setattr(mod, "get_client", lambda: client)
        return client

    return _install
