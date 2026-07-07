"""User-configurable LLM client (OpenAI-compatible).

Modeled on Strategy/TextGrad/src/llm_client.py: one client class over the
`openai` SDK with api_key / base_url / model, chat-completions for normal
models and the responses API for gpt-5.* series.

Configuration lives in config/llm.yaml (gitignored):

    model: "gpt-5-mini"
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"

The execution core (engines, validation, reports) never touches the LLM;
the research-intelligence layer (intent parsing, idea extraction, refine,
auto-optimize) is LLM-required with no rule-based fallback. Callers must
surface LLMError clearly (a 5xx to the UI), never swallow it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, TypeVar

import yaml

from ..config import config_dir

logger = logging.getLogger(__name__)

_CONFIG_NAME = "llm.yaml"


class LLMError(RuntimeError):
    pass


def _config_file():
    return config_dir() / _CONFIG_NAME


def load_llm_config() -> Optional[Dict[str, str]]:
    """Read config/llm.yaml; None when absent or incomplete."""
    path = _config_file()
    if not path.exists():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    conf = {
        "model": str(raw.get("model", "")).strip(),
        "api_key": str(raw.get("api_key", "")).strip(),
        "base_url": str(raw.get("base_url", "")).strip(),
    }
    if not (conf["model"] and conf["api_key"] and conf["base_url"]):
        return None
    return conf


def save_llm_config(model: str, api_key: str, base_url: str) -> None:
    _config_file().write_text(
        yaml.safe_dump(
            {"model": model, "api_key": api_key, "base_url": base_url},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class LLMClient:
    """Thin wrapper over the OpenAI SDK (chat-completions / responses)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        chat_args: Optional[Dict[str, Any]] = None,
        # A real 30-page Chinese factor research report (~70k chars of
        # dense, table-heavy pdftotext output) measured at 506s for one
        # idea-extraction call — 60s was failing outright on exactly the
        # documents this feature exists for. This is generous, not a
        # guarantee: a large enough/slower document can still exceed it.
        timeout: float = 600.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("`pip install openai` to enable LLM features") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model_name = model_name
        self.chat_args = dict(chat_args or {})

    def _query_chat_completions(self, user_prompt: str, system_prompt: Optional[str]):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **self.chat_args
        )
        message = response.choices[0].message
        text = (message.content or "").strip()
        reasoning = (getattr(message, "reasoning_content", "") or "").strip()
        return text, reasoning

    def _query_responses(self, user_prompt: str, system_prompt: Optional[str]):
        inputs = []
        if system_prompt:
            inputs.append({"role": "system", "content": system_prompt})
        inputs.append({"role": "user", "content": user_prompt})
        response = self.client.responses.create(
            model=self.model_name, input=inputs, **self.chat_args
        )
        return (response.output_text or "").strip(), ""

    def query(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        return_reasoning_content: bool = False,
    ):
        logger.info("Querying %s", self.model_name)
        try:
            # gpt-5.* series speaks the responses API (same routing as the
            # TextGrad reference client); everything else chat-completions
            if self.model_name.startswith("gpt-5."):
                text, reasoning = self._query_responses(user_prompt, system_prompt)
            else:
                text, reasoning = self._query_chat_completions(
                    user_prompt, system_prompt
                )
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        if return_reasoning_content:
            return text, reasoning
        return text


_T = TypeVar("_T")


def query_structured(
    client: LLMClient,
    user_prompt: str,
    system_prompt: Optional[str],
    parse: Callable[[str], _T],
    max_attempts: int = 3,
) -> _T:
    """Query the LLM and apply `parse` to the reply, retrying on a malformed
    or hallucinated reply — `parse` should raise ValueError with a message
    describing what was wrong, which is fed back to the model so it can
    self-correct on the next attempt.

    LLMError (connection/auth/timeout — the model is unreachable, not
    wrong) is NOT retried: retrying a broken `base_url` three times would
    just make a misconfigured system fail three times slower instead of
    failing loudly, which defeats the point of not having a fallback.
    """
    prompt = user_prompt
    last_exc: Optional[ValueError] = None
    for attempt in range(1, max_attempts + 1):
        reply = client.query(prompt, system_prompt=system_prompt)
        try:
            return parse(reply)
        except ValueError as exc:
            last_exc = exc
            logger.warning("LLM reply invalid (attempt %d/%d): %s", attempt, max_attempts, exc)
            prompt = (
                f"{user_prompt}\n\n"
                f"Your previous reply was invalid: {exc}\n"
                "Reply again with ONLY the corrected output in the exact "
                "format requested — no markdown fence, no explanation."
            )
    assert last_exc is not None
    raise last_exc


def get_client() -> Optional[LLMClient]:
    """Client from config/llm.yaml, or None when LLM is unconfigured."""
    conf = load_llm_config()
    if conf is None:
        return None
    return LLMClient(
        api_key=conf["api_key"],
        base_url=conf["base_url"],
        model_name=conf["model"],
    )


def test_connection() -> Dict[str, Any]:
    """Round-trip check used by the UI's Test button and the CLI."""
    conf = load_llm_config()
    if conf is None:
        return {"ok": False, "error": "config/llm.yaml missing or incomplete"}
    import time

    started = time.monotonic()
    try:
        client = LLMClient(
            api_key=conf["api_key"],
            base_url=conf["base_url"],
            model_name=conf["model"],
        )
        reply = client.query(
            "Reply with exactly: pong", system_prompt="You are a health check."
        )
    except LLMError as exc:
        return {"ok": False, "model": conf["model"], "error": str(exc)}
    return {
        "ok": True,
        "model": conf["model"],
        "base_url": conf["base_url"],
        "latency_s": round(time.monotonic() - started, 2),
        "reply": reply[:200],
    }
