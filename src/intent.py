"""Intent parsing: natural language (EN/ZH) -> TaskSpec.

LLM-backed by design (Vibe Quant principle #1, Intent-Driven): the model
reads the free-form prompt and proposes a complete task, self-disclosing
whatever it guessed or defaulted in `clarifications` so the caller (CLI /
Web UI) can show the resulting YAML for confirmation before running
(Plan-and-Act). Every proposal is re-validated against the DSL
(`TaskSpec.from_yaml` -> `TaskSpec.validate`) before it reaches the
caller, so a malformed or hallucinated task is rejected rather than
silently accepted.

There is no keyword/regex fallback: if the LLM is unconfigured, a
malformed/hallucinated reply is retried up to 3 times (`query_structured`,
feeding the validation error back so the model can self-correct), and if
it's still invalid after that (or the LLM is unreachable), `parse_prompt`
raises `IntentError` rather than degrading to a guess. Callers must
surface that clearly (not swallow it) — a misconfigured `config/llm.yaml`
should be loud, not silently "work" on a worse code path.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .dsl import DSLError, TaskSpec
from .llm import LLMError, get_client, query_structured
from .research.llm_ideas import _expression_valid, _FUNCTION_CATALOG
from .strategies import REGISTRY


class IntentError(RuntimeError):
    pass


@dataclass
class ParseResult:
    spec: TaskSpec
    clarifications: List[str] = field(default_factory=list)
    recognized: bool = True  # the LLM was confident this describes a concrete strategy


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        import re

        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text


def _strategy_catalog() -> str:
    lines = []
    for name, tpl in sorted(REGISTRY.items()):
        params = ", ".join(sorted(tpl.defaults)) or "(no params)"
        lines.append(f"  - {name}{{{params}}}: {tpl.summary_en}")
    return "\n".join(lines)


_SYSTEM_TEMPLATE = """You are the intent-parsing layer of VibeQuant, an AI-native
quantitative trading system. You turn a researcher's free-form request (English
or Chinese) into ONE complete task for the system's YAML DSL.

Strategy templates available (name{{params}}: description):
{catalog}

DSL rules you must respect:
- top-level: name, intent, kind ("strategy" for this endpoint), data, strategy,
  execution, report. Do not invent keys.
- data.source: "etf"|"stock"|"index"|"hk"|"us"|"crypto"|"synthetic"|"akshare".
  Use "etf" for A-share ETF codes (5xxxxx/1[5-6]xxxx), "stock" for A-share
  stock codes (6-digit), "synthetic" only when no real market/symbol is
  implied (use symbol "DEMO"). Only pick "akshare" if the user explicitly
  asks for it.
- data.universe: "pool24" ONLY if the user asks for the curated/built-in
  24-ETF pool (24支ETF池/内置24ETF); otherwise "custom".
- data.symbols: explicit list of ticker strings. CRITICAL: always QUOTE
  every symbol in the YAML (e.g. "000001", "600000") — an unquoted
  leading-zero number is parsed as octal by YAML and silently corrupts the
  code. If data.universe is "pool24", just put ["DEMO"] here — the caller
  overwrites it with the real 24-ETF list, don't try to guess the codes.
- data.start/end: "YYYY-MM-DD". Resolve relative ranges ("past 3 years",
  "近5年") against today's date, given below. If nothing is stated or
  implied, default to the last 3 years ending today.
- execution.initial_cash: a positive number (parse "100万" -> 1000000,
  "50k"/"50K" -> 50000, "2m" -> 2000000). Default 1000000 if unstated.
- report.language: "zh" if the request is in Chinese, else "en".
- strategy.params is ALWAYS a flat mapping of the template's own param names
  (from the catalog above) to numbers — never a nested object, never a key
  named "template", "config", or anything else. Two exceptions:
  - strategy.name: "custom" (see below), whose params are "source" (a
    string) and "warmup" (an int).
  - strategy.name: "factor_rotation", whose params.expressions is a list
    of factor definitions, each formatted EXACTLY as "Name = Expr" using
    ONLY these functions: {func_catalog}
    over Open/High/Low/Close/Volume, plus +,-,*,/ and numeric constants —
    e.g. "Mom20 = Delta(Close, 20) / Delay(Close, 20)". Do NOT write a bare
    label like "momentum_20" — it must be a real, evaluable expression.

Only two strategy templates exist: "factor_rotation" for genuinely
cross-sectional multi-factor strategies (score every symbol, hold the
top-K, rebalance periodically), and "custom" for literally everything
else — moving averages, RSI, momentum, Bollinger bands, buy-and-hold,
statistical/ML models (regression, classification, any model fitting),
or any other single-symbol rule. There is no library of prebuilt rule
templates: write the logic yourself in "custom" rather than trying to
name a template that doesn't exist. For "custom":
- params.source: complete Python source defining exactly one function:
    def signal(closes, position):
        ...
  `closes` is a list of floats (chronological closes up to and including
  the current bar). `position` is the currently held QUANTITY for this
  symbol (0 if flat, positive if long — not a weight or a percentage).
  Return a new TARGET WEIGHT in [0, 1] (fraction of equity) to change
  position, or None to leave it unchanged — note the input and output
  units differ (quantity in, weight out); test `position <= 0` /
  `position > 0` for flat/long, don't compare it to a fraction like 0.5.
  `numpy` and `pandas` are installed and may be imported and used freely
  (e.g. `import numpy as np`) — this is a real Python execution
  environment, not a restricted expression language; write whatever
  computation (including model fitting) the request actually needs.
  `scikit-learn`/`scipy` are NOT installed — implement any modeling you
  need (regression, etc.) directly with numpy.
- params.warmup: how many leading bars to skip before your first signal
  needs a full window (e.g. 20 if you compute a 20-bar rolling average).
- Write params.source as a YAML literal block scalar (the `|` syntax
  below) so indentation and newlines survive — never inline/escaped.

Example of a complete, correctly-shaped task_yaml value (note the `|`
block scalar for params.source):
name: "ma-cross-demo"
intent: "5/20 day moving-average crossover on synthetic DEMO"
kind: "strategy"
data:
  source: "synthetic"
  universe: "custom"
  symbols: ["DEMO"]
  start: "2022-01-01"
  end: "2024-12-31"
strategy:
  name: "custom"
  params:
    warmup: 21
    source: |
      def signal(closes, position):
          if len(closes) < 21:
              return None
          fast = sum(closes[-5:]) / 5
          slow = sum(closes[-20:]) / 20
          fast_prev = sum(closes[-6:-1]) / 5
          slow_prev = sum(closes[-21:-1]) / 20
          if fast_prev <= slow_prev and fast > slow:
              return 1.0
          if fast_prev >= slow_prev and fast < slow:
              return 0.0
          return None
execution:
  initial_cash: 1000000
report:
  language: "en"

Reply with ONLY JSON (no markdown fence):
{{"task_yaml": "<complete YAML task, top-level keys as above>",
 "clarifications": ["<one short sentence per default/guess you made, in the
 request's language, e.g. 'No date range recognized; defaulting to ...'>"],
 "recognized": true|false}}

Set "recognized" to true only if the request describes a concrete,
tradeable strategy/instruction. Set it to false if it reads like a vague
goal, a research question, or something that needs literature analysis
rather than a direct strategy build — but still propose your best-effort
task with clarifications explaining the gaps.

Today's date: {today}"""


def _parse_task(reply: str) -> Dict[str, Any]:
    try:
        payload = json.loads(_strip_fence(reply))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"reply was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not payload.get("task_yaml"):
        raise ValueError("reply JSON must be an object with a non-empty 'task_yaml' field")
    try:
        spec = TaskSpec.from_yaml(str(payload["task_yaml"]))
    except DSLError as exc:
        raise ValueError(f"task_yaml failed DSL validation: {exc}") from exc

    # TaskSpec.validate() only checks factor.expressions for kind="factor";
    # factor_rotation's expressions live under strategy.params and aren't
    # DSL-validated, so a bare label like "momentum_20" (not a real
    # expression) would only fail at backtest time. Catch it here instead,
    # so query_structured's retry loop can make the model fix it.
    if spec.strategy.name == "factor_rotation":
        bad = [
            e for e in spec.strategy.params.get("expressions", [])
            if not _expression_valid(str(e))
        ]
        if bad:
            raise ValueError(
                f"strategy.params.expressions contains invalid factor "
                f"expressions (not \"Name = Expr\" using the allowed "
                f"functions): {bad}"
            )

    payload["_spec"] = spec
    return payload


def parse_prompt(prompt: str) -> ParseResult:
    prompt = prompt.strip()
    client = get_client()
    if client is None:
        raise IntentError(
            "LLM not configured — set up config/llm.yaml before generating "
            "strategies from natural language."
        )

    system_prompt = _SYSTEM_TEMPLATE.format(
        catalog=_strategy_catalog(),
        func_catalog=_FUNCTION_CATALOG,
        today=_dt.date.today().isoformat(),
    )
    try:
        payload = query_structured(client, prompt, system_prompt, _parse_task)
    except LLMError as exc:
        raise IntentError(f"LLM intent parsing failed: {exc}") from exc
    except ValueError as exc:
        raise IntentError(f"LLM produced an invalid task after 3 attempts: {exc}") from exc

    spec = payload.pop("_spec")
    if not spec.intent:
        spec.intent = prompt
    if not spec.name or spec.name == "untitled-task":
        import re

        slug = re.sub(r"[^a-zA-Z0-9一-鿿]+", "-", prompt)[:32].strip("-")
        spec.name = slug or f"task-{_dt.date.today().isoformat()}"

    clarifications = [str(c) for c in (payload.get("clarifications") or [])]
    if spec.data.universe == "pool24":
        # the model is only asked to *select* pool24, not to enumerate its
        # symbols from memory — fill in the authoritative list ourselves.
        from .data_sources.etf_pool import POOL_SYMBOLS

        spec.data.symbols = list(POOL_SYMBOLS)
        spec.validate()
        clarifications.append(
            "已选用精选24ETF池作为标的池。" if spec.report.language == "zh"
            else "Using the curated 24-ETF pool as the universe."
        )
    recognized = bool(payload.get("recognized", True))
    return ParseResult(spec=spec, clarifications=clarifications, recognized=recognized)
