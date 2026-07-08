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
from .research.llm_ideas import _FUNCTION_CATALOG
from .strategy_source import StrategySourceError, validate_strategy_params


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


_SYSTEM_TEMPLATE = """You are the intent-parsing layer of VibeQuant, an AI-native
quantitative trading system. You turn a researcher's free-form request (English
or Chinese) into ONE complete task for the system's YAML DSL.

DSL rules you must respect:
- top-level: name, intent, kind ("strategy" for this endpoint), data, strategy,
  execution, report. Do not invent keys.
- data.source: "etf"|"stock"|"index"|"hk"|"us"|"crypto"|"synthetic"|"akshare".
  Use "etf" for A-share ETF codes (5xxxxx/1[5-6]xxxx), "stock" for A-share
  stock codes (6-digit), "synthetic" only when no real market/symbol is
  implied (use symbol "DEMO"). Only pick "akshare" if the user explicitly
  asks for it.
- data.universe: "pool24" if the user asks for the curated/built-in 24-ETF
  pool (24支ETF池/内置24ETF) OR names no specific instrument/ticker/index at
  all (a vague/generic request with nothing concrete to trade) — pool24 is
  the system default universe. Use "custom" only when the request names
  actual ticker(s), an index, or another concrete instrument.
- data.symbols: explicit list of ticker strings. CRITICAL: always QUOTE
  every symbol in the YAML (e.g. "000001", "600000") — an unquoted
  leading-zero number is parsed as octal by YAML and silently corrupts the
  code. If data.universe is "pool24", just put ["DEMO"] here — the caller
  overwrites it with the real 24-ETF list, don't try to guess the codes.
- data.start/end: "YYYY-MM-DD". Resolve relative ranges ("past 3 years",
  "近5年") against today's date, given below. If nothing is stated or
  implied, default to start="2025-01-01", end="2025-12-31".
- execution.initial_cash: a positive number (parse "100万" -> 1000000,
  "50k"/"50K" -> 50000, "2m" -> 2000000). Default 1000000 if unstated.
- report.language: "zh" if the request is in Chinese, else "en".
- strategy.name: a short descriptive label only (e.g. "custom",
  "factor_rotation", "ma_cross") — purely informational, does not select a
  fixed template. ALL strategies execute the same way: see below.
- strategy.params.source: THE strategy. Complete Python source defining a
  class that inherits `Strategy` (already in scope — it IS akquant's own
  Strategy base class, imported for you), using akquant's real API
  directly — the same class you'd write against akquant itself, not a
  restricted callback. Available on `self` (a partial list; there is
  more, but this covers nearly everything a trading rule needs):
    self.get_position(symbol) -> float          current held quantity (0 if flat)
    self.get_portfolio_value() -> float         total account equity (cash + positions)
    self.get_cash() -> float                    free cash, not the same as equity
      (there is NO self.portfolio/.equity attribute — use these methods)
    self.get_history(count, symbol, field="close") -> np.ndarray   chronological history
      (this IS already a plain numpy array — do NOT call `.values` on it,
      that's a pandas Series/DataFrame method and raises AttributeError here;
      just index/slice it directly, e.g. `closes[-20:]`, `closes.mean()`)
    self.buy(symbol=.., quantity=..) / self.sell(symbol=.., quantity=..)
    self.close_position(symbol=..)
    self.order_target_percent(symbol=.., target_percent=..)   fraction of equity, must be >= 0
    self.order_target_weights(target_weights={{sym: pct, ...}}, liquidate_unmentioned=False, rebalance_tolerance=0.01)
      target_percent/weights must all be >= 0 -- short selling is disabled by
      default (matches real A-share retail constraints); a negative weight
      raises "target weight ... must be >= 0". For a long-short-flavored
      idea, implement it long-only instead (e.g. skip the short leg, or go
      to cash / underweight the worst names rather than shorting them).
  and the hooks you can override (define whichever ones the strategy needs):
    def on_bar(self, bar):            fires every bar; bar.symbol/.open/.high/.low/.close/.volume
      and bar.timestamp_iso (str, ISO 8601, e.g. "2022-01-02T16:00:00Z" — the
      easy way to get the bar's date/time, e.g. bar.timestamp_iso[:10] for
      "YYYY-MM-DD", or int(bar.timestamp_iso[5:7]) for the month). There is
      NO bar.date/.datetime/.time attribute — do not guess one; bar.timestamp
      exists too but is a raw int of nanoseconds since epoch, not directly
      comparable to a month/day — prefer timestamp_iso unless you specifically
      need the raw int.
    def on_daily_rebalance(self, trading_date, timestamp):   fires once per day, before that day's bars
    def __init__(self):               only if you need state; MUST call super().__init__() and take NO
                                       extra constructor arguments (akquant instantiates the class itself)
  `numpy as np` and `pandas as pd` are already in scope, as are the
  globals `SYMBOLS` (list of every ticker in this task), `START`, `END`
  (the task's date range as strings) -- `self.symbols`/`self.start`/
  `self.end` are ALSO set to these automatically before your `__init__`
  runs, so either style works. This is a real, unsandboxed Python
  execution environment — write whatever computation the request needs
  (including model fitting with numpy; scikit-learn/scipy are NOT
  installed).
- strategy.params.expressions: OPTIONAL, only add this for cross-sectional
  multi-factor strategies. A list of factor definitions, each formatted
  EXACTLY as "Name = Expr" using ONLY these functions: {func_catalog}
  over Open/High/Low/Close/Volume, plus +,-,*,/ and numeric constants —
  e.g. "Mom20 = Delta(Close, 20) / Delay(Close, 20)". When present, the
  caller precomputes real per-date scores from these expressions (no
  lookahead) and injects them as a `FACTOR_SCORES` global — a
  `{{"YYYY-MM-DD": {{symbol: score}}}}` dict your `on_daily_rebalance` can
  rank on, e.g. `FACTOR_SCORES.get(str(trading_date)[:10], {{}})` (its
  lookup also tolerates passing `trading_date` directly, unconverted, but
  prefer the string form to be explicit). Do NOT write a bare label like
  "momentum_20" here — it must be a real, evaluable expression, and it is
  only meaningful together with a `source` that actually reads
  FACTOR_SCORES.
- Write strategy.params.source as a YAML literal block scalar (the `|`
  syntax below) so indentation and newlines survive — never inline/escaped.

There is no library of prebuilt rule templates (no ma_cross/rsi/momentum/
bollinger/buy_hold) and no restrictive per-symbol callback contract:
write the logic directly as an akquant Strategy class, whether that's a
single-symbol rule, a statistical/ML model, or a cross-sectional
multi-factor rotation (score every symbol, hold the top-K, rebalance
periodically — see the second example below).

Example 1 — single-symbol rule (note the `|` block scalar for
params.source):
name: "ma-cross-demo"
intent: "5/20 day moving-average crossover on synthetic DEMO"
kind: "strategy"
data:
  source: "synthetic"
  universe: "custom"
  symbols: ["DEMO"]
  start: "2025-01-01"
  end: "2025-12-31"
strategy:
  name: "ma_cross"
  params:
    source: |
      class Strategy(BaseStrategy):
          fast, slow = 5, 20

          def on_bar(self, bar):
              closes = self.get_history(count=self.slow + 1, symbol=bar.symbol, field="close")
              if len(closes) < self.slow + 1:
                  return
              fast_now = closes[-self.fast:].mean()
              slow_now = closes[-self.slow:].mean()
              fast_prev = closes[-self.fast - 1:-1].mean()
              slow_prev = closes[-self.slow - 1:-1].mean()
              position = self.get_position(bar.symbol)
              if fast_prev <= slow_prev and fast_now > slow_now and position <= 0:
                  self.order_target_percent(target_percent=0.95, symbol=bar.symbol)
              elif fast_prev >= slow_prev and fast_now < slow_now and position > 0:
                  self.close_position(bar.symbol)
execution:
  initial_cash: 1000000
report:
  language: "en"

Example 2 — cross-sectional multi-factor rotation (top-K by combined
factor score, two-phase rebalance so exits settle before entries submit —
avoids cash-rejected buy orders):
strategy:
  name: "factor_rotation"
  params:
    expressions:
      - "Mom20 = Delta(Close, 20) / Delay(Close, 20)"
    source: |
      class Strategy(BaseStrategy):
          top_k = 5
          rebalance_days = 5

          def __init__(self):
              super().__init__()
              self.day_count = -1
              self.pending_target = None
              self.held = set()

          def on_daily_rebalance(self, trading_date, timestamp):
              if self.pending_target is not None:
                  self.order_target_weights(
                      target_weights=self.pending_target,
                      liquidate_unmentioned=False,
                      rebalance_tolerance=0.01,
                  )
                  self.held = set(self.pending_target)
                  self.pending_target = None
              self.day_count += 1
              if self.day_count % self.rebalance_days:
                  return
              day_scores = FACTOR_SCORES.get(str(trading_date)[:10]) or {{}}
              if len(day_scores) < self.top_k:
                  return
              ranked = sorted(day_scores, key=day_scores.get, reverse=True)
              weight = 0.95 / self.top_k
              target = {{s: weight for s in ranked[: self.top_k]}}
              for symbol in self.held - set(target):
                  if float(self.get_position(symbol)) > 0:
                      self.close_position(symbol)
              self.pending_target = target

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

    # TaskSpec.validate() doesn't reach into strategy.params (it's an
    # opaque dict as far as the DSL is concerned), so a missing/malformed
    # Strategy class or an invalid factor expression would only fail at
    # backtest time. Catch it here instead, so query_structured's retry
    # loop can make the model fix it before the user ever sees the task.
    try:
        validate_strategy_params(spec.strategy.params)
    except StrategySourceError as exc:
        raise ValueError(str(exc)) from exc

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
