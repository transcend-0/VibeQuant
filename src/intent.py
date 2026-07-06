"""Intent parsing: natural language (EN/ZH) -> TaskSpec.

Rule-based on purpose: the minimal loop must work offline and
deterministically. The parser is honest about uncertainty — everything
it guessed or defaulted is recorded in `clarifications`, and the caller
(CLI / Web UI) shows the resulting YAML for confirmation before running
(Plan-and-Act). An LLM can later replace `parse_prompt` behind the same
signature.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .dsl import DataSpec, ExecutionSpec, ReportSpec, StrategySpec, TaskSpec


@dataclass
class ParseResult:
    spec: TaskSpec
    clarifications: List[str] = field(default_factory=list)
    recognized: bool = True  # a concrete strategy pattern was detected


_CJK = re.compile(r"[一-鿿]")

_STRATEGY_RULES: List[Tuple[str, re.Pattern]] = [
    ("ma_cross", re.compile(
        r"双均线|均线|金叉|死叉|ma[\s_-]?cross|moving\s+average|sma|golden\s+cross",
        re.I)),
    ("bollinger", re.compile(r"布林|bollinger|\bboll\b", re.I)),
    ("rsi_reversion", re.compile(r"\brsi\b|超卖|超买|oversold|overbought", re.I)),
    ("factor_rotation", re.compile(r"轮动|rotation|多因子", re.I)),
    ("momentum", re.compile(r"动量|momentum|趋势跟踪|trend[\s-]?follow", re.I)),
    ("buy_hold", re.compile(r"买入持有|持有不动|buy\s*(and|&|-)?\s*hold", re.I)),
]

_SYMBOL_RE = re.compile(r"\b((?:sh|sz|bj)?\d{6}|[A-Z]{2,5})\b")
_SYMBOL_STOPWORDS = {
    "RSI", "MACD", "SMA", "EMA", "MA", "KDJ", "BOLL", "ETF", "IPO",
    "YAML", "CSV", "API", "HTML", "PDF", "OK", "AI", "LLM", "CN", "US",
}

_DATE_RE = re.compile(r"(\d{4})[-/年](\d{1,2})[-/月]?(\d{1,2})?")
_LAST_YEARS_RE = re.compile(r"(?:过去|最近|近|last|past)\s*(\d+)\s*(?:年|years?)", re.I)
_CASH_WAN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*万")
_CASH_EN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(k|m)\b", re.I)
_CASH_PLAIN_RE = re.compile(
    r"(?:cash|资金|本金|初始资金)\D{0,6}(\d[\d,]{3,})", re.I)


def parse_prompt(prompt: str) -> ParseResult:
    prompt = prompt.strip()
    clarifications: List[str] = []
    lang = "zh" if _CJK.search(prompt) else "en"

    # the curated 24-ETF pool, by any of its common names
    pool_requested = bool(re.search(
        r"(内置|精选)\s*24\s*支?\s*ETF|24\s*支?\s*ETF\s*池|ETF\s*池|24\s*etf", prompt, re.I
    ))

    strategy_name = _detect_strategy(prompt)
    recognized = strategy_name is not None
    if strategy_name is None:
        strategy_name = "ma_cross"
        clarifications.append(
            "未识别出策略类型，默认使用双均线 (ma_cross)。"
            if lang == "zh"
            else "No strategy recognized; defaulting to ma_cross."
        )
    params = _extract_params(strategy_name, prompt)

    symbols = _extract_symbols(prompt)
    if pool_requested:
        from .data_sources.etf_pool import POOL_SYMBOLS

        symbols = list(POOL_SYMBOLS)
        clarifications.append(
            "已选用精选24ETF池作为标的池。" if lang == "zh"
            else "Using the curated 24-ETF pool as the universe."
        )
    if not symbols:
        symbols = ["DEMO"]
        clarifications.append(
            "未识别出标的，使用合成数据标的 DEMO。"
            if lang == "zh"
            else "No symbols recognized; using synthetic symbol DEMO."
        )

    start, end = _extract_dates(prompt)
    if start is None and end is None:
        start, end = "2022-01-01", "2024-12-31"
        clarifications.append(
            "未识别出回测区间，默认 2022-01-01 ~ 2024-12-31。"
            if lang == "zh"
            else "No date range recognized; defaulting to 2022-01-01 ~ 2024-12-31."
        )

    source = "synthetic"
    universe = "custom"
    etf_pattern = r"(?:sh|sz)?(?:5\d{5}|1[5-6]\d{4})(?:\.(?:SH|SZ))?"
    if pool_requested:
        source = "etf"
        universe = "pool24"
    elif re.search(r"akshare|真实数据|实际数据|real\s+data", prompt, re.I):
        source = "akshare"
    elif symbols != ["DEMO"] and all(
        re.fullmatch(etf_pattern, s, re.I) for s in symbols
    ):
        source = "etf"  # A-share ETF codes -> real daily bars (free, cached)
        clarifications.append(
            "识别到 A 股 ETF 代码，将使用真实日线数据（eastmoney/tencent，本地缓存）。"
            if lang == "zh"
            else "A-share ETF codes detected; using real daily bars "
            "(eastmoney/tencent with local cache)."
        )
    elif symbols != ["DEMO"] and all(re.fullmatch(r"(?:sh|sz|bj)?\d{6}", s) for s in symbols):
        source = "stock"  # A-share stock codes -> real daily bars
        clarifications.append(
            "识别到 A 股个股代码，将使用真实日线数据（eastmoney/tencent，本地缓存）。"
            if lang == "zh"
            else "A-share stock codes detected; using real daily bars "
            "(eastmoney/tencent with local cache)."
        )

    cash = _extract_cash(prompt) or 1_000_000.0

    name_slug = re.sub(r"[^a-zA-Z0-9一-鿿]+", "-", prompt)[:32].strip("-")
    spec = TaskSpec(
        name=name_slug or f"task-{_dt.date.today().isoformat()}",
        intent=prompt,
        data=DataSpec(
            source=source, universe=universe, symbols=symbols,
            start=start, end=end,
        ),
        strategy=StrategySpec(name=strategy_name, params=params),
        execution=ExecutionSpec(initial_cash=cash),
        report=ReportSpec(language=lang),
    )
    spec.validate()
    return ParseResult(
        spec=spec, clarifications=clarifications, recognized=recognized
    )


# ------------------------------------------------------------- helpers
def _detect_strategy(prompt: str) -> Optional[str]:
    for name, pattern in _STRATEGY_RULES:
        if pattern.search(prompt):
            return name
    return None


def _extract_params(strategy: str, prompt: str) -> Dict:
    if strategy == "ma_cross":
        pairs = re.findall(r"(\d+)\s*(?:日|天|day|d\b)?[^\d]{0,8}?(\d+)\s*(?:日|天|day|d\b)", prompt, re.I)
        nums = re.findall(r"\b(\d{1,3})\b", prompt)
        if pairs:
            fast, slow = int(pairs[0][0]), int(pairs[0][1])
        elif len(nums) >= 2:
            fast, slow = int(nums[0]), int(nums[1])
        else:
            return {}
        if 0 < fast < slow <= 250:
            return {"fast": fast, "slow": slow}
        return {}
    if strategy == "rsi_reversion":
        params: Dict = {}
        period = re.search(r"rsi\s*\(?(\d{1,3})\)?|(\d{1,3})\s*日\s*rsi", prompt, re.I)
        if period:
            value = int(period.group(1) or period.group(2))
            if 2 <= value <= 100:
                params["period"] = value
        thresholds = re.findall(r"\b(\d{1,2})\s*[/,，]\s*(\d{2})\b", prompt)
        if thresholds:
            low, high = float(thresholds[0][0]), float(thresholds[0][1])
            if low < high <= 100:
                params["oversold"], params["overbought"] = low, high
        return params
    if strategy == "momentum":
        lookback = re.search(r"(\d{1,3})\s*(?:日|天|day)", prompt, re.I)
        if lookback and 2 <= int(lookback.group(1)) <= 250:
            return {"lookback": int(lookback.group(1))}
        return {}
    if strategy == "bollinger":
        period = re.search(r"(\d{1,3})\s*(?:日|天|day)", prompt, re.I)
        if period and 5 <= int(period.group(1)) <= 250:
            return {"period": int(period.group(1))}
        return {}
    return {}


def _extract_symbols(prompt: str) -> List[str]:
    symbols: List[str] = []
    for match in _SYMBOL_RE.findall(prompt):
        token = match.strip()
        if token.upper() in _SYMBOL_STOPWORDS:
            continue
        if re.fullmatch(r"\d{4}", token):  # bare year fragments
            continue
        if token not in symbols:
            symbols.append(token)
    return symbols[:10]


def _extract_dates(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    matches = _DATE_RE.findall(prompt)
    dates: List[str] = []
    for year, month, day in matches:
        y, m = int(year), int(month)
        if not 1990 <= y <= 2100 or not 1 <= m <= 12:
            continue
        d = int(day) if day else 1
        try:
            dates.append(_dt.date(y, m, d).isoformat())
        except ValueError:
            continue
    if len(dates) >= 2:
        return min(dates), max(dates)
    if len(dates) == 1:
        return dates[0], None

    rel = _LAST_YEARS_RE.search(prompt)
    if rel:
        years = int(rel.group(1))
        end = _dt.date.today()
        start = end.replace(year=end.year - min(years, 30))
        return start.isoformat(), end.isoformat()
    return None, None


def _extract_cash(prompt: str) -> Optional[float]:
    wan = _CASH_WAN_RE.search(prompt)
    if wan:
        return float(wan.group(1)) * 10_000
    en = _CASH_EN_RE.search(prompt)
    if en:
        value = float(en.group(1))
        return value * (1_000 if en.group(2).lower() == "k" else 1_000_000)
    plain = _CASH_PLAIN_RE.search(prompt)
    if plain:
        return float(plain.group(1).replace(",", ""))
    return None
