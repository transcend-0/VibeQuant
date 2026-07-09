"""Deterministic, rule-driven universe construction.

Builds a Universe from a small whitelisted filter pipeline (never free
LLM-authored code — a wrong universe silently corrupts every downstream
backtest without raising an error, unlike strategy source, which either
crashes or produces an obviously-wrong result). Output is written in the
exact snapshot-CSV shape `constituents.py` already reads for hs300
(columns updateDate, code, code_name) so a rule-built pool is
indistinguishable from a curated one to every downstream consumer
(factor_compute's membership_mask, the strategy-side membership
injection) — no other module needs to know a pool was built by a rule.

Known, disclosed limitations (surfaced via `describe_rule`, not hidden):
  - the ETF candidate set is today's directory (`fetch_all_etf_list`),
    not a true historical directory, so a delisted/merged ETF cannot
    re-enter a historical cross-section — survivorship bias.
  - "turnover" is volume * close, a proxy for real yuan turnover (the
    free daily-bar sources here don't expose the traded-amount field).
  - leveraged/inverse classification is a name-keyword heuristic, not an
    authoritative regulatory category.
  - listing date is proxied by each symbol's earliest available bar.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import raw_data_dir
from .constituents import constituents_dir
from .market import MarketDataError, fetch_all_etf_list, fetch_daily

TURNOVER_PROXY_NOTE = "amount proxy = volume * close (no real turnover-yuan field in the free daily-bar sources)"

_LEVERAGE_KEYWORDS = ("杠杆", "两倍", "2倍", "三倍", "3倍", "日内交易型", "槓桿")
_INVERSE_KEYWORDS = ("反向", "沽空", "看跌", "做空")

# type -> (required params -> python type, implemented, one-line doc for the LLM prompt)
FILTER_CATALOG: Dict[str, Dict[str, Any]] = {
    "turnover_percentile": {
        "params": {"top_pct": float},
        "implemented": True,
        "doc": "keep the smallest set of symbols whose cumulative daily turnover "
        "covers >= top_pct of the candidate pool's total that day (0<top_pct<=1)",
    },
    "exclude_leveraged": {
        "params": {},
        "implemented": True,
        "doc": "drop leveraged/geared ETFs (name-keyword heuristic, not authoritative)",
    },
    "exclude_inverse": {
        "params": {},
        "implemented": True,
        "doc": "drop inverse/short ETFs (name-keyword heuristic, not authoritative)",
    },
    "exclude_zero_turnover": {
        "params": {},
        "implemented": True,
        "doc": "drop symbols with zero volume that day (suspended or untraded)",
    },
    "min_listing_days": {
        "params": {"days": int},
        "implemented": True,
        "doc": "require >= `days` trading days since the symbol's first available bar",
    },
    "min_avg_turnover": {
        "params": {"window": int, "min_amount": float},
        "implemented": True,
        "doc": "require trailing `window`-day mean turnover-proxy >= min_amount",
    },
    "max_suspend_days_in_window": {
        "params": {"window": int, "max_days": int},
        "implemented": True,
        "doc": "drop if zero-volume days in the trailing `window` days exceed max_days",
    },
    "exclude_st": {
        "params": {},
        "implemented": False,
        "doc": "NOT YET SUPPORTED — needs an ST-status history data source",
    },
    "fundamental_screen": {
        "params": {"field": str, "op": str, "threshold": float},
        "implemented": False,
        "doc": "NOT YET SUPPORTED — needs a point-in-time financial-statement data source "
        "(field examples: roe, operating_cash_flow, debt_asset_ratio)",
    },
}


def filter_catalog_prompt() -> str:
    """Whitelist description for the LLM system prompt (mirrors _FUNCTION_CATALOG)."""
    lines = []
    for name, spec in FILTER_CATALOG.items():
        params = ", ".join(f"{k}: {t.__name__}" for k, t in spec["params"].items())
        flag = "" if spec["implemented"] else " [NOT IMPLEMENTED]"
        lines.append(f"{name}({params}){flag} -- {spec['doc']}")
    return "\n".join(lines)


class UniverseRuleError(ValueError):
    """Raised when a rule is malformed or references an unsupported filter."""


def _normalize_filter(f: Any) -> Dict[str, Any]:
    """Accept the common shapes LLMs emit for a filter entry and return the
    canonical {"type": <name>, ...params} form.

    Canonical:       {"type": "turnover_percentile", "top_pct": 0.8}
    Name-as-key:     {"turnover_percentile": {"top_pct": 0.8}}
    Bare name:       "exclude_leveraged"  or  {"exclude_leveraged": null}
    Nested params:   {"type": "min_listing_days", "params": {"days": 60}}

    Only the shape is normalized here — unknown filter types and missing/
    mistyped params still fail validation in from_dict, so this widens what
    we accept syntactically without widening what we accept semantically.
    """
    if isinstance(f, str) and f in FILTER_CATALOG:
        return {"type": f}
    if isinstance(f, dict):
        if "type" in f:
            out = dict(f)
            params = out.pop("params", None)
            if isinstance(params, dict):
                for k, v in params.items():
                    out.setdefault(k, v)
            return out
        if len(f) == 1:
            (name, params), = f.items()
            if name in FILTER_CATALOG:
                out = {"type": name}
                if isinstance(params, dict):
                    out.update(params)
                return out
    raise UniverseRuleError(f"malformed filter entry: {f!r}")


@dataclass
class UniverseRule:
    asset_type: str  # only "etf" is implemented today
    base_pool: str = "all_etf"
    rebalance: Dict[str, Any] = field(default_factory=lambda: {"freq": "daily"})
    filters: List[Dict[str, Any]] = field(default_factory=list)
    start: str = ""
    end: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UniverseRule":
        if not isinstance(raw, dict):
            raise UniverseRuleError("universe_rule must be an object")
        asset_type = raw.get("asset_type")
        if asset_type != "etf":
            raise UniverseRuleError(
                f"asset_type {asset_type!r} not supported yet — only 'etf' rules are "
                "implemented (stock rules need fundamentals/ST data not yet integrated)"
            )
        base_pool = raw.get("base_pool", "all_etf")
        if base_pool != "all_etf":
            raise UniverseRuleError(f"base_pool {base_pool!r} not supported yet")
        rebalance = raw.get("rebalance") or {"freq": "daily"}
        freq = rebalance.get("freq")
        if freq not in ("daily", "annual"):
            raise UniverseRuleError(f"rebalance.freq {freq!r} must be 'daily' or 'annual'")
        if freq == "annual":
            month, day = rebalance.get("month"), rebalance.get("day", 1)
            if not isinstance(month, int) or not 1 <= month <= 12:
                raise UniverseRuleError("rebalance.month must be an int 1-12 for freq='annual'")
            if not isinstance(day, int) or not 1 <= day <= 31:
                raise UniverseRuleError("rebalance.day must be an int 1-31 for freq='annual'")
        filters = raw.get("filters") or []
        if not isinstance(filters, list) or not filters:
            raise UniverseRuleError("filters must be a non-empty list")
        filters = [_normalize_filter(f) for f in filters]
        for f in filters:
            spec = FILTER_CATALOG.get(f["type"])
            if spec is None:
                raise UniverseRuleError(
                    f"unknown filter type {f['type']!r}; allowed: {sorted(FILTER_CATALOG)}"
                )
            if not spec["implemented"]:
                raise UniverseRuleError(
                    f"filter {f['type']!r} is not yet supported: {spec['doc']}"
                )
            for pname, ptype in spec["params"].items():
                if pname not in f:
                    raise UniverseRuleError(f"filter {f['type']!r} missing param {pname!r}")
                try:
                    f[pname] = ptype(f[pname])
                except (TypeError, ValueError) as exc:
                    raise UniverseRuleError(
                        f"filter {f['type']!r} param {pname!r} must be {ptype.__name__}"
                    ) from exc
        start = raw.get("start")
        if not start:
            raise UniverseRuleError("start date is required")
        return cls(
            asset_type=asset_type, base_pool=base_pool, rebalance=rebalance,
            filters=filters, start=str(start), end=raw.get("end"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def content_hash(self) -> str:
        blob = json.dumps(
            {"asset_type": self.asset_type, "base_pool": self.base_pool,
             "rebalance": self.rebalance, "filters": self.filters,
             "start": self.start},  # `end` deliberately excluded: a rolling
            sort_keys=True,          # daily rule should reuse the same pool_id
        )                            # as its coverage window grows over time
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]

    def pool_id(self) -> str:
        return f"rule-{self.content_hash()}"


# --------------------------------------------------------------- describe
def describe_rule(
    rule: UniverseRule, pool_id: str, member_rows: Optional[int],
    language: str = "zh", end: Optional[str] = None,
) -> str:
    """Human-readable summary appended to intent-parsing clarifications.

    `member_rows=None` means the pool has not been built yet (plan-time
    description: parsing only proposes; the build_universe step runs after
    the user confirms) — the text then says so instead of citing counts.

    `end` is the actual coverage end passed to `build_pool` (usually
    spec.data.end) -- `rule.end` itself is almost always None (the LLM
    doesn't set it; the caller's own task date range decides it), so
    falling back to `rule.end` for display would misleadingly show
    "today" even when the build only covers up to a fixed past date.
    """
    parts_zh = []
    for f in rule.filters:
        spec = FILTER_CATALOG[f["type"]]
        extra = ", ".join(f"{k}={f[k]}" for k in spec["params"])
        parts_zh.append(f"{f['type']}({extra})" if extra else f["type"])
    filters_txt = " · ".join(parts_zh)
    freq = rule.rebalance.get("freq")
    freq_txt_zh = "每日" if freq == "daily" else f"每年{rule.rebalance.get('month')}月{rule.rebalance.get('day')}日后首个交易日"
    freq_txt_en = "daily" if freq == "daily" else f"annual (first trading day on/after {rule.rebalance.get('month')}/{rule.rebalance.get('day')})"
    end_display = end or rule.end or "至今"
    if language == "zh":
        head_zh = (
            f"已根据规则自动构建 Universe（{pool_id}）" if member_rows is not None
            else f"已识别 Universe 构建规则（{pool_id}），将在你确认计划并执行后自动构建"
        )
        rows_zh = (
            f"，当前共 {member_rows} 条成员记录" if member_rows is not None else ""
        )
        return (
            f"{head_zh}：{rule.asset_type} · "
            f"{freq_txt_zh}调仓 · {filters_txt}，数据区间 {rule.start} 至 "
            f"{end_display}{rows_zh}。"
            f"注意：候选池基于当前ETF目录反向构建，已退市/合并的ETF不包含在内，"
            f"存在幸存者偏差；成交额为 volume×close 的代理值，非真实成交金额。"
        )
    filters_txt_en = " · ".join(
        f"{f['type']}({', '.join(f'{k}={f[k]}' for k in FILTER_CATALOG[f['type']]['params'])})"
        for f in rule.filters
    )
    head_en = (
        f"Auto-built Universe ({pool_id}) from your rule" if member_rows is not None
        else f"Recognized a Universe-building rule ({pool_id}); it will be "
        "built automatically once you confirm the plan and run"
    )
    rows_en = (
        f", {member_rows} membership rows so far" if member_rows is not None else ""
    )
    return (
        f"{head_en}: {rule.asset_type} · "
        f"{freq_txt_en} rebalance · {filters_txt_en}, range {rule.start}..{end or rule.end or 'today'}"
        f"{rows_en}. Caveat: the candidate set is today's ETF "
        f"directory (delisted/merged ETFs can't re-enter historically — survivorship bias); "
        f"turnover is a volume*close proxy, not the real traded amount."
    )


# ---------------------------------------------------------------- calendar
def _trading_days(start: str, end: str) -> List[pd.Timestamp]:
    """Real A-share trading-day list, derived from the CSI300 index's own
    bar dates (no dedicated calendar API among the free sources here).

    Returned as genuine `pd.Timestamp` objects (via DatetimeIndex, not a
    bare numpy array's `.unique()`) so they hash-match the `pd.Timestamp`
    group keys a later `full.groupby("date")` produces — a numpy.datetime64
    and a pd.Timestamp for the same instant do NOT hash equal, so mixing
    them silently drops every dict lookup instead of raising.
    """
    df = fetch_daily("sh000300", "index", start, end, cache_dir=raw_data_dir("index"))
    return list(pd.DatetimeIndex(sorted(pd.to_datetime(df["date"]).unique())))


def resolve_rebalance_dates(rule: UniverseRule, end: str) -> List[pd.Timestamp]:
    days = _trading_days(rule.start, end)
    freq = rule.rebalance.get("freq", "daily")
    if freq == "daily":
        return days
    month, day = rule.rebalance["month"], rule.rebalance.get("day", 1)
    out = []
    for year in range(pd.Timestamp(rule.start).year, pd.Timestamp(end).year + 1):
        anchor = pd.Timestamp(year=year, month=month, day=day)
        candidates = [d for d in days if d >= anchor]
        if candidates:
            out.append(min(candidates))
    return out


# -------------------------------------------------------------- candidates
_MAX_CANDIDATES = 150  # liquidity cap; see describe_rule's caveat and the
# module docstring's cost note -- each candidate costs ~1 request/sec
# (rate-limited) times however many ~640-bar pages its date range needs,
# so this is the dominant lever on build time (500 candidates measured at
# ~2000s / 33min for an 8-year daily rule; 150 cuts that to ~10min)


def _candidate_symbols(rule: UniverseRule) -> Dict[str, str]:
    """{code: name} for the rule's candidate pool. `all_etf` = today's ETF
    directory, capped to the most-traded `_MAX_CANDIDATES` for build time —
    an ETF outside this cap today is very unlikely to ever clear a
    turnover-percentile cutoff historically, but this is a disclosed,
    practical shortcut, not an exact scan of the full ~1500-ETF directory."""
    entries = fetch_all_etf_list(cache_dir=raw_data_dir("etf"), top=_MAX_CANDIDATES)
    return {e["code"]: e["name"] for e in entries}


def _is_leveraged(name: str) -> bool:
    return any(k in name for k in _LEVERAGE_KEYWORDS)


def _is_inverse(name: str) -> bool:
    return any(k in name for k in _INVERSE_KEYWORDS)


# ------------------------------------------------------------- symbol panel
def _max_lookback_days(rule: UniverseRule) -> int:
    """How many extra trading days of history to fetch before rule.start.

    Covers every filter that needs to "look back" from a rebalance date:
    the two rolling-window filters, AND min_listing_days -- fetching from
    exactly rule.start would make bars_seen-so-far undercount every
    already-old symbol as "newly listed" for the first `days` of the rule's
    own date range. Fetching `days` of extra lookback fixes that WITHOUT
    needing a separate full-history-since-2000 listing-date lookup: we only
    ever need to know whether a symbol clears the threshold, not its exact
    age, and a window this wide already proves "clears it" whenever true.
    """
    windows = [
        f.get("window", 0) for f in rule.filters
        if f["type"] in ("min_avg_turnover", "max_suspend_days_in_window")
    ] + [
        f.get("days", 0) for f in rule.filters if f["type"] == "min_listing_days"
    ]
    return max(windows, default=0)


def _build_symbol_panel(symbol: str, name: str, rule: UniverseRule, end: str) -> pd.DataFrame:
    lookback = _max_lookback_days(rule)
    fetch_start = (pd.Timestamp(rule.start) - pd.Timedelta(days=int(lookback * 1.6) + 10)).strftime("%Y-%m-%d")
    try:
        df = fetch_daily(symbol, "etf", fetch_start, end, cache_dir=raw_data_dir("etf"))
    except MarketDataError:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["name"] = name
    df["amount_proxy"] = df["volume"].astype(float) * df["close"].astype(float)
    df["is_zero_turnover"] = df["volume"].astype(float) <= 0
    if any(f["type"] == "min_avg_turnover" for f in rule.filters):
        for f in rule.filters:
            if f["type"] == "min_avg_turnover":
                w = int(f["window"])
                df[f"avg_turnover_{w}"] = df["amount_proxy"].rolling(w, min_periods=w).mean()
    if any(f["type"] == "max_suspend_days_in_window" for f in rule.filters):
        for f in rule.filters:
            if f["type"] == "max_suspend_days_in_window":
                w = int(f["window"])
                df[f"suspend_count_{w}"] = df["is_zero_turnover"].rolling(w, min_periods=w).sum()
    # listing age in trading days = bars seen so far within THIS fetch --
    # correct (not an undercount) precisely because _max_lookback_days
    # already widened fetch_start by any min_listing_days threshold, so a
    # symbol older than that threshold has already accumulated >= that
    # many bars before rule.start even begins; we never need its true age
    # beyond "does it clear the threshold", so there's no need to know
    # exactly how much older it might be, nor to fetch its full history.
    df["listing_trading_days"] = range(1, len(df) + 1)
    return df


def _apply_filters(day_panel: pd.DataFrame, rule: UniverseRule) -> pd.DataFrame:
    out = day_panel
    for f in rule.filters:
        ftype = f["type"]
        if ftype == "exclude_zero_turnover":
            out = out[~out["is_zero_turnover"]]
        elif ftype == "exclude_leveraged":
            out = out[~out["name"].map(_is_leveraged)]
        elif ftype == "exclude_inverse":
            out = out[~out["name"].map(_is_inverse)]
        elif ftype == "min_listing_days":
            out = out[out["listing_trading_days"] >= int(f["days"])]
        elif ftype == "min_avg_turnover":
            w = int(f["window"])
            out = out[out[f"avg_turnover_{w}"] >= float(f["min_amount"])]
        elif ftype == "max_suspend_days_in_window":
            w = int(f["window"])
            out = out[out[f"suspend_count_{w}"] <= int(f["max_days"])]
        elif ftype == "turnover_percentile":
            top_pct = float(f["top_pct"])
            total = out["amount_proxy"].sum()
            if total <= 0:
                out = out.iloc[0:0]
            else:
                sorted_out = out.sort_values("amount_proxy", ascending=False)
                cum_before = sorted_out["amount_proxy"].cumsum() - sorted_out["amount_proxy"]
                keep = cum_before < top_pct * total
                out = sorted_out[keep]
        if out.empty:
            break
    return out


# ------------------------------------------------------------------- build
def is_fresh(pool_id: str, end: str) -> bool:
    """A build is fresh if its recorded coverage reaches `end`."""
    rule_file = constituents_dir() / pool_id / "rule.json"
    if not rule_file.exists():
        return False
    meta = json.loads(rule_file.read_text(encoding="utf-8"))
    covered_end = meta.get("covered_end")
    return bool(covered_end) and pd.Timestamp(covered_end) >= pd.Timestamp(end)


def build_pool(rule: UniverseRule, end: str) -> str:
    """Compute membership and write hs300-shaped snapshot CSV(s). Returns pool_id."""
    pool_id = rule.pool_id()
    if is_fresh(pool_id, end):
        return pool_id

    candidates = _candidate_symbols(rule)
    rebalance_dates = resolve_rebalance_dates(rule, end)
    if not rebalance_dates:
        raise UniverseRuleError(f"no trading days in range {rule.start}..{end}")

    panels = []
    for symbol, name in candidates.items():
        panel = _build_symbol_panel(symbol, name, rule, end)
        if not panel.empty:
            panels.append(panel)
    if not panels:
        raise UniverseRuleError("no candidate ETF data could be fetched")
    full = pd.concat(panels, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])

    rebalance_set = set(rebalance_dates)
    by_date = {d: g for d, g in full[full["date"].isin(rebalance_set)].groupby("date")}

    rows = []
    for d in rebalance_dates:
        day_panel = by_date.get(d)
        if day_panel is None or day_panel.empty:
            continue
        survivors = _apply_filters(day_panel, rule)
        for code in survivors["symbol"]:
            rows.append((d.strftime("%Y-%m-%d"), code, candidates.get(code, "")))

    pool_dir = constituents_dir() / pool_id
    pool_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = pool_dir / f"{pool_id}_list_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv"
    pd.DataFrame(rows, columns=["updateDate", "code", "code_name"]).to_csv(
        snapshot_path, index=False
    )
    # only one snapshot file kept per pool: a rebuild fully supersedes the
    # previous one (rows already carry every rebalance date, unlike hs300's
    # append-only curated snapshots), so stale files would just double-count
    for stale in pool_dir.glob(f"{pool_id}_list_*.csv"):
        if stale != snapshot_path:
            stale.unlink()

    (pool_dir / "rule.json").write_text(
        json.dumps(
            {"rule": rule.to_dict(), "pool_id": pool_id, "covered_end": end,
             "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
             "member_rows": len(rows)},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return pool_id


def member_count(pool_id: str) -> int:
    rule_file = constituents_dir() / pool_id / "rule.json"
    if not rule_file.exists():
        return 0
    return json.loads(rule_file.read_text(encoding="utf-8")).get("member_rows", 0)


def load_rule(pool_id: str) -> Optional[UniverseRule]:
    rule_file = constituents_dir() / pool_id / "rule.json"
    if not rule_file.exists():
        return None
    meta = json.loads(rule_file.read_text(encoding="utf-8"))
    return UniverseRule(**meta["rule"])
