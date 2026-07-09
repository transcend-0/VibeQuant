"""Multi-asset daily bars via free no-auth endpoints, Vibe-Trading style.

One client, several asset kinds — the same secid/code addressing scheme
Vibe-Trading's loaders use, with an ordered fallback chain per kind,
per-host throttling, backward paging past per-request caps, and a local
CSV cache under data/raw/<kind>/.

kind      symbols accepted                        examples
-------   -------------------------------------   -----------------------
etf       A-share ETF codes                       510300, sh510300, 159915.SZ
stock     A-share stock codes                     600000, sz000001, 300750
index     A-share index codes                     000300, sh000300, 399006
hk        HK stocks (5 digits) / indices          00700, hk00700, HSI
us        US tickers / indices (dot-prefixed)     AAPL, QQQ, .NDX, .INX, .DJI
crypto    coin pairs (default quote USDT)         BTC, BTC-USDT, ETH/USDT

Fallback chains ordered by IP-ban risk (Vibe-Trading's ordering):
A-share kinds tencent -> eastmoney; US yahoo -> tencent -> eastmoney;
HK eastmoney -> tencent; crypto OKX -> Binance.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

KINDS = ("etf", "stock", "index", "hk", "us", "crypto")


class MarketDataError(RuntimeError):
    pass


class FetchCancelled(RuntimeError):
    """A host UI cancelled the operation mid-download.

    Deliberately NOT a MarketDataError: the per-symbol/per-source fallback
    machinery swallows MarketDataError and moves on to the next candidate,
    which is exactly wrong for a user-initiated abort — this must
    propagate all the way out of the fetch loops.
    """


# optional threading.Event installed by a host UI (webui/server.py): every
# throttled network request checks it, so cancelling aborts a multi-minute
# per-symbol download loop within ~one request instead of at its end
CANCEL_EVENT = None

_UA = "Mozilla/5.0 (X11; Linux x86_64) VibeQuant/0.3"
_MIN_INTERVAL = 1.0
_last_call: Dict[str, float] = {}


def _throttled_json(host_key: str, url: str, headers: Optional[Dict] = None):
    if CANCEL_EVENT is not None and CANCEL_EVENT.is_set():
        raise FetchCancelled("cancelled by user")
    wait = _last_call.get(host_key, 0.0) + _MIN_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.load(resp)
    finally:
        _last_call[host_key] = time.monotonic()
    return payload


# ------------------------------------------------------- normalization
def normalize_symbol(raw: str, kind: str) -> Tuple[str, str]:
    """Return (code, exchange_tag) for an asset kind.

    exchange_tag: SH|SZ for A-share kinds; HK for hk; OQ|N|P|IDX|"" for us.
    """
    token = str(raw).strip().upper()
    if kind in ("etf", "stock", "index"):
        match = re.fullmatch(r"(?:(SH|SZ|BJ)\.?)?(\d{6})(?:\.(SH|SZ|BJ))?", token)
        if not match:
            raise MarketDataError(f"not an A-share {kind} symbol: {raw!r}")
        prefix, code, suffix = match.groups()
        exchange = prefix or suffix
        if exchange is None:
            if kind == "etf":
                exchange = "SH" if code.startswith("5") else "SZ"
            elif kind == "index":
                exchange = "SZ" if code.startswith("39") else "SH"
            else:  # stock
                exchange = "SH" if code[0] in "679" else "SZ"
        return code, exchange
    if kind == "hk":
        match = re.fullmatch(r"(?:HK\.?)?(\d{1,5}|[A-Z]{2,6})", token)
        if not match:
            raise MarketDataError(f"not a HK symbol: {raw!r}")
        code = match.group(1)
        if code.isdigit():
            code = code.zfill(5)
        return code, "HK"
    if kind == "us":
        match = re.fullmatch(r"(\.?[A-Z][A-Z0-9.]{0,9})", token)
        if not match:
            raise MarketDataError(f"not a US symbol: {raw!r}")
        code = match.group(1)
        if code.startswith("."):
            return code[1:], "IDX"
        if "." in code:
            ticker, _, suffix = code.partition(".")
            return ticker, suffix
        return code, ""
    if kind == "crypto":
        pair = token.replace("/", "-")
        if "-" not in pair:
            for quote in ("USDT", "USDC", "BUSD"):
                if pair.endswith(quote) and len(pair) > len(quote):
                    pair = f"{pair[:-len(quote)]}-{quote}"
                    break
            else:
                pair = f"{pair}-USDT"
        base, _, quote = pair.partition("-")
        if not re.fullmatch(r"[A-Z0-9]{2,10}", base) or not re.fullmatch(
            r"[A-Z]{3,6}", quote
        ):
            raise MarketDataError(f"not a crypto pair: {raw!r}")
        return f"{base}-{quote}", "CRYPTO"
    raise MarketDataError(f"unknown asset kind {kind!r}")


def canonical(raw: str, kind: str) -> str:
    code, tag = normalize_symbol(raw, kind)
    if kind in ("etf", "stock"):
        return f"{code}.{tag}"
    if kind == "index":
        return f"{tag}{code}".lower()  # sh000300 style, unambiguous
    if kind == "hk":
        return f"hk{code}"
    if kind == "crypto":
        return code  # BTC-USDT
    return f".{code}" if tag == "IDX" else code  # us


# ------------------------------------------------------------ sources
def _frame(rows: List[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _eastmoney_secid(code: str, tag: str, kind: str) -> Optional[str]:
    if kind in ("etf", "stock", "index"):
        return f"{'1' if tag == 'SH' else '0'}.{code}"
    if kind == "hk":
        return f"116.{code}" if code.isdigit() else None
    if kind == "us":
        if tag == "IDX":
            return f"100.{code}"
        return None  # US equities need the suggest API; tencent handles them
    return None


def _fetch_eastmoney(kind: str):
    def fetch(code: str, tag: str, start: str, end: str) -> pd.DataFrame:
        secid = _eastmoney_secid(code, tag, kind)
        if secid is None:
            return _frame([])
        params = urllib.parse.urlencode(
            dict(
                secid=secid, klt=101, fqt=1,
                beg=start.replace("-", ""), end=end.replace("-", ""),
                lmt=1_000_000,
                fields1="f1,f2,f3,f4,f5,f6",
                fields2="f51,f52,f53,f54,f55,f56,f57",
            )
        )
        payload = _throttled_json(
            "eastmoney",
            f"https://push2his.eastmoney.com/api/qt/stock/kline/get?{params}",
        )
        klines = ((payload or {}).get("data") or {}).get("klines") or []
        rows = []
        for line in klines:
            parts = line.split(",")
            if len(parts) >= 6:
                rows.append((parts[0], float(parts[1]), float(parts[3]),
                             float(parts[4]), float(parts[2]), float(parts[5])))
        return _frame(rows)

    return fetch


_US_SUFFIXES = (".OQ", ".N", ".P", "")  # NASDAQ, NYSE, ARCA, bare


def _tencent_codes(code: str, tag: str, kind: str) -> List[str]:
    if kind in ("etf", "stock", "index"):
        return [f"{tag.lower()}{code}"]
    if kind == "hk":
        return [f"hk{code}"]
    # us
    if tag == "IDX":
        return [f"us.{code}"]
    if tag:
        return [f"us{code}.{tag}"]
    return [f"us{code}{sfx}" for sfx in _US_SUFFIXES]


def _fetch_tencent(kind: str):
    def fetch(code: str, tag: str, start: str, end: str) -> pd.DataFrame:
        for tcode in _tencent_codes(code, tag, kind):
            url = (
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
                f"param={tcode},day,{start},{end},640,qfq"
            )
            payload = _throttled_json(
                "tencent", url, headers={"Referer": "https://web.ifzq.gtimg.cn/"}
            )
            node = ((payload or {}).get("data") or {}).get(tcode) or {}
            bars = node.get("qfqday") or node.get("day") or []
            rows = []
            for bar in bars:
                if len(bar) >= 6:
                    rows.append((bar[0], float(bar[1]), float(bar[3]),
                                 float(bar[4]), float(bar[2]), float(bar[5])))
            if rows:
                return _frame(rows)
        return _frame([])

    return fetch


def _ts_ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


def _fetch_okx(code: str, tag: str, start: str, end: str) -> pd.DataFrame:
    """OKX daily candles, self-paginating (100 bars/request, newest first)."""
    rows: List[tuple] = []
    after = _ts_ms(end) + 86_400_000
    start_ms = _ts_ms(start)
    for _ in range(80):  # 80 * 100 daily bars ≈ 20+ years
        params = urllib.parse.urlencode(
            dict(instId=code, bar="1Dutc", limit=100, after=after)
        )
        payload = _throttled_json(
            "okx", f"https://www.okx.com/api/v5/market/history-candles?{params}"
        )
        candles = (payload or {}).get("data") or []
        if not candles:
            break
        for c in candles:  # [ts, o, h, l, c, vol, ...]
            ts = int(c[0])
            if ts < start_ms:
                continue
            date = pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            rows.append((date, float(c[1]), float(c[2]), float(c[3]),
                         float(c[4]), float(c[5])))
        oldest = int(candles[-1][0])
        if oldest <= start_ms:
            break
        after = oldest
    return _frame(rows)


def _fetch_binance(code: str, tag: str, start: str, end: str) -> pd.DataFrame:
    """Binance daily klines, paginating forward (1000 bars/request)."""
    symbol = code.replace("-", "")
    rows: List[tuple] = []
    cursor = _ts_ms(start)
    end_ms = _ts_ms(end) + 86_400_000
    for _ in range(20):
        params = urllib.parse.urlencode(
            dict(symbol=symbol, interval="1d", limit=1000,
                 startTime=cursor, endTime=end_ms)
        )
        payload = _throttled_json(
            "binance", f"https://api.binance.com/api/v3/klines?{params}"
        )
        if not isinstance(payload, list) or not payload:
            break
        for k in payload:  # [openTime, o, h, l, c, vol, ...]
            date = pd.Timestamp(int(k[0]), unit="ms", tz="UTC").strftime("%Y-%m-%d")
            rows.append((date, float(k[1]), float(k[2]), float(k[3]),
                         float(k[4]), float(k[5])))
        if len(payload) < 1000:
            break
        cursor = int(payload[-1][0]) + 86_400_000
    return _frame(rows)


_YAHOO_INDEX_MAP = {"INX": "^GSPC", "NDX": "^NDX", "DJI": "^DJI", "IXIC": "^IXIC"}


def _fetch_yahoo(code: str, tag: str, start: str, end: str) -> pd.DataFrame:
    """Yahoo chart API (US): keyless, full range in one request."""
    if tag == "IDX":
        symbol = _YAHOO_INDEX_MAP.get(code, f"^{code}")
    else:
        symbol = code  # exchange suffix not needed on yahoo
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp()) + 86_400
    params = urllib.parse.urlencode(
        dict(period1=p1, period2=p2, interval="1d")
    )
    payload = _throttled_json(
        "yahoo",
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}?{params}",
    )
    result = (((payload or {}).get("chart") or {}).get("result") or [None])[0]
    if not result:
        return _frame([])
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    rows = []
    for i, ts in enumerate(timestamps):
        o, h, low, c = (quote.get(k, [None] * len(timestamps))[i]
                        for k in ("open", "high", "low", "close"))
        v = quote.get("volume", [None] * len(timestamps))[i]
        if None in (o, h, low, c):
            continue
        date = pd.Timestamp(ts, unit="s", tz="UTC").strftime("%Y-%m-%d")
        rows.append((date, float(o), float(h), float(low), float(c),
                     float(v or 0)))
    return _frame(rows)


def _paged(fetcher: Callable, code: str, tag: str, start: str, end: str) -> pd.DataFrame:
    """Page backwards past the ~640-bars-per-request caps."""
    frames: List[pd.DataFrame] = []
    cursor_end = end
    for _ in range(40):
        chunk = fetcher(code, tag, start, cursor_end)
        if chunk.empty:
            break
        frames.append(chunk)
        first = chunk["date"].min()
        if first <= pd.Timestamp(start) or len(chunk) < 500:
            break
        cursor_end = (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if not frames:
        return _frame([])
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]


# ---------------------------------------------------------------- api
def fetch_daily(
    raw_symbol: str,
    kind: str,
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily qfq OHLCV for one instrument with fallback chain + cache."""
    if kind not in KINDS:
        raise MarketDataError(f"unknown kind {kind!r}; expected one of {KINDS}")
    code, tag = normalize_symbol(raw_symbol, kind)
    symbol = canonical(raw_symbol, kind)

    cache_file = None
    if cache_dir is not None and use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace(".", "_")
        cache_file = cache_dir / f"{safe}__{start}__{end}.csv"
        if cache_file.exists():
            cached = pd.read_csv(cache_file, parse_dates=["date"])
            if not cached.empty:
                return cached

    # fallback chains ordered by IP-ban risk (Vibe-Trading's ordering):
    # never-banned public sources first, throttled ones last
    if kind == "crypto":
        # both fetchers self-paginate over the full window
        chain = [("okx", _fetch_okx), ("binance", _fetch_binance)]
    elif kind in ("etf", "stock", "index"):
        chain = [
            ("tencent", _fetch_tencent(kind)),
            ("eastmoney", _fetch_eastmoney(kind)),
        ]
    elif kind == "us":
        chain = [
            ("yahoo", _fetch_yahoo),
            ("tencent", _fetch_tencent(kind)),
            ("eastmoney", _fetch_eastmoney(kind)),
        ]
    else:  # hk: eastmoney leads Vibe-Trading's HK chain
        chain = [
            ("eastmoney", _fetch_eastmoney(kind)),
            ("tencent", _fetch_tencent(kind)),
        ]
    errors: List[str] = []
    for name, fetcher in chain:
        try:
            df = _paged(fetcher, code, tag, start, end)
        except FetchCancelled:
            raise  # user abort: never "try the next source"
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        if df.empty:
            errors.append(f"{name}: empty response")
            continue
        if cache_file is not None:
            df.to_csv(cache_file, index=False)
        return df

    raise MarketDataError(
        f"all sources failed for {kind} {symbol} ({start}..{end}): "
        + "; ".join(errors)
    )


# ------------------------------------------------------- ETF directory
def fetch_all_etf_list(
    cache_dir: Optional[Path] = None, top: Optional[int] = None
) -> List[Dict[str, str]]:
    """All exchange-listed ETFs [{code, name, amount}], turnover-descending.

    Eastmoney clist boards MK0021-24 cover exchange funds. Cached per day
    under data/raw/etf/. `top` truncates to the N most-traded ETFs.
    """
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d")
        cache_file = cache_dir / f"_etf_list__{stamp}.json"
        if cache_file.exists():
            entries = json.loads(cache_file.read_text(encoding="utf-8"))
            return entries[:top] if top else entries

    entries: List[Dict[str, str]] = []
    for page in range(1, 8):  # 7 * 500 covers the ~1500-ETF directory
        params = urllib.parse.urlencode(
            dict(pn=page, pz=500, po=1, np=1, fltt=2, invt=2, fid="f6",
                 fs="b:MK0021,b:MK0022,b:MK0023,b:MK0024", fields="f12,f14,f6")
        )
        payload = None
        for host in ("push2", "82.push2", "88.push2"):  # rotating mirrors
            try:
                payload = _throttled_json(
                    "eastmoney",
                    f"https://{host}.eastmoney.com/api/qt/clist/get?{params}",
                )
                break
            except FetchCancelled:
                raise
            except Exception:
                continue
        diff = ((payload or {}).get("data") or {}).get("diff") or []
        if not diff:
            break
        for item in diff:
            code = str(item.get("f12", ""))
            if code.isdigit() and len(code) == 6:
                entries.append(
                    {"code": code, "name": str(item.get("f14", "")),
                     "amount": float(item.get("f6") or 0)}
                )
        total = ((payload or {}).get("data") or {}).get("total") or 0
        if len(entries) >= total:
            break
    if not entries:
        entries = _sina_etf_list()
    if not entries:
        raise MarketDataError(
            "ETF directory unavailable (eastmoney clist + sina both failed)"
        )
    if cache_file is not None:
        cache_file.write_text(
            json.dumps(entries, ensure_ascii=False), encoding="utf-8"
        )
    return entries[:top] if top else entries


def _sina_etf_list() -> List[Dict[str, str]]:
    """Fallback ETF directory via sina's fund node (GBK JSON, 80/page)."""
    entries: List[Dict[str, str]] = []
    base = (
        "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "Market_Center.getHQNodeData?num=80&sort=amount&asc=0&node=etf_hq_fund&page="
    )
    for page in range(1, 26):  # 25 * 80 covers the ~1600-ETF directory
        if CANCEL_EVENT is not None and CANCEL_EVENT.is_set():
            raise FetchCancelled("cancelled by user")
        wait = _last_call.get("sina", 0.0) + _MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            base + str(page),
            headers={"User-Agent": _UA,
                     "Referer": "http://vip.stock.finance.sina.com.cn/"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("gbk", errors="replace"))
        except Exception:
            break
        finally:
            _last_call["sina"] = time.monotonic()
        if not data:
            break
        for item in data:
            code = str(item.get("symbol", ""))[-6:]
            if code.isdigit():
                entries.append(
                    {"code": code, "name": str(item.get("name", "")),
                     "amount": float(item.get("amount") or 0)}
                )
        if len(data) < 80:
            break
    return entries
