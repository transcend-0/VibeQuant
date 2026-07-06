"""Built-in A-share ETF pool with categories.

The default research universe. Categories double as the grouping for
industry neutralization (`factor.neutralization: industry`).
"""

from __future__ import annotations

from typing import Dict, List

# category key -> (label_en, label_zh)
CATEGORIES = {
    "abroad": ("Abroad", "海外"),
    "commodity": ("Commodities", "商品"),
    "bond": ("Government bonds", "国债"),
    "index": ("Broad index", "宽基指数"),
    "industry": ("Industry", "行业"),
}

# symbol -> (category, name_zh)
DEFAULT_ETF_POOL: Dict[str, tuple] = {
    # Abroad 海外
    "513100": ("abroad", "纳指ETF"),
    "513520": ("abroad", "日经ETF"),
    "513030": ("abroad", "德国ETF"),
    # Commodities 商品
    "518880": ("commodity", "黄金ETF"),
    "159980": ("commodity", "有色ETF"),
    "159985": ("commodity", "豆粕ETF"),
    "501018": ("commodity", "南方原油"),
    # Government bonds 国债
    "511010": ("bond", "5年国债ETF"),
    # Broad index 宽基指数
    "510300": ("index", "沪深300ETF"),
    "512100": ("index", "中证1000ETF"),
    "159915": ("index", "创业板100"),
    "513130": ("index", "恒生科技ETF"),
    # Industry 行业
    "512800": ("industry", "银行ETF"),
    "512710": ("industry", "证券ETF"),
    "512290": ("industry", "生物医药ETF"),
    "159851": ("industry", "创新药ETF"),
    "516670": ("industry", "畜牧养殖ETF"),
    "515030": ("industry", "半导体ETF"),  # per user list: 新能源车? keep name from list
    "159997": ("industry", "电子ETF"),
    "159806": ("industry", "新能源车ETF"),
    "516160": ("industry", "新能源ETF"),
    "159928": ("industry", "消费ETF"),
    "159607": ("industry", "中概互联ETF"),
    "515980": ("industry", "人工智能ETF"),
}

POOL_SYMBOLS: List[str] = list(DEFAULT_ETF_POOL)


def pool_categories() -> Dict[str, List[str]]:
    """category key -> symbols."""
    out: Dict[str, List[str]] = {}
    for sym, (cat, _name) in DEFAULT_ETF_POOL.items():
        out.setdefault(cat, []).append(sym)
    return out


def industry_groups(symbols: List[str]) -> Dict[str, str]:
    """symbol (any spelling) -> group key, for industry neutralization.

    Symbols outside the built-in pool fall into group "other".
    """
    from .market import normalize_symbol

    known = {}
    for raw in symbols:
        try:
            code, _ = normalize_symbol(raw, "etf")
        except Exception:
            code = str(raw)
        entry = DEFAULT_ETF_POOL.get(code)
        known[str(raw)] = entry[0] if entry else "other"
    return known
