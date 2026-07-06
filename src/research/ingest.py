"""Research input ingestion: idea text, PDF papers, arXiv links, forum URLs.

    source ──► extract text ──► detect research ideas ──► candidate TaskSpecs
               (pdftotext /       (bilingual keyword        (factor mode and/or
                arXiv API /        rules, LLM-swappable)     strategy mode)
                HTML strip)

Deterministic and offline-safe by design: URLs and PDFs fail gracefully
with a clear message, and every produced idea lists the evidence (matched
keywords) so the user can judge the mapping before running anything.
`extract_ideas` is the LLM seam — swap it for a model call later without
touching callers.
"""

from __future__ import annotations

import html
import html.parser
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) VibeQuant/0.1"}
_MAX_TEXT = 40_000  # chars kept for idea scanning
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


class IngestError(RuntimeError):
    pass


@dataclass
class Idea:
    key: str  # stable id, e.g. "momentum"
    kind: str  # "factor" | "strategy"
    title_en: str
    title_zh: str
    evidence: List[str] = field(default_factory=list)  # matched keywords
    score: int = 0  # match count, for ranking
    factor_expressions: List[str] = field(default_factory=list)
    strategy_name: Optional[str] = None
    strategy_params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


@dataclass
class ResearchBrief:
    source_type: str  # idea | pdf | arxiv | url
    source: str
    title: str = ""
    excerpt: str = ""
    ideas: List[Idea] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    text: str = ""  # full analyzed text; NOT serialized (LLM extractor input)

    def to_dict(self) -> Dict:
        return {
            "source_type": self.source_type,
            "source": self.source,
            "title": self.title,
            "excerpt": self.excerpt,
            "ideas": [i.to_dict() for i in self.ideas],
            "notes": self.notes,
        }


# --------------------------------------------------------------- idea rules
# Each rule: bilingual keyword patterns -> factor expressions (akquant
# expression engine syntax) and/or a strategy template. Price/volume proxies
# only — fundamentals aren't available on the ETF/synthetic data paths.
_RULES = [
    dict(
        key="momentum",
        kind="factor",
        title_en="Price momentum",
        title_zh="价格动量",
        pattern=re.compile(r"momentum|动量|趋势跟踪|trend[\s-]?follow|12-1|相对强弱", re.I),
        factor_expressions=[
            "Mom20 = Delta(Close, 20) / Delay(Close, 20)",
            "Mom60 = Delta(Close, 60) / Delay(Close, 60)",
        ],
        strategy_name="momentum",
    ),
    dict(
        key="reversal",
        kind="factor",
        title_en="Short-term reversal / mean reversion",
        title_zh="短期反转 / 均值回归",
        pattern=re.compile(
            r"reversal|revert|reversion|反转|均值回归|超卖|超买|oversold|overbought|contrarian",
            re.I,
        ),
        factor_expressions=["Rev5 = -Delta(Close, 5) / Delay(Close, 5)"],
        strategy_name="rsi_reversion",
    ),
    dict(
        key="low_volatility",
        kind="factor",
        title_en="Low-volatility anomaly",
        title_zh="低波动异象",
        pattern=re.compile(
            r"low[\s-]?vol|volatility|低波动|波动率|idiosyncratic|beta anomaly", re.I
        ),
        factor_expressions=[
            "LowVol20 = -(Ts_Std(Close, 20) / Ts_Mean(Close, 20))"
        ],
    ),
    dict(
        key="volume",
        kind="factor",
        title_en="Volume / turnover signals",
        title_zh="成交量 / 换手信号",
        pattern=re.compile(r"volume|turnover|成交量|换手|量价|放量|liquidity|流动性", re.I),
        factor_expressions=[
            "PVCorr10 = Rank(Ts_Corr(Close, Volume, 10))",
            "VolSurge = Volume / Ts_Mean(Volume, 20)",
        ],
    ),
    dict(
        key="ma_cross",
        kind="strategy",
        title_en="Moving-average crossover",
        title_zh="均线交叉",
        pattern=re.compile(r"moving\s+average|ma\s+cross|均线|金叉|死叉|sma|ema", re.I),
        strategy_name="ma_cross",
    ),
    dict(
        key="bollinger",
        kind="strategy",
        title_en="Bollinger band reversion",
        title_zh="布林带回归",
        pattern=re.compile(r"bollinger|布林|band|通道突破", re.I),
        strategy_name="bollinger",
    ),
    dict(
        key="value_note",
        kind="factor",
        title_en="Value / fundamentals (needs fundamental data)",
        title_zh="价值 / 基本面（需基本面数据）",
        pattern=re.compile(r"\bvalue\b|估值|市盈率|市净率|\bP/?E\b|\bP/?B\b|book[\s-]?to[\s-]?market|ROE", re.I),
        factor_expressions=[],
    ),
]


def extract_ideas(text: str) -> List[Idea]:
    """Scan text for research ideas. Rule-based; the LLM seam."""
    sample = text[:_MAX_TEXT]
    ideas: List[Idea] = []
    for rule in _RULES:
        hits = rule["pattern"].findall(sample)
        if not hits:
            continue
        unique = sorted({h if isinstance(h, str) else h[0] for h in hits})[:8]
        ideas.append(
            Idea(
                key=rule["key"],
                kind=rule["kind"],
                title_en=rule["title_en"],
                title_zh=rule["title_zh"],
                evidence=[u for u in unique if u],
                score=len(hits),
                factor_expressions=list(rule.get("factor_expressions") or []),
                strategy_name=rule.get("strategy_name"),
            )
        )
    ideas.sort(key=lambda i: i.score, reverse=True)
    return ideas


# ------------------------------------------------------------ text sources
class _HTMLText(html.parser.HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.chunks: List[str] = []
        self._skip_depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and not self.title:
            self.title = data.strip()
        if not self._skip_depth:
            text = data.strip()
            if text:
                self.chunks.append(text)


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _pdf_text(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", "-l", "30", str(path), "-"],
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise IngestError(
            "pdftotext not found — install poppler-utils to ingest PDFs"
        ) from exc
    if proc.returncode != 0:
        raise IngestError(f"pdftotext failed: {proc.stderr.decode()[:200]}")
    return proc.stdout.decode("utf-8", errors="replace")


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read(5_000_000)


def _arxiv_brief(arxiv_id: str, source: str) -> ResearchBrief:
    raw = _fetch(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    entry = root.find("a:entry", ns)
    if entry is None:
        raise IngestError(f"arXiv id {arxiv_id} not found")
    title = _clean((entry.findtext("a:title", "", ns) or "").replace("\n", " "))
    abstract = _clean(entry.findtext("a:summary", "", ns) or "")
    brief = ResearchBrief(
        source_type="arxiv",
        source=source,
        title=title,
        excerpt=abstract[:800],
        ideas=extract_ideas(title + "\n" + abstract),
        text=title + "\n" + abstract,
    )
    brief.notes.append(f"arXiv:{arxiv_id} — analyzed title + abstract")
    return brief


def ingest_source(
    source: str, pdf_bytes: Optional[bytes] = None, filename: str = ""
) -> ResearchBrief:
    """Turn any supported input into a ResearchBrief.

    - pdf_bytes given            -> uploaded PDF
    - local path ending in .pdf  -> PDF file
    - arxiv.org/abs|pdf/<id>     -> arXiv API (title + abstract)
    - http(s) URL                -> fetched page, tags stripped
    - anything else              -> treated as a plain research idea
    """
    source = (source or "").strip()

    if pdf_bytes is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            text = _pdf_text(Path(tmp.name))
        brief = ResearchBrief(
            source_type="pdf",
            source=filename or "uploaded.pdf",
            title=filename or "uploaded.pdf",
            excerpt=_clean(text)[:800],
            ideas=extract_ideas(text),
            text=text,
        )
        brief.notes.append("analyzed first 30 pages of the uploaded PDF")
        return brief

    arxiv = _ARXIV_ID.search(source)
    if arxiv:
        return _arxiv_brief(arxiv.group(1), source)

    if re.match(r"https?://", source, re.I):
        raw = _fetch(source)
        parser = _HTMLText()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = _clean("\n".join(parser.chunks))
        if len(text) < 80:
            raise IngestError("page fetched but no readable text extracted")
        brief = ResearchBrief(
            source_type="url",
            source=source,
            title=parser.title or source,
            excerpt=text[:800],
            ideas=extract_ideas(text),
            text=text,
        )
        brief.notes.append("analyzed visible page text")
        return brief

    path = Path(source)
    if source.lower().endswith(".pdf") and path.exists():
        text = _pdf_text(path)
        brief = ResearchBrief(
            source_type="pdf",
            source=str(path),
            title=path.name,
            excerpt=_clean(text)[:800],
            ideas=extract_ideas(text),
            text=text,
        )
        brief.notes.append("analyzed first 30 pages of the PDF")
        return brief

    if not source:
        raise IngestError("empty input")
    return ResearchBrief(
        source_type="idea",
        source=source,
        title=source[:80],
        excerpt=source[:800],
        ideas=extract_ideas(source),
        text=source,
    )
