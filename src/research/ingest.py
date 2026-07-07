"""Research input ingestion: idea text, PDF papers, arXiv links, forum URLs.

    source ──► extract text ──► LLM idea extraction ──► candidate TaskSpecs
               (pdftotext /       (src.research.llm_ideas,   (factor mode and/or
                arXiv API /        the caller's job)          strategy mode)
                HTML strip)

`ingest_source` only turns a source into text + metadata (a `ResearchBrief`
with `ideas=[]`); URLs and PDFs fail gracefully with a clear message.
Turning that text into candidate ideas is the LLM's job exclusively (see
`src.research.llm_ideas.llm_extract_ideas`) — there is no keyword-rule
fallback, so callers must treat extraction failure as an error, not as a
reason to degrade.
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
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) VibeQuant/0.1"}
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)")
_BARE_URL = re.compile(r"https?://\S+")


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
    # any instruction typed alongside a URL/link (e.g. "只用600519和600036
    # 构建策略" next to a pasted paper link) — NOT serialized, used by the
    # caller to extract a universe/symbols override (see llm_ideas' sibling
    # module research.tasks.extract_universe_hint).
    user_instruction: str = ""

    def to_dict(self) -> Dict:
        return {
            "source_type": self.source_type,
            "source": self.source,
            "title": self.title,
            "excerpt": self.excerpt,
            "ideas": [i.to_dict() for i in self.ideas],
            "notes": self.notes,
        }


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


def _split_url(source: str) -> Tuple[Optional[str], str]:
    """Find the first URL (markdown-linked or bare) anywhere in `source`.

    Returns (url_or_None, instruction) where `instruction` is `source` with
    the link syntax collapsed to its label text (markdown) or removed
    (bare URL) — so a sentence like "根据这篇[论文](https://...)的思路，
    在精选24ETF上构建策略" keeps its accompanying instruction instead of
    losing it once the URL is pulled out.
    """
    md = _MD_LINK.search(source)
    if md:
        instruction = (source[: md.start()] + md.group(1) + source[md.end():]).strip()
        return md.group(2), instruction
    bare = _BARE_URL.search(source)
    if bare:
        instruction = (source[: bare.start()] + source[bare.end():]).strip()
        return bare.group(0), instruction
    return None, source


def _arxiv_brief(arxiv_id: str, source: str, instruction: str = "") -> ResearchBrief:
    raw = _fetch(f"http://export.arxiv.org/api/query?id_list={arxiv_id}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    entry = root.find("a:entry", ns)
    if entry is None:
        raise IngestError(f"arXiv id {arxiv_id} not found")
    title = _clean((entry.findtext("a:title", "", ns) or "").replace("\n", " "))
    abstract = _clean(entry.findtext("a:summary", "", ns) or "")
    paper_text = title + "\n" + abstract
    text = f"User instruction: {instruction}\n\n{paper_text}" if instruction else paper_text
    brief = ResearchBrief(
        source_type="arxiv",
        source=source,
        title=title,
        excerpt=abstract[:800],
        text=text,
        user_instruction=instruction,
    )
    brief.notes.append(f"arXiv:{arxiv_id} — analyzed title + abstract")
    if instruction:
        brief.notes.append(f"用户附加说明: {instruction}")
    return brief


def ingest_source(
    source: str, pdf_bytes: Optional[bytes] = None, filename: str = "",
    instruction: str = "",
) -> ResearchBrief:
    """Turn any supported input into a ResearchBrief.

    - pdf_bytes given            -> uploaded PDF (`instruction`: any question
                                     typed alongside the upload, e.g. "根据
                                     PDF，生成因子" — a separate input from
                                     the file picker, so it must be passed
                                     explicitly here rather than embedded in
                                     `source`)
    - local path ending in .pdf  -> PDF file
    - arxiv.org/abs|pdf/<id>     -> arXiv API (title + abstract)
    - http(s) URL                -> fetched page, tags stripped
    - anything else              -> treated as a plain research idea
    """
    source = (source or "").strip()
    instruction = (instruction or "").strip()

    if pdf_bytes is not None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            text = _pdf_text(Path(tmp.name))
        full_text = f"User instruction: {instruction}\n\n{text}" if instruction else text
        brief = ResearchBrief(
            source_type="pdf",
            source=filename or "uploaded.pdf",
            title=filename or "uploaded.pdf",
            excerpt=_clean(text)[:800],
            text=full_text,
            user_instruction=instruction,
        )
        brief.notes.append("analyzed first 30 pages of the uploaded PDF")
        if instruction:
            brief.notes.append(f"用户附加说明: {instruction}")
        return brief

    url, instruction = _split_url(source)

    if url:
        arxiv = _ARXIV_ID.search(url)
        if arxiv:
            return _arxiv_brief(arxiv.group(1), source, instruction)

        raw = _fetch(url)
        parser = _HTMLText()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = _clean("\n".join(parser.chunks))
        if len(text) < 80:
            host = urlparse(url).hostname or ""
            if host.endswith("weixin.qq.com"):
                raise IngestError(
                    "微信公众号文章无法通过程序直接抓取——微信会向非浏览器请求返回"
                    "反爬验证页（\"环境异常\"）而非文章正文。请把文章正文复制粘贴到"
                    "输入框，作为普通研究想法提交。"
                )
            raise IngestError("page fetched but no readable text extracted")
        full_text = f"User instruction: {instruction}\n\n{text}" if instruction else text
        brief = ResearchBrief(
            source_type="url",
            source=source,
            title=parser.title or url,
            excerpt=text[:800],
            text=full_text,
            user_instruction=instruction,
        )
        brief.notes.append("analyzed visible page text")
        if instruction:
            brief.notes.append(f"用户附加说明: {instruction}")
        return brief

    path = Path(source)
    if source.lower().endswith(".pdf") and path.exists():
        text = _pdf_text(path)
        brief = ResearchBrief(
            source_type="pdf",
            source=str(path),
            title=path.name,
            excerpt=_clean(text)[:800],
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
        text=source,
    )
