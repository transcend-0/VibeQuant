<p align="center">
  <b>English</b> | <a href="README_zh.md">中文</a>
</p>

<h1 align="center">VibeQuant: Your Personal Quant Research Workbench</h1>

<p align="center">
  <b>From a Research Idea to a Validated Strategy — in One Conversation</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Engine-akquant-007ec6?style=flat" alt="akquant">
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=flat" alt="FastAPI">
  <img src="https://img.shields.io/badge/UI-Bilingual%20EN%2F%E4%B8%AD%E6%96%87-orange?style=flat" alt="Bilingual">
</p>

---

## 💡 What Is VibeQuant?

VibeQuant is an intent-driven quant research workbench built on the [akquant](https://github.com/akfamily/akquant) engine (which stays completely untouched — all integration goes through two thin adapters). Feed it a research idea, an arXiv link, a forum URL, or a PDF paper; review the generated plan; run a factor study or a strategy backtest on real market data; read a bilingual report with statistical validation; then deploy the strategy as a daily post-close signal email.

```
idea / paper / URL / YAML → TaskSpec (DSL) → planner → tools → akquant adapters → report / library / signals
```

## ✨ Key Features

| | |
|---|---|
| 🔍 **Intent-Driven Research** | One Analyze button routes your input: concrete instructions parse instantly; ideas, papers (PDF/arXiv), and web pages go through LLM-backed idea extraction (keyword-rule fallback, every guess disclosed) |
| 🧪 **Two Research Modes** | **Factor Research**: WorldQuant-Brain-style workbench — expressions, neutralization (incl. industry), decay, truncation, ADV20 liquidity caps, IC/ICIR + quantile layers. **Strategy Research**: template backtests incl. `factor_rotation` (top-K by combined factor score) with akquant's native HTML report + benchmark panel |
| 📊 **Cross-Market Data** | A-share ETF/stocks, HK, US, crypto through free keyless sources with fallback chains ordered by IP-ban risk (tencent→eastmoney, yahoo→…, OKX→Binance), throttled, paged, and cached locally |
| 🛡️ **Honest Validation** | Every run gets an overfit-risk verdict: cross-sectional permutation test with **trial-count deflation** (thresholds tighten with every factor you've tried on that universe) + sub-period consistency — explicitly *not* sold as out-of-sample |
| 🤖 **Two Optimization Paths** | Ask a follow-up question ("try industry neutralization") → one validated revision to review; or let the **agent self-iterate**: run → reflect → revise → run, with data and task kind pinned so it can't cheat |
| 📮 **Signal Deployment** | Daily post-close scheduler (Asia/Shanghai) replays the frozen strategy on fresh bars and emails next-day signals (BUY/SELL/HOLD per symbol). Signals only — no broker, no orders |

## ⚡ Quick Start

```bash
cd VibeQuant
pip install -e .

vq ui        # Web UI at http://127.0.0.1:8321  (Strategy · Factor · Deploy · ⚙)
```

```bash
# CLI equivalents
vq ask "5/20 MA cross on 510300, 2021-01-01 to 2024-12-31"
vq run tasks/factor_etf_demo.yaml
vq deploy add tasks/ma_cross_demo_etf.yaml --email you@example.com --at 16:30
vq runs
```

## 🧭 Markets & Universes

Both research modes share one **Market → Universe** hierarchy backed by the same data layer:

| Market | Universes |
|---|---|
| A-share ETF | curated **24-ETF pool** (abroad/commodities/bonds/index/industry — doubles as the industry-neutralization grouping), **All ETF** (top-200 by turnover or the complete ~1,580-fund directory), custom |
| A-share stocks | **CSI 300 constituents as a point-in-time pool**, blue-chip sample, custom |
| Hong Kong | Hang Seng index, big-tech sample, custom |
| US | S&P 500 / NASDAQ 100 indices, Magnificent 7, custom |
| Crypto | top coins, custom |

## 📡 Data Sources & Fallback

One client, six asset kinds, free keyless endpoints only. Chains are ordered by IP-ban risk:

- **A-share (etf/stock/index)** → `tencent` · `eastmoney`
- **US** → `yahoo` · `tencent` · `eastmoney`
- **HK** → `eastmoney` · `tencent`
- **Crypto** → `okx` · `binance`

All bars cache under `data/raw/<kind>/`; computed factor panels live in the factor library (`data/factors/` + `registry.jsonl`); every run's artifacts (task.yaml, report.md/html, result.json) persist under `workspace/runs/<id>/`.

## 🔬 Validation Philosophy

In-sample statistics cannot detect a researcher who has already seen the whole sample. VibeQuant's validation therefore targets what *can* be caught, and says so:

1. **Permutation test with trial-count deflation** — is the IC distinguishable from shuffling, at a threshold of 0.05/T where T is how many factors you've already tried on this universe (counted from the experiment log)?
2. **Sub-period consistency** — does the signal exist in every regime, or only in one lucky year? (Computed by windowing the single run; nothing is re-run.)
3. The agent's auto-optimize objective and history are persisted, so every iteration is auditable.

## 🤖 LLM: Intelligence and Execution Are Separated by Design

The **research intelligence** — reading papers, extracting testable ideas, proposing revisions, agent self-iteration — is LLM-driven, and configuring one is the intended way to use VibeQuant (everything the model produces passes schema validation before it can run).

The **execution core** — DSL, planner, backtest/factor engines, statistical validation, reports — is deliberately deterministic and never touches the LLM: results must be reproducible and auditable, and a validator that depends on a sampling model cannot serve as a referee. This split follows the design blueprint's own architecture (deterministic capability layer, LLM at the orchestration layer).

Without an LLM configured the workbench degrades gracefully to keyword-rule fallbacks instead of stalling — useful when the API is down or offline, but a degraded mode, not the full experience. Configure in **⚙ Settings** or `config/llm.yaml`:

```yaml
model: "gpt-5"
api_key: "sk-..."
base_url: "https://api.openai.com/v1"   # or any OpenAI-compatible endpoint
```

## Disclaimer

VibeQuant is for research and education. Nothing it produces is investment advice.
