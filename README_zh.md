<p align="center">
  <a href="README.md">English</a> | <b>中文</b>
</p>

<h1 align="center">VibeQuant：你的个人量化研究工作台</h1>

<p align="center">
  <b>从一个研究想法到一个通过验证的策略——在一次对话里完成</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/引擎-akquant-007ec6?style=flat" alt="akquant">
  <img src="https://img.shields.io/badge/后端-FastAPI-009688?style=flat" alt="FastAPI">
  <img src="https://img.shields.io/badge/界面-中英双语-orange?style=flat" alt="双语">
</p>

---

## 💡 VibeQuant 是什么？

VibeQuant 是一个意图驱动的量化研究工作台，构建在 [akquant](https://github.com/akfamily/akquant) 引擎之上（akquant 本身零改动——所有集成通过两个薄适配器完成）。输入一个研究想法、一条 arXiv 链接、一个论坛网页或一份 PDF 论文；确认系统生成的执行计划；在真实行情上运行因子研究或策略回测；阅读带统计验证的双语报告；最后把策略部署为每日收盘后的信号邮件。

系统**只做研究与回测**：内置安全闸直接拒绝实盘/模拟盘交易模式，部署产出的是信号——永远不是订单。

```
想法 / 论文 / URL / YAML → 任务 DSL → 计划器 → 工具链 → akquant 适配器 → 报告 / 因子库 / 信号
```

## ✨ 核心特性

| | |
|---|---|
| 🔍 **意图驱动的研究** | 一个"解析"按钮自动分流：明确指令即刻解析成任务；想法、论文（PDF/arXiv）、网页走 LLM 想法提取（关键词规则兜底，所有猜测都会明示） |
| 🧪 **两种研究模式** | **因子研究**：WorldQuant Brain 风格工作台——表达式、中性化（含行业中性）、衰减、截断、ADV20 流动性约束、IC/ICIR 与分层回测。**策略研究**：模板回测，含 `factor_rotation` 多因子轮动（因子综合得分持有 top-K），输出 akquant 原生 HTML 报告 + 基准对比面板 |
| 📊 **跨市场数据** | A股 ETF/个股、港股、美股、加密货币，全部走免费免密钥数据源，回退链按封禁风险排序（tencent→eastmoney、yahoo→…、OKX→Binance），限速、分页、本地缓存 |
| 🛡️ **诚实的验证** | 每次运行给出过拟合风险判定：截面置换检验 + **试验次数折减**（在同一标的池上试过的因子越多，显著性门槛越严）+ 子区间一致性——明确不冒充"样本外" |
| 🤖 **两条优化路径** | 追问优化（"试试行业中性化"）→ 返回一个经校验的修订任务供确认；或让 **Agent 自动迭代**：运行 → 反思 → 修改 → 再运行，数据区间与任务类型被锁定，无法靠换样本"作弊" |
| 📮 **信号部署** | 每日收盘后调度器（Asia/Shanghai）在最新行情上重放定型策略，邮件发送次日信号（每标的 BUY/SELL/HOLD）。仅信号——无券商、无下单 |

## ⚡ 快速开始

```bash
cd VibeQuant
pip install -e .

vq ui        # Web UI：http://127.0.0.1:8321（策略研究 · 因子研究 · 部署 · ⚙）
```

```bash
# CLI 等价用法
vq ask "在 510300 上做 5/20 双均线策略回测，2021-01-01 到 2024-12-31"
vq run tasks/factor_etf_demo.yaml
vq deploy add tasks/ma_cross_demo_etf.yaml --email you@example.com --at 16:30
vq runs
```

## 🧭 市场与标的池

两种研究模式共享同一套 **市场 → Universe** 层级（底层同一数据）：

| 市场 | Universe |
|---|---|
| A股 ETF | **精选24ETF**（海外/商品/国债/宽基/行业分类，同时充当行业中性化分组）、**全部 ETF**（成交额前200 或完整 ~1580 只目录）、自定义 |
| A股 个股 | **沪深300成分股（时点池）**、蓝筹样本、自定义 |
| 港股 | 恒生指数、科技龙头样本、自定义 |
| 美股 | 标普500 / 纳斯达克100 指数、七巨头、自定义 |
| 加密货币 | 主流币、自定义 |

## 📡 数据源与回退链

一个客户端、六类资产、全部免费免密钥。链序按封禁风险排列：

- **A股（etf/stock/index）** → `tencent` · `eastmoney`
- **美股** → `yahoo` · `tencent` · `eastmoney`
- **港股** → `eastmoney` · `tencent`
- **加密货币** → `okx` · `binance`

行情缓存于 `data/raw/<kind>/`；因子值入库 `data/factors/`（因子库 + registry.jsonl）；每次运行的产物（task.yaml、report.md/html、result.json）落盘 `workspace/runs/<id>/`。

## 🔬 验证哲学

样本内统计检验防不住"研究者已经看过全样本"。所以 VibeQuant 的验证只声称它真能防住的东西：

1. **试验次数折减的置换检验**——这个 IC 与随机打乱可区分吗？门槛为 0.05/T，T 是你在该池上已试过的因子数（从实验日志统计）；
2. **子区间一致性**——信号在每个区制都存在，还是只靠某个幸运年份？（单次运行事后切窗，不重跑）；
3. Agent 自动迭代的目标与历史全部落盘，每一轮可审计。

## 🤖 LLM：智能与执行的分层设计

**研究智能层**——读论文提取想法、追问优化、Agent 自动迭代——由 LLM 驱动，配置 LLM 才是本系统的完整用法（模型产出的一切在运行前都经过 schema 校验）。

**执行核心层**——DSL、计划器、回测/因子引擎、统计验证、报告——刻意做成确定性的，永不经过 LLM：回测结果必须可复现、可审计，一个依赖采样模型的验证器没有资格当裁判。这个分层遵循设计蓝图自身的架构（能力服务层确定性，LLM 位于编排层）。

未配置 LLM 时，系统优雅降级为关键词规则兜底而不是瘫痪——适用于 API 故障或离线场景，但那是**降级模式**，不是完整体验。在 **⚙ 设置页** 或 `config/llm.yaml` 中配置：

```yaml
model: "gpt-5"
api_key: "sk-..."
base_url: "https://api.openai.com/v1"   # 任何 OpenAI 兼容接口均可
```


## 免责声明

VibeQuant 仅用于研究与学习，产出内容不构成投资建议。
