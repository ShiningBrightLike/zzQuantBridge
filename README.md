# zzQuantBridge

面向个人研究与交易的 **vectorbt + 人工执行** 量化系统。项目自动获取和整理行情数据，运行策略与回测，维护模拟或真实组合状态，并生成可供人工确认的交易建议；实际下单由人在证券公司交易系统中完成，成交结果再回录系统。

> 当前仓库已完成 M0 项目骨架、M1 数据层和组合/建议层的第一批实现：配置可读取，CLI 统一入口可创建运行目录和输入清单，支持本地 OHLCV 校验、Parquet 快照归档、AkShare 可选适配器、可插拔的均线目标仓位策略、SQLite 成交账本和人工交易建议。vectorbt 回测报告会在后续里程碑中接入。

## 开始使用

使用 Python 3.11 或更高版本安装项目及测试依赖：

```bash
python -m pip install -e ".[data,backtest,test]"
```

也可以直接安装当前完整依赖列表：

```bash
python -m pip install -r requirements.txt
```

需要在线获取 A 股数据时安装数据依赖：

```bash
python -m pip install -e ".[data]"
```

每个 CLI 命令都会创建 `runs/<date>/<run_id>/`，并写入 `input_manifest.json` 和当前状态文件。M0 阶段命令先负责建立可追溯的运行边界：

```bash
python -m quant download --date 2026-10-08 --config configs/universe.yaml
python -m quant backtest --config configs/strategy.yaml
python -m quant validate-data --file data/processed/sample.csv --symbols 600000
python -m quant reconcile --file fills.csv --db db/portfolio.sqlite --initial-cash 100000
```

`download` 默认使用 AkShare；需要离线复现时可以指定本地快照：

```bash
python -m quant download --provider local --file snapshot.csv \
  --config configs/universe.yaml --start 2026-01-01 --end 2026-10-08
```

`validate-data` 会在运行目录写入 `data_quality.json`；缺失、重复或异常 OHLCV 会标记为 `BLOCKED`。`backtest` 等命令的回测业务将在后续阶段填充；在实现完成前，运行状态会标记为 `INITIALIZED`，不会生成可执行交易建议。

`reconcile` 会幂等导入成交 CSV，并在运行目录保存 `portfolio_snapshot.json`。CSV 至少需要 `trade_id,symbol,side,quantity,price,fee,executed_at` 列。

## 项目目标

第一版围绕三个问题建立可追溯的日线量化工作流：

1. 今天使用了什么数据和模型？
2. 系统建议了哪些动作，为什么？
3. 实际成交后，组合状态是否能准确恢复和对账？

核心输出包括：

- 当前组合状态、现金、持仓和净值曲线
- 策略计算出的目标仓位
- 相对当前持仓的买入、卖出或减仓建议
- 建议数量、参考价格、风险限制和信号原因
- 实际成交后的组合更新与绩效记录

## 设计原则

- **人工执行优先**：第一版不接券商 API，不自动下单。
- **建议与成交分离**：目标仓位、交易建议和实际成交分别保存，建议不会被当作成交。
- **研究与组合分离**：研究结果可重复运行，组合状态则以成交记录为准。
- **风险检查前置**：任何风险检查失败时，建议状态为 `BLOCKED`，不生成可执行动作。
- **数据可追溯**：原始数据只追加不覆盖，每次运行保存数据清单、模型版本和输出文件。
- **先规则基线，后机器学习**：先用简单、可解释的策略验证整个链路，再接入 DRL 等模型。

## 总体流程

```text
数据源
  -> 原始数据归档（Parquet）
  -> 清洗、复权、时间对齐
  -> 特征生成
  -> 策略或模型
  -> 目标仓位
  -> 风控检查
  -> 交易建议报告
  -> 人工下单
  -> 成交回录
  -> 组合状态与绩效
```

研究层负责历史数据、特征、信号和回测；组合层负责现金、持仓、成本和成交；建议层将目标仓位转换为具体动作；人工执行层记录真实成交结果。

## 计划技术栈

| 能力 | 技术 | 用途 |
| --- | --- | --- |
| Python 环境 | Python 3.11 + uv/Poetry | 固定依赖、复现运行环境 |
| 数据处理 | pandas，必要时引入 polars | 清洗行情和计算特征 |
| 回测 | vectorbt | 信号扫描、成本分析和组合绩效 |
| 存储 | Parquet + DuckDB；SQLite | 行情/特征归档；组合和成交记录 |
| 配置 | YAML | 标的池、策略、成本和风险参数 |
| 报告 | Jinja2 + HTML，CSV 兜底 | 人工查看、分享和存档 |
| 测试 | pytest | 数据、信号、仓位和组合状态测试 |
| 调度 | Windows Task Scheduler / cron | 日线或小时任务 |

## 目标目录结构

```text
zzQuantBridge/
├─ pyproject.toml
├─ README.md
├─ configs/
│  ├─ universe.yaml
│  ├─ strategy.yaml
│  └─ risk.yaml
├─ data/
│  ├─ raw/                 # 原始行情，只追加不覆盖
│  ├─ processed/           # 清洗后的 OHLCV
│  ├─ features/            # 特征快照
│  └─ metadata/            # 数据源、版本、时间范围
├─ db/
│  └─ portfolio.sqlite
├─ src/quant/
│  ├─ data/                # 数据源适配、校验、清洗
│  ├─ features/            # 特征计算
│  ├─ strategies/          # 规则策略和 DRL 推理
│  ├─ backtest/            # vectorbt 封装和评估
│  ├─ portfolio/           # 持仓、现金、成交、净值
│  ├─ risk/                # 敞口、换手、亏损和流动性限制
│  ├─ signals/             # 目标仓位到交易建议
│  ├─ reports/             # HTML/CSV 报告
│  └─ cli.py               # 命令行入口
├─ tests/
├─ notebooks/
└─ runs/YYYY-MM-DD/<run_id>/
   ├─ input_manifest.json
   ├─ signals.csv
   ├─ orders.csv
   ├─ portfolio_snapshot.json
   └─ report.html
```

## 核心数据对象

系统至少区分以下三类对象：

- `TargetPosition`：策略在某一时点希望达到的标的权重、原因和模型版本。
- `TradeSuggestion`：结合当前组合、价格和风险检查后生成的具体动作、数量和参考价格。
- `ExecutedTrade`：人工实际完成的成交，包括价格、费用、成交时间和券商参考号。

行情数据至少包含 `symbol`、`timestamp`、`open`、`high`、`low`、`close` 和 `volume`，同时记录数据源、下载时间、时区、频率、复权状态和最后一个完整 bar 的时间。

## 回测与风控要求

回测必须明确处理：

- 手续费、滑点、交易时点和仓位约束
- 现金约束、基准组合和训练/验证/测试区间
- 总收益、年化收益、波动率、Sharpe、最大回撤、Calmar、胜率、换手率、交易次数和费用占比
- 手续费/滑点扩大、信号延迟一个 bar、滚动时间窗口三组敏感性分析

生成建议前必须检查单标的和行业权重、总杠杆、现金缓冲、单日换手、最小成交金额、流动性、亏损与回撤阈值、数据新鲜度以及模型输出异常。检查失败时只记录阻断原因，不放行建议。

## 计划中的 CLI 工作流

以下命令是目标接口，待项目骨架完成后启用：

```bash
python -m quant download --date 2026-10-08
python -m quant validate-data --date 2026-10-08
python -m quant backtest --config configs/strategy.yaml
python -m quant generate-signals --date 2026-10-08
python -m quant reconcile --file fills.csv
python -m quant report --date 2026-10-08
```

运行失败时不生成“可执行建议”，只保留错误报告和告警信息。每次运行使用唯一 `run_id`，并将输入清单、信号、建议、组合快照和报告保存到对应的 `runs/` 目录。

## 开发路线

| 阶段 | 目标 | 主要验收标准 |
| --- | --- | --- |
| M0 | 项目骨架 | 环境可安装、配置可读、CLI 统一入口、pytest 可运行 |
| M1 | 数据层 | 原始/处理数据分层，重复运行稳定，完成缺失和异常检查 |
| M2 | 基线策略与回测 | 无未来函数，包含成本和滑点，输出完整绩效与敏感性分析 |
| M3 | 组合与建议层 | 可导入持仓和现金，生成整数数量建议，建议与成交分离 |
| M4 | 纸面运行 | 连续运行至少四周，能处理漏跑、重复运行、改仓和对账 |
| M5 | DRL 接入 | 使用与基线一致的目标仓位接口，并完成公平对比 |
| M6 | 极小资金实盘 | 仓位与亏损上限生效，kill-switch 可用，每日完成对账 |

推荐先实现买入持有和简单均线策略，再逐步加入成本、延迟、流动性约束和滚动验证；纸面运行至少四周后，再评估是否使用极小资金。

## 停止条件

出现数据源异常、数据时间戳落后、组合无法对账、重复生成建议、成交记录不一致、模型输出异常、超过单日亏损或最大回撤阈值，或无法确认订单状态时，停止执行建议并保留错误与审计记录。

## 相关文档

- [vectorbt 手动执行项目规划](vectorbt-manual-execution-plan.md)

## 免责声明

本项目用于个人量化研究、回测和人工交易辅助，不构成投资建议。任何实盘操作都应由使用者独立确认，并在可承受损失的范围内进行。

