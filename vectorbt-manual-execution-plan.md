# vectorbt + 手动执行量化项目规划

## 1. 项目目标

构建一个面向日线/小时线的个人量化研究与交易建议系统：自动获取行情数据，运行策略和模型，使用 vectorbt 回测，维护模拟/真实组合状态，生成可人工确认的交易建议；实际下单由人在证券公司交易系统中完成，成交结果再回录系统。

第一版不接券商 API，不负责自动下单。系统的核心输出是：

- 当前组合状态和净值曲线
- 目标仓位
- 相对当前持仓的交易建议
- 建议数量、参考价格、风险限制和信号原因
- 实际成交后的组合更新

## 2. 范围和默认假设

- 频率：日线优先，小时线作为第二阶段能力。
- 市场：通过数据源适配器隔离，第一阶段只接一个市场和一种资产类型。
- 执行：人工下单，CSV/SQLite 录入成交。
- 策略：先实现一个规则基线，再接入机器学习/DRL。
- 回测：必须包含手续费、滑点、交易时点和仓位约束。
- 部署：个人电脑定时运行，后续再迁移到服务器。

## 3. 总体架构

```text
数据源
  -> 原始数据归档（Parquet）
  -> 清洗/复权/时间对齐
  -> 特征生成
  -> 策略/模型
  -> 目标仓位
  -> 风控检查
  -> 交易建议报告
  -> 人工下单
  -> 成交回录
  -> 组合状态与绩效
```

建议把研究和交易状态分开：

- **研究层**：可重复运行，输入历史数据和参数，输出信号与回测结果。
- **组合层**：记录现金、持仓、成本、成交和企业行动。
- **建议层**：将目标仓位转换为具体交易动作。
- **人工执行层**：不假设建议一定成交，成交结果必须单独录入。

## 4. 推荐技术栈

| 功能 | 技术 | 说明 |
|---|---|---|
| Python 环境 | Python 3.11 + uv/Poetry | 固定依赖和可复现环境 |
| 数据处理 | pandas、polars | 先用 pandas，数据量增大再引入 polars |
| 回测 | vectorbt | 信号扫描、成本分析、组合绩效 |
| 存储 | Parquet + DuckDB | 行情和特征；SQLite 保存组合与成交 |
| 配置 | YAML | 标的池、频率、成本和风险参数 |
| 报告 | Jinja2 + HTML，CSV 作为兜底 | 方便人工查看和存档 |
| 测试 | pytest | 数据、信号、仓位和组合状态测试 |
| 调度 | Windows Task Scheduler / cron | 日线或小时任务 |
| 日志 | Python logging | 每次运行生成 run_id 和日志 |

## 5. 目录结构

```text
quant-manual-trader/
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
│  └─ cli.py               # download/run/backtest/report/reconcile
├─ tests/
├─ notebooks/
└─ runs/YYYY-MM-DD/<run_id>/
   ├─ input_manifest.json
   ├─ signals.csv
   ├─ orders.csv
   ├─ portfolio_snapshot.json
   └─ report.html
```

## 6. 核心数据模型

### 行情数据

至少包含：`symbol, timestamp, open, high, low, close, volume`。另外记录：

- 数据源和下载时间
- 时区和频率
- 是否复权
- 最后一个完整 bar 的时间
- 数据校验结果

### 目标仓位

```python
@dataclass
class TargetPosition:
    symbol: str
    timestamp: datetime
    target_weight: float
    reason: str
    model_version: str
```

### 交易建议

```python
@dataclass
class TradeSuggestion:
    symbol: str
    action: Literal["BUY", "SELL", "REDUCE", "HOLD"]
    quantity: int
    target_weight: float
    reference_price: float
    max_price: float | None
    min_price: float | None
    risk_flags: list[str]
    reason: str
```

### 实际成交

```python
@dataclass
class ExecutedTrade:
    trade_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    fee: float
    executed_at: datetime
    broker_reference: str | None
    note: str | None
```

`TargetPosition`、`TradeSuggestion` 和 `ExecutedTrade` 必须分开存储，不能把建议当成成交。

## 7. 模块设计

### 7.1 数据模块

实现统一接口：

```python
class MarketDataProvider(Protocol):
    def fetch(self, symbols, start, end, timeframe) -> DataFrame: ...
```

职责：下载、去重、排序、时区统一、缺失 bar 检查、复权处理、原始数据归档。每次运行生成数据清单，避免同一回测悄悄使用了不同数据。

### 7.2 特征和策略模块

策略输入只能使用当前 bar 及之前的数据。第一版实现一个简单基线，例如：趋势过滤 + 波动率调整 + 最大持仓数。输出目标仓位，不直接生成券商订单。

```python
def generate_targets(features, portfolio, config) -> list[TargetPosition]:
    ...
```

DRL 接入时，将模型封装成同一接口：模型只给出信号或目标仓位，不能绕过风险层。

### 7.3 vectorbt 回测模块

统一封装：

- entries/exits 或目标仓位输入
- 手续费和滑点
- 仓位上限和现金约束
- 交易日/交易时段
- 基准组合
- 训练、验证、测试区间

至少输出：总收益、年化收益、波动率、Sharpe、最大回撤、Calmar、胜率、换手率、交易次数、费用占比和逐笔交易记录。

回测必须做三组敏感性分析：

1. 手续费和滑点扩大后的结果。
2. 信号延迟一个 bar 后的结果。
3. 训练参数在滚动时间窗口上的稳定性。

### 7.4 组合模块

组合模块维护：现金、持仓数量、平均成本、已实现盈亏、未实现盈亏、净值和最后对账时间。每天先导入实际成交，再计算新目标仓位。

建议使用 SQLite 事务写入成交，禁止直接修改历史成交；修正通过反向冲销或更正记录完成。

### 7.5 风控模块

在生成建议前强制检查：

- 单标的和行业最大权重
- 总杠杆和现金缓冲
- 单日最大换手
- 最小成交金额和最小数量
- 流动性/成交量约束
- 单日亏损和组合回撤阈值
- 数据过期、缺失或异常价格
- 模型输出 NaN、极端仓位或异常换手

任何检查失败时，建议状态为 `BLOCKED`，而不是自动放行。

### 7.6 建议和报告模块

报告按“先风险、后动作、再原因”展示：

```text
运行时间 / 数据截止时间 / 模型版本
风险状态：PASS 或 BLOCKED
组合净值、现金、回撤、当日盈亏
需要执行的买入/卖出/减仓
每笔建议的数量、参考价、目标仓位和风险标记
未执行建议和上次建议对比
```

每次报告都保存到 `runs/<date>/<run_id>`，便于回溯当时使用的数据和模型。

## 8. CLI 工作流

```bash
python -m quant download --date 2026-10-08
python -m quant validate-data --date 2026-10-08
python -m quant backtest --config configs/strategy.yaml
python -m quant generate-signals --date 2026-10-08
python -m quant reconcile --file fills.csv
python -m quant report --date 2026-10-08
```

运行失败时不生成“可执行建议”，只生成错误报告并告警。

## 9. 分阶段里程碑

### M0：项目骨架（1–2 天）

验收：环境可安装；配置文件可读取；CLI 有统一入口；pytest 可运行；每次运行有 `run_id`。

### M1：数据层（3–5 天）

验收：下载一个市场的日线数据；原始/处理数据分层；重复运行结果稳定；缺失、重复、时区和异常价格有检查。

### M2：基线策略和 vectorbt 回测（5–7 天）

验收：实现一个无未来函数的基线策略；包含手续费和滑点；输出完整绩效和交易明细；完成成本和延迟敏感性分析。

### M3：组合和建议层（5–7 天）

验收：能导入当前持仓和现金；生成目标仓位及买卖建议；建议数量符合最小交易单位；建议和成交分开保存。

### M4：纸面运行（至少 4 周）

验收：每天/每小时定时运行；连续记录数据、建议、人工执行结果；能处理漏跑、重复运行和手动改仓；每周完成一次策略与实际组合对账。

### M5：DRL 接入（2–4 周）

验收：DRL 模型实现与基线相同的目标仓位接口；固定数据快照可复现；与基线、买入持有策略比较；通过成本、延迟和滚动窗口测试。

### M6：极小资金实盘（持续观察）

验收：只使用可承受损失的资金；单日亏损和仓位上限生效；人工 kill-switch 可用；每日完成成交、持仓和现金对账；异常时停止执行。

## 10. 测试与验收重点

- 数据：时区、缺失 bar、重复 bar、复权和数据截止时间。
- 防未来函数：特征只能看到信号时点以前的数据。
- 回测：手续费、滑点、信号延迟和成交价假设。
- 组合：买入、卖出、部分成交、费用、分红/拆股或合约换月。
- 建议：整数数量、最小交易单位、现金不足、风险阻断。
- 恢复：重复运行幂等、断点重跑、数据库事务和报告留档。

## 11. 推荐的首个策略开发顺序

1. 先做买入持有和简单均线策略作为基线。
2. 加入手续费、滑点、延迟和流动性约束。
3. 用滚动时间窗口验证参数稳定性。
4. 接入一个 DRL 模型，只替换信号模块。
5. 比较 DRL 与规则基线在相同成本和风险约束下的表现。
6. 纸面运行至少四周，再决定是否使用极小资金。

## 12. 主要风险和停止条件

出现以下任一情况时停止执行建议：数据源异常、数据时间戳落后、组合无法对账、程序重复生成建议、实际成交与记录不一致、模型输出异常、超过单日亏损或最大回撤阈值、交易系统无法确认订单状态。

这个项目的第一版成功标准不是回测收益最大，而是能稳定回答三个问题：

1. 今天使用了什么数据和模型？
2. 系统建议了哪些动作，为什么？
3. 实际成交后，组合状态是否能准确恢复和对账？
