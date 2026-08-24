# Signal Portfolio Optimize

[English](README.en.md) | [Skill 工作流](SKILL.md)

将一个冻结的股票截面信号转换为基准相对、仅做多的目标权重，并在统一框架内完成开放式 Barra-style 风险建模、组合约束、滚动回测和诊断。

| 项目 | 当前状态 |
| --- | --- |
| Catalog 状态 | `active` |
| 验证等级 | `runnable` |
| 实现版本 | `1.0.0` |
| Python | CI 使用 3.12 |
| 许可证 | GPL-3.0-only |

这里的 `active` 是项目生命周期状态，`runnable` 是仓库声明并由本地校验支持的 L2 目标；社区正式收录和等级认定仍以 QuantSkills 维护者评审为准。

## 解决什么问题

上游模型通常只回答“哪些股票更值得买”，并不直接回答“每只股票买多少”。本 Skill 将 LightGBM 预测或单因子信号作为 alpha 输入，使用与信号分离的风险模型和约束计算目标权重：

```text
冻结信号 -> 全截面校准 -> 候选股票池 -> 风险模型 -> 两阶段优化
         -> 目标权重 -> 下一交易日滚动回测 -> 风险与约束诊断
```

它不训练预测模型、不选择因子、不下单，也不复现 MSCI Barra 的专有模型。

## 核心能力

- 输入一个 `date | ticker | prediction` 最终信号，支持 LightGBM 和单因子结果。
- 接收资产协方差，或直接使用因子暴露 `X`、因子协方差 `F`、特异风险 `D`。
- 从历史收益和市值估计开放式结构化风险模型。
- 控制单股权重、主动权重、SIZE/其他风格暴露、行业暴露、换手率、跟踪误差和停牌冻结。
- 使用 v1.0.0 两阶段目标：先最大化主动信号效用，再保留最低信号捕获率并降低线性交易成本和权重扰动。
- 支持下一交易日生效的滚动回测、动态风险缓存、断点续跑和参数扫描。
- 输出目标权重、风险摘要、约束诊断、信号诊断和带输入哈希的运行清单。

## 风险因子

| 代码 | 中文含义 | 当前计算 |
| --- | --- | --- |
| MARKET | 市场因子 | 是 |
| SIZE | 市值因子 | 是，示例配置默认约束 |
| BETA | 市场敏感度 | 是，可选约束 |
| MOMENTUM | 动量 | 是，可选约束 |
| RESVOL | 残差波动率 | 是，可选约束 |
| NLSIZE | 非线性市值 | 是，可选约束 |
| INDUSTRY:* | 行业哑变量 | 仅在提供严格时点行业区间数据时计算 |

“已经计算”不等于“自动约束”。需要在配置中明确设置目标暴露和容忍区间。行业功能已实现，但默认关闭；当前行业历史不能证明严格的 PIT 区间语义时，禁止用今天的行业分类回填历史。

## 安装

```bash
git clone git@github.com:quantskills/skill-signal-portfolio-optimize.git
cd skill-signal-portfolio-optimize
python -m pip install -r requirements.txt
```

## 最小输入

单日资产协方差模式至少需要：

- 信号：`date | ticker | prediction`
- 基准：`date | ticker | benchmark_weight`
- 协方差：行列均为股票代码的方阵
- 配置：参考 [examples/config.yaml](examples/config.yaml)

候选股票池 `date | ticker` 为可选输入；不提供时，全部信号股票都作为候选。启用换手、行业、风格或停牌约束时，还需要当前权重、行业、暴露或可交易状态文件。完整字段和校验规则见 [references/input-schema.md](references/input-schema.md)。

## 单日快速开始

```bash
python scripts/run_single_date.py \
  --config examples/config.yaml \
  --signal-file /path/to/predictions.parquet \
  --candidate-file /path/to/candidates.parquet \
  --covariance-file /path/to/covariance.parquet \
  --benchmark-file /path/to/benchmark_weights.parquet \
  --date 20230104 \
  --output-dir outputs/date=20230104
```

成功时标准输出类似：

```json
{
  "asset_count": 197,
  "date": "20230104",
  "optimized_active_volatility": 0.05552,
  "optimized_expected_return": 0.07836,
  "solver_iterations": 17,
  "status": "success"
}
```

核心产物包括：

- `target_weights.parquet`
- `constraint_diagnostics.json`
- `risk_summary.json`
- `signal_diagnostics.json`
- `run_manifest.json`
- `optimization_summary.json`

输出是研究用目标权重，不是可执行订单。消费产物前请阅读 [references/output-contract.md](references/output-contract.md)。

## 滚动实验

```bash
python scripts/run_rolling_experiment.py \
  --config examples/config.yaml \
  --signal-file /path/to/predictions.parquet \
  --candidate-file /path/to/candidates.parquet \
  --covariance-root /path/to/risk_model \
  --exposure-root /path/to/risk_model \
  --benchmark-file /path/to/benchmark_weights.parquet \
  --asset-returns-file /path/to/asset_returns.parquet \
  --transaction-cost-bps 7 \
  --output-dir outputs/rolling
```

动态风险建模、因子形式输入、缓存签名和断点续跑参数见 [references/risk-model.md](references/risk-model.md) 与 [references/backtest-contract.md](references/backtest-contract.md)。

## 验证

```bash
python scripts/validate_skill.py .
python -m compileall -q scripts
python -m pytest -q
```

同一 Alpha191+LightGBM 开发区间上的共同对比结果如下。基线是冻结的 Top 200 等权信号组合，不是市场指数：

| 组合 | 夏普率 |
| --- | ---: |
| Top 200 等权基线 | 1.0929 |
| v0.8.0 全市场信号校准 | 1.1471 |
| v0.9.0 正态秩变换最佳方案 | 1.2054 |
| v1.0.0 两阶段信号捕获最佳方案 | 1.2147 |

详细口径、约束证据和解释限制见 [references/validation-notes.md](references/validation-notes.md)。该区间已被反复用于开发，结果不能视为封存留出集，也不能证明方法对其他信号或市场阶段普遍有效。

## 项目边界

- 仅用于量化研究和组合构建实验。
- 不包含实时行情、经纪商连接、订单生成、成交冲击或实盘风控。
- 不应将回测结果表述为投资建议或收益保证。
- 外部数据与实验输出应保存在仓库之外；源码边界见 [references/source-boundary.md](references/source-boundary.md)。

## License

[GPL-3.0-only](LICENSE)
