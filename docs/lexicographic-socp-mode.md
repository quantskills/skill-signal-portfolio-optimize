# Mode: Lexicographic SOCP (`lexicographic_signal_cost`, schema 5)

Branch: `v1.8.0-lexicographic-socp`. 本文档汇总该模式在 4 个冻结信号上的全部验证结论
（2025-2026 窗口，oos_tuning，样本内；样本外能力须以未来数据验证）。

## 方法定义

两阶段词典序（lexicographic）目标，CVXPY + Clarabel 二阶锥规划求解：

1. **第一阶段**：在 TE/换手/单票/行业/风格/满仓约束下，最大化主动信号效用 s'(w−b) → U*；
2. **第二阶段**：在 s'(w−b) ≥ 0.995·U*（`minimum_signal_capture: 0.995`）下，
   最小化单边换手成本 + L2 稳定项。

风险模型：Barra 式结构模型（5 风格 SIZE/BETA/MOMENTUM/RESVOL/NLSIZE + 31 行业），月频刷新；
TE 约束用因子形式 X'F'X（36×36）。执行：stockdemo-compat，次日 TWAP、1.4‰、keep=1.0、
100 股手数、ST/涨跌停过滤、停牌 carry-forward。

## 通用默认参数（已锁定，新信号直接复用）

`defaults/default_loose_te08_turn20_mw040_ind2.yaml`：

| 项 | 值 |
|---|---|
| TE 上限 | 8% |
| 换手预算 | 20%/期（单周 REBAL=5） |
| 单票上限 | 4%（max_weight 与 max_active_weight 同调） |
| 行业偏离 | ±2%（31 行业 PIT 冻结标签） |
| 风格中性 | SIZE = 0±5σ、BETA = 0±0.3σ（其余仅建模不约束；schema5 强制 SIZE 启用） |

## 验证证据（4 信号 × 4 配置迁移矩阵，官方 hedged 超额Sharpe）

| 信号 | 官方基线 | 紧(6/12/2) | 中(8/15/3) | 松(8/20/4) | FVopt(10/20/4) |
|---|---|---|---|---|---|
| LGBM300 | 2.4427 | 4.5517 | 4.6008 | 4.1206 | 4.3138 |
| FactorVAE | 1.8053 | 1.6950 ✗ | 2.0470 | 2.2272 | 2.4909 |
| TRA | 1.5241 | 3.1142 | 2.8173 | 2.9642 | 2.6397 |
| small_moe | 2.0302 | 3.2666 | 3.1242 | 3.0902 | 不可行（数值） |

- **通用默认 = 宽松组**：唯一无绝对失败的配置（最差格 +23.4%，4/4 超各自基线 +23%~+95%）；
  三配置均值打平（~+60%）。低成本备选 = 中间档；紧组仅适合分散排名型信号特化。
- **参数不通用、优化器数学通用**：跨信号迁移损失 -6%~-10%，四个信号四个冠军——无免费午餐。

## 机制归因（臂 A/C 消融实证）

- **回撤降低 100% 来自优化器，唯一主控 = TE 约束**：相对MDD 随 TE 单调
  （6/8/10/12% → 15.0/15.4/16.5/18.2%），TE 绑定 75/77 期；
- **行业 ±2% 是 Sharpe 保护墙**（去掉 -6.5% Sharpe、MDD 微降）——正则化信号噪声而非压回撤；
- **BETA/SIZE 在 FV 上无因果贡献**（不绑定，冗余保险）；
- **每周调仓的价值在成本控制而非 alpha**：同引擎同成本下每日调仓对 3/4 快信号更好
  （TRA -31%、small_moe -27%、FV -4%，仅 LGBM +8%）；每周把换手从 19%/日压到 ~4%/日。

## 新信号接入流程（5 步）

1. 体检信号（长表 date/ticker/prediction；ticker 转 `XXXXXX.SZ/SH`）；
2. 建实验目录、软链 derived/prepared/shared、复制本默认配置为 `generated_te08_turn20_mw040_ind2.yaml`；
3. `REBAL=5 bash scripts/run_case.sh validation te08_turn20_mw040_ind2`；
4. 官方 hedged 口径打分（hedged=(1+组合日收益−指数日收益).cumprod()，首行收益=首日净值/初始−1）；
5. 结果记入迁移矩阵（滚动元训练集）。可选：紧组对照用于分散排名型信号极限挖掘。

## 免责声明

全部结论系 oos_tuning 样本内结果；时间切分（2025 调参/2026 冻结验证）与成本压力测试
按项目决定暂未执行。数据：`transfer_matrix.csv`（4×4 迁移矩阵原始数据）。
