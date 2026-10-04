# SPEC.md — 股票组合市场风险管理系统（精简版）

> 本文档是写给 Claude Code 的需求规格，也是写给我自己的学习笔记。
> 背景：券商的市场风险管理系统通常覆盖估值、VaR、回测、压力测试和限额监控，本项目只取其中与股票组合相关、统计味最浓的部分。
> 标有 ⚠️ 的地方是需要我拍板的决策，括号里是默认值；不确定时直接用默认值。

---

## 0. 一句话目标

给定一个 A 股股票组合的每日持仓和行情，系统每天自动回答四个问题：

1. 组合现在值多少钱，风险暴露在哪里？（估值与敞口）
2. 明天最坏情况会亏多少？（VaR / ES）
3. 我算出来的"最坏情况"靠谱吗？（回测）
4. 有没有触碰红线？（压力测试与限额监控）

---

## 1. 范围

### 做
- 资产类别：**仅 A 股股票**（含北交所，用来处理北交所"920 号段代码切换"这个真实业务问题）
- 数据管道：行情与持仓的接入、校验、入库
- 计量引擎：市值、收益率、VaR（历史模拟法 + 参数法）、ES、组合层级汇总
- 模型验证：VaR 回测（Kupiec 检验、巴塞尔红绿灯）
- VaR 差异归因（简化版）
- 压力测试：历史情景 + 假设情景
- 限额管理：规则配置、每日监控、告警
- 监控看板：一个能看的页面，不追求美观

### 不做（完整的市场风险系统通常有，本项目明确放弃）
- 债券、期权、期货等一切非股票资产（→ 不需要 DV01、久期、凸性、Delta/Gamma/Vega）
- 利率曲线、信用曲线
- 国产化适配（国产数据库、操作系统、中间件）——这是迁移工作，不是功能
- 权限管理、机构/部门/角色（单用户系统）
- 报表设计器、工作台设计器（Streamlit 页面写死即可）
- 与真实交易系统、资讯商的接口（用 mock 数据生成器替代）

---

## 2. 术语速查（用统计的语言）

| 术语 | 金融里的意思 | 翻译成统计 |
|---|---|---|
| 头寸 / 持仓 (position) | 某天持有某只股票多少股 | 一条记录：(日期, 股票代码, 数量) |
| 市值 (market value) | 数量 × 收盘价 | 一个加权和 |
| 收益率 (return) | 今天价格相对昨天的变化 | 默认简单收益 P_t / P_{t-1} − 1（可按市值线性加权成组合收益）；对数收益 ln(P_t / P_{t-1}) 作为可选项。是我们的随机变量 |
| 组合损益 P&L | 组合一天赚了或亏了多少钱 | 各股收益率以市值加权求和，再乘以组合总市值 |
| VaR (Value at Risk) | "在 99% 的情况下，一天最多亏 X" | 损失分布的 99% 分位数 |
| ES (Expected Shortfall) | "在那 1% 的坏情况里，平均亏多少" | 超过 VaR 的条件期望 E[L \| L > VaR] |
| 历史模拟法 | 用过去 N 天真实收益率当作未来的可能性 | 经验分布的分位数，非参数 |
| 参数法 (方差-协方差法) | 假设收益率服从正态分布 | 用样本协方差矩阵算组合方差，再乘 z 值 |
| 置信度 | VaR 的 99% | 分位数水平 α |
| 持有期 | VaR 衡量的时间跨度 | 我们只做 1 天 |
| 回测 (backtesting) | 检查历史上实际亏损超过 VaR 的次数是否合理 | 假设检验：H0: 超越率 = 1% |
| 突破 / 例外 (exception) | 某天实际亏损 > 当天 VaR | 一次伯努利试验中的"成功" |
| Kupiec 检验 | 回测的标准检验 | 对超越次数做似然比检验，服从 χ²(1) |
| 压力测试 | "如果 2015 年股灾重演，我们亏多少" | 把特定情景的收益率向量代入组合 |
| 敞口 (exposure) | 组合在某个维度上暴露了多少 | 按行业/个股分组的市值汇总与占比 |
| 限额 (limit) | 风控部门定的红线 | 一个阈值 + 一个比较规则 |
| 差异归因 | 今天 VaR 比昨天变了，是持仓变了还是市场变了？ | 把变化量分解为两个（或三个）部分 |
| 北交所 920 号段 | 北交所股票代码从 8 开头改成 920 开头（2025 年） | 同一实体的主键变更，需要映射表 |

---

## 3. 数据模型

数据库：PostgreSQL（开发阶段可用 SQLite，表结构保持兼容）。

### 3.1 instrument — 证券主数据
| 字段 | 类型 | 说明 |
|---|---|---|
| instrument_id | text PK | 内部统一 ID，形如 `SH600000` / `BJ920001` |
| exchange | text | SH / SZ / BJ |
| symbol | text | 交易所代码，6 位 |
| name | text | 股票简称 |
| industry_l1 | text | 一级行业（申万或中信，⚠️ 默认申万） |
| board | text | main（主板）/ chinext（创业板）/ star（科创板）/ bj（北交所），由代码推断，DQ04 涨跌幅限制用 |
| list_date | date | 上市日 |
| status | text | active / delisted |

### 3.2 instrument_code_map — 代码变更映射
解决北交所 920 号段切换这类问题：同一只股票在不同时间段用不同代码。
| 字段 | 类型 | 说明 |
|---|---|---|
| instrument_id | text FK | |
| old_symbol | text | 切换前代码 |
| new_symbol | text | 切换后代码 |
| effective_date | date | 生效日 |

### 3.3 daily_price — 日行情
| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | date | 复合主键 |
| instrument_id | text FK | 复合主键 |
| close | numeric | 收盘价 |
| adj_close | numeric | 后复权收盘价（算收益率用这个）。= close × 当日生效的后复权因子；历史值不随新的除权事件改变，回灌数据不会过时 |
| volume | bigint | 成交量 |
| is_suspended | bool | 是否停牌 |
| ingested_at | timestamp | 入库时间（北京时间），DQ10 用 |
| load_type | text | backfill（历史回灌）/ daily（每日增量），DQ10 只评估 daily |

### 3.4 portfolio — 组合定义
| 字段 | 类型 |
|---|---|
| portfolio_id | text PK |
| name | text |
| base_currency | text（固定 CNY） |

### 3.5 position — 日终持仓
| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | date | 复合主键 |
| portfolio_id | text FK | 复合主键 |
| instrument_id | text FK | 复合主键 |
| quantity | numeric | 持仓股数，可为负（融券），⚠️ 默认只做多头 |
| cost_price | numeric | 成本价（可选） |

### 3.6 risk_result — 每日风险指标结果
长表：每个置信度各存一行，主键 (trade_date, portfolio_id, level, level_key, confidence)。
| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | date | |
| portfolio_id | text | |
| level | text | portfolio / industry / instrument |
| level_key | text | 层级键，组合级为 portfolio_id，行业级为行业名，个股级为 instrument_id |
| confidence | numeric | 置信度，0.95 / 0.99 |
| market_value | numeric | |
| var_hs | numeric | 历史模拟法 VaR，1 日，置信度见 confidence |
| var_param | numeric | 参数法 VaR |
| es_hs | numeric | 历史模拟法 ES |
| run_id | text | 本次计算批次 |

回测用的损益不放在这张表里，见 §3.12 pnl_hypo。

### 3.7 limit_rule — 限额规则
| 字段 | 类型 | 说明 |
|---|---|---|
| rule_id | text PK | |
| portfolio_id | text | |
| metric | text | var_hs_99 / es_hs_99 / concentration_instrument / concentration_industry |
| scope | text | portfolio / industry:银行 / instrument:SH600000 |
| operator | text | `<=` / `<` |
| threshold | numeric | |
| threshold_type | text | absolute（金额）/ pct_of_mv（占市值比例） |
| severity | text | warn / breach |
| is_active | bool | |

### 3.8 limit_check — 限额监控结果
| 字段 | 类型 |
|---|---|
| trade_date | date |
| rule_id | text |
| actual_value | numeric |
| threshold | numeric |
| status | text（ok / warn / breach） |

### 3.9 dq_rule / dq_result — 数据质量规则与结果
规则用 SQL 或 Python 表达式描述，结果记录每日每条规则的通过/失败及失败明细。

dq_rule：
| 字段 | 类型 | 说明 |
|---|---|---|
| rule_id | text PK | DQ01…DQ12 |
| name | text | |
| category | text | 完整性 / 准确性 / 一致性 / 及时性 / 合理性 |
| severity | text | blocking / warn |
| description | text | |
| is_active | bool | |

dq_result：主键 (run_id, check_date, rule_id)
| 字段 | 类型 | 说明 |
|---|---|---|
| run_id | text | 本次校验批次 |
| check_date | date | 被校验的交易日 |
| rule_id | text FK | |
| status | text | pass / fail / skip（当日无可检查对象） |
| n_checked | integer | 检查条数 |
| n_failed | integer | 失败条数 |
| detail_json | jsonb | 失败明细样本（最多 20 条） |
| created_at | timestamp | |

### 3.10 scenario — 压力测试情景
| 字段 | 类型 | 说明 |
|---|---|---|
| scenario_id | text PK | |
| type | text | historical / hypothetical |
| definition | jsonb | historical: {start, end}；hypothetical: {market: -0.05} 或 {industry: {银行: -0.10}} |

### 3.11 trade_calendar — 交易日历
DQ01 对照、停牌补行、调度判断是否交易日都以此为准。
| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | date PK | 交易日 |

### 3.12 pnl_hypo — 假设损益
本系统没有真实成交，回测用的是"持仓不动"的假设损益：持有 t 日收盘持仓到下一交易日 t+1。
主键 (position_date, portfolio_id, level, level_key)。
| 字段 | 类型 | 说明 |
|---|---|---|
| position_date | date | 持仓日 t（对应 risk_result.trade_date） |
| pnl_date | date | 损益实现日 t+1（下一交易日） |
| portfolio_id | text | |
| level / level_key | text | 同 risk_result |
| market_value | numeric | t 日市值 = Σ quantity × close_t |
| pnl | numeric | t 日市值 × 后复权简单收益 (adj_close_{t+1} / adj_close_t − 1)，t+1 停牌的股票收益为 0。与 VaR 的 PnL 同口径，不用未复权价差 |
| run_id | text | |

### 3.13 backtest_result — 回测结果
主键 (asof_date, portfolio_id, method, confidence, window_days)。
| 字段 | 类型 | 说明 |
|---|---|---|
| asof_date | date | 窗口内最后一个观测的损益实现日 |
| portfolio_id | text | |
| method | text | hs / param |
| confidence | numeric | 0.95 / 0.99 |
| window_days | integer | 观测数窗口，250 |
| n_obs | integer | 窗口内观测数 |
| n_exceptions | integer | 例外数 |
| lr_pof / p_value_pof | numeric | Kupiec POF 统计量与 p 值 |
| lr_ind / p_value_ind | numeric | Christoffersen 独立性统计量与 p 值 |
| zone | text | green / yellow / red，仅 method = hs 且 confidence = 0.99；其余为空 |
| run_id | text | |

### 3.14 var_attribution — VaR 差异归因
主键 (trade_date, portfolio_id, metric, confidence, effect, level, level_key)。
| 字段 | 类型 | 说明 |
|---|---|---|
| trade_date | date | t（解释 t−1 → t 的变化） |
| portfolio_id | text | |
| metric | text | var_hs / var_param / es_hs |
| confidence | numeric | 0.95 / 0.99 |
| effect | text | quantity / revaluation_scale / revaluation_drift / window / total，见 §5.3；前四项之和 = total |
| level | text | portfolio / instrument |
| level_key | text | 组合级为 portfolio_id，个股级为 instrument_id |
| value | numeric | 组合级为效应值；个股级为该股的 Euler 贡献，各股之和等于组合级 |
| window_dropped | date | 仅 effect = window：当天移出窗口的收益日期 |
| run_id | text | |

---

## 4. 模块一：数据管道

### 4.1 数据源
- 行情：⚠️ 默认 **akshare**（免费，覆盖沪深北）；备选 tushare（需积分）、baostock
- 行业分类：akshare 的申万行业接口
- 持仓：**mock 生成器**，生成一个 30–50 只股票、跨 5+ 行业、每周有小幅调仓的模拟组合，时间跨度 ≥ 2 年

### 4.2 ETL 任务
1. `fetch_prices(start, end)`：拉取行情 → 校验 → 写 daily_price
2. `fetch_instruments()`：更新证券主数据与行业
3. `apply_code_map()`：处理代码变更（920 号段）
4. `generate_positions()`：生成或加载持仓
5. 调度：⚠️ 默认 **APScheduler**（工作日 16:30，北京时间），不上 Airflow。早于 DQ10 的 17:00 到达时限，否则每日增量必然判失败

### 4.3 数据质量规则（首批 12 条，生产系统通常内置上百条校验规则）
| 编号 | 规则 | 类型 |
|---|---|---|
| DQ01 | 交易日行情不缺失（对照交易日历） | 完整性 |
| DQ02 | 持仓中的每只股票当日有行情或标记停牌 | 完整性 |
| DQ03 | 收盘价 > 0 | 准确性 |
| DQ04 | 单日涨跌幅在涨跌停限制内（主板 ±10%，创业板/科创板 ±20%，北交所 ±30%，ST ±5%） | 准确性 |
| DQ05 | 后复权价与原始价的比例（后复权因子）在同一股票内随时间单调不减 | 准确性 |
| DQ06 | 持仓数量为 100 的整数倍（A 股一手 100 股；北交所允许 100 + 任意增量） | 准确性 |
| DQ07 | 持仓中不含已退市股票 | 一致性 |
| DQ08 | instrument 表中的 symbol 在交易所内唯一 | 一致性 |
| DQ09 | 代码映射表中新旧代码不重叠 | 一致性 |
| DQ10 | 行情数据到达时间（ingested_at）在当日 17:00 前；只评估 load_type=daily 的行，历史回灌数据记为 skip | 及时性 |
| DQ11 | 组合总市值日环比变化 < 30%（防持仓数据错误） | 合理性 |
| DQ12 | 每日新增行情条数与前一日差异 < 5% | 合理性 |

规则失败时：写 dq_result，严重级别为 blocking 的规则失败则阻断当日计算。

---

## 5. 模块二：计量引擎

所有计算写成纯函数，输入 DataFrame/ndarray，输出 DataFrame，不碰数据库。数据库读写由调用层负责。**每个函数必须有单元测试，且至少一个测试用手算或极小样本验证数值。**

### 5.1 参数（config.yaml）
```yaml
confidence: [0.95, 0.99]  # 两个都算、都落库；回测与限额用 0.99
holding_period_days: 1
lookback_days: 250        # ⚠️ 默认 250 个交易日 ≈ 1 年
return_type: simple       # simple / log；默认 simple，因为组合损益要按市值线性加权，对数收益不可线性加总
parametric_dist: normal   # normal / t（t 为进阶）
```

### 5.2 计算步骤（每日一次）
1. **市值**：`mv = quantity × close`，按个股 / 行业 / 组合三层汇总
2. **收益率矩阵**：过去 lookback_days 天、组合内所有股票的日收益率，形状 (T, N)。停牌日收益率填 0 并记录
3. **权重向量** w：各股市值 / 组合总市值
4. **历史模拟法 VaR**：
   - 组合历史损益序列 `pnl_t = MV_total × (R_t · w)`，得到 T 个值
   - `VaR_hs = −quantile(pnl, 1 − confidence)`（取正数表示损失）
   - ⚠️ 分位数插值方法默认 `linear`（numpy 默认）
5. **参数法 VaR**：
   - `Σ = cov(R)`，`σ_p = sqrt(wᵀ Σ w)`
   - `VaR_param = MV_total × z_α × σ_p`，z_α = Φ⁻¹(α)（标准正态分位数，取精确值，99% 时约 2.3263，95% 时约 1.6449）
6. **ES**：`ES_hs = −mean(pnl[pnl < −VaR_hs])`
7. **分层 VaR**：个股级 VaR 用单只股票的历史损益算；行业级用行业子组合算。**不要求各层 VaR 可加**（VaR 本身不满足次可加性，这是面试常考点）
8. 写入 risk_result

### 5.3 VaR 差异归因（简化版）
记 V(W, P) 为用收益率窗口 W、持仓市值向量 P 算出的风险指标；q 为持股数量，c 为收盘价。今日 − 昨日按顺序分解为三项：
- **数量效应**：V(W_{t−1}, q_t · c_{t−1}) − V(W_{t−1}, q_{t−1} · c_{t−1})
- **重估效应**：V(W_{t−1}, q_t · c_t) − V(W_{t−1}, q_t · c_{t−1})，再拆成两部分：
  - 记 V1 = V(W_{t−1}, q_t · c_{t−1})，V2 = V(W_{t−1}, q_t · c_t)，λ = Σ q_t · c_t / Σ q_t · c_{t−1}
  - **规模**：(λ − 1) · V1，即组合整体涨跌带来的变化
  - **权重漂移**：V2 − λ · V1，即个股涨跌不同步、权重改变带来的变化；所有股票同比例涨跌时为 0
  - 依据：三个指标都是一次齐次的，V(W, λP) = λ · V(W, P)，所以不需要额外评估
- **窗口效应**：V(W_t, q_t · c_t) − V(W_{t−1}, q_t · c_t)，同时记下当天移出窗口的收益日期
- 数量、规模、权重漂移、窗口四项之和精确等于 V_t − V_{t−1}；不调仓的日子数量效应精确为 0
- 对 var_hs、var_param、es_hs 两个置信度都做

个股贡献用 Euler 分配，各股之和精确等于组合值：
- var_hs：取决定分位数的两个情景日（线性插值的上下两天），按插值权重对个股当天损益加权；
- es_hs：尾部各情景日个股损益的平均；
- var_param：z_α × e_i × (Σe)_i / σ_p。
每个效应的个股贡献 = 前后两次评估的个股分量之差；规模和权重漂移的个股贡献分别为 (λ − 1) · 分量_i(V1) 和 分量_i(V2) − λ · 分量_i(V1)（Euler 分量同样一次齐次）。

输出：组合级四项效应 + 个股级贡献（展示 top 10），写入 var_attribution。

### 5.4 回测
每日用"t 日算的 VaR"对比"t → t+1 的假设损益"（pnl_hypo，组合级）：
- **例外**：`pnl < −VaR`（严格小于）
- **范围**：method ∈ {hs, param} × confidence ∈ {0.95, 0.99} 都做检验；红绿灯只用于 hs × 0.99
- **窗口**：滚动 250 个观测；窗口内任一观测缺失（例如被 DQ 拦下）时，不输出该日结果
- **Kupiec POF 检验**：
  - 样本 T 天，例外数 x，期望超越率 p = 1 − confidence
  - `LR = −2 ln[(1−p)^(T−x) p^x] + 2 ln[(1−x/T)^(T−x) (x/T)^x]`，x = 0 或 x = T 时按 0·ln0 = 0
  - LR ~ χ²(1)，5% 显著性水平临界值取 χ²₁ 的 95% 分位精确值（约 3.8415）
- **Christoffersen 独立性检验**：对窗口内相邻观测的转移计数 n00、n01、n10、n11
  - π01 = n01/(n00+n01)，π11 = n11/(n10+n11)，π = (n01+n11)/(n00+n01+n10+n11)
  - `LR_ind = −2 ln[(1−π)^(n00+n10) π^(n01+n11)] + 2 ln[(1−π01)^n00 π01^n01 (1−π11)^n10 π11^n11]` ~ χ²(1)
  - 计数为 0 的项按 0·ln0 = 0
- **巴塞尔红绿灯**（250 天窗口）：例外 ≤ 4 绿，5–9 黄，≥ 10 红
- 输出：backtest_result + 例外日历图

### 5.5 性能要求
50 只股票、250 天窗口，全部指标单次计算 < 1 秒（NumPy 向量化，不允许 Python 循环遍历日期）。

---

## 6. 模块三：压力测试与限额

### 6.1 压力测试
- **历史情景**（⚠️ 默认预置这三个）：
  - 2015-06-12 至 2015-07-08（2015 股灾）
  - 2018-01-29 至 2018-02-09（2018 初回调）
  - 2020-01-23 至 2020-02-03（疫情冲击，含长假后首日）
  - 计算：把情景期间每日收益率代入当前持仓，输出累计损益、最大单日损益、按行业分解
- **假设情景**：
  - 全市场 −5% / −10%
  - 某一行业 −10%，其余不变
  - 用户自定义：JSON 描述
- **重大损失情景**：跑所有情景，按损失排序输出 top N

### 6.2 限额规则（⚠️ 默认预置）
| 规则 | 阈值 |
|---|---|
| 组合 VaR（历史模拟，99%） | ≤ 组合市值 × 3% |
| 组合 ES | ≤ 组合市值 × 4% |
| 单只股票市值占比 | ≤ 10% |
| 单一行业市值占比 | ≤ 30% |
| 2015 股灾情景累计损失 | ≤ 组合市值 × 25% |

每条规则有 warn 阈值（80% 触发）和 breach 阈值（100% 触发）。

### 6.3 监控与告警
- 每日计算完成后自动跑所有 active 规则，写 limit_check
- breach 触发：写日志 + 看板红色高亮 + ⚠️ 默认 **不发邮件**（进阶可接 SMTP）

---

## 7. 模块四：监控看板

技术：⚠️ 默认 **Streamlit**（不做 React）。要求：能看、能点、有数，不要求美观。

页面清单（即风险系统常说的"驾驶舱"）：
1. **总览**：指定日期的组合市值、VaR（两种方法）、ES，及与上一日的变动；限额状态灯
2. **趋势**：VaR / ES / 市值近 1 月、3 月、半年、1 年折线图
3. **敞口**：行业分布饼图/条形图，个股 top 10
4. **回测**：例外日历、Kupiec 检验结果、红绿灯状态
5. **归因**：今日 VaR 变化的分解
6. **压力测试**：各情景损失表，可选情景查看行业分解
7. **限额**：规则列表、当日检查结果、历史突破记录
8. **数据质量**：当日各规则通过情况

---

## 8. 技术栈

| 层 | 选择 |
|---|---|
| 语言 | Python 3.11+ |
| 数据处理 | pandas, numpy, scipy |
| 数据库 | PostgreSQL（SQLAlchemy），开发期 SQLite |
| 行情 | akshare |
| 调度 | APScheduler |
| API（可选） | FastAPI |
| 看板 | Streamlit + Plotly |
| 测试 | pytest |
| 依赖管理 | uv 或 poetry |

---

## 9. 项目结构

```
mrisk/
├── CLAUDE.md
├── SPEC.md
├── config.yaml
├── src/mrisk/
│   ├── data/        # 拉数、校验、mock 生成
│   ├── engine/      # 纯计算函数：returns, var, es, attribution, stress, backtest
│   ├── limits/      # 规则引擎
│   ├── db/          # 表定义与读写
│   ├── jobs/        # 每日调度任务
│   └── dashboard/   # Streamlit
├── tests/
├── docs/            # 每个模块完成后的设计说明
└── notebooks/       # 探索与验证
```

---

## 10. 里程碑（4 周）

| 周 | 交付 | 验收标准 |
|---|---|---|
| 1 | 数据模型 + 数据管道 + mock 持仓 + 12 条 DQ 规则 | 2 年行情入库，DQ 全部通过，代码映射表处理一只北交所股票 |
| 2 | 计量引擎（市值、VaR ×2、ES、分层）+ 单元测试 | 3 只股票的手算 VaR 与程序结果一致到小数点后 4 位 |
| 3 | 回测 + 归因 + 压力测试 + 限额 | Kupiec 检验跑通；历史情景损失可复算；限额有 breach 记录 |
| 4 | 看板 + docs + README | 8 个页面可点；README 能让陌生人 10 分钟跑起来 |

---

## 11. ⚠️ 待我拍板的决策清单

| # | 问题 | 默认值 | 影响 |
|---|---|---|---|
| 1 | 行业分类用申万还是中信？ | 申万 | 只影响敞口分组的标签 |
| 2 | 行情数据源用 akshare 还是 tushare？ | akshare | tushare 数据更稳但要积分 |
| 3 | 置信度 99% 还是 95%？ | 99%（巴塞尔标准） | 99% 例外少，回测统计功效低；可以两个都算 |
| 4 | 回望窗口 250 天还是 500 天？ | 250 | 越长越稳但越不敏感 |
| 5 | 是否支持空头（负持仓）？ | 不支持 | 支持会让 mock 与限额规则复杂一档 |
| 6 | 看板用 Streamlit 还是写 FastAPI + React？ | Streamlit | 后者多花一周，简历上不加分 |

---

## 12. 每个模块对应的简历/面试素材

| 模块 | 投产品岗 | 投数分岗 | 投风控岗 |
|---|---|---|---|
| 数据管道 + DQ | "如何把模糊需求转成可执行规则" | "指标体系与数据质量框架设计" | "数据治理是风控的前提" |
| 计量引擎 | — | "向量化实现与性能优化" | "VaR 两种方法的取舍、VaR 不可加性" |
| 回测 | — | "假设检验在模型监控中的应用" | "Kupiec 检验、巴塞尔红绿灯" |
| 归因 | "如何解释一个数字为什么变了" | "指标异动归因方法" | "持仓效应 vs 市场效应" |
| 压测 + 限额 | "规则引擎的产品设计" | — | "情景设计、限额体系" |
| 看板 | "从需求清单到页面的取舍" | "监控看板设计" | — |
