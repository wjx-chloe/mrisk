# 第 4 周规划与交接（看板 + 离线样例 + README + docs）

这份文件是给接手的 session（包括云端 session）的交接。没看过之前聊天记录也能直接开工。
开工前按顺序读：`CLAUDE.md`（工作守则）→ `SPEC.md` → 本文件 → 与当前模块相关的 `docs/week*.md`。

---

## 0. 当前状态（2026-10-18）

- 第 1–3 周已验收：数据管道 + DQ、计量引擎、回测 + 归因、压测 + 限额。各周的设计说明在 `docs/`。
- 4A 看板已完成：见 `docs/week4a-dashboard.md`。**4B 已完成，待用户验收**：见 `docs/week4b-offline-demo.md`；Release 上传后还要跑一遍 `fetch-snapshot` 验证。
- 仓库：`https://github.com/wjx-chloe/mrisk`（第 4 周内私有，第 4 周结束时改为公开）。
- 测试：250 个，全部通过（`uv run pytest -q`，1.5–4 分钟，随机器负载波动）；另有 3 个 slow 回归测试，跑在 `MRISK_REGRESSION_DB` 指定的库上。
- 预置限额 5 条（`src/mrisk/limits/seed/limit_rule.csv`），最后一条是 2015 情景限额（损失 ≤ 市值 45%，warn 90%，含代理部分），见 `docs/week3b-stress-limits.md` §5.3。

### 0.1 环境约束（云端 session 必读）

- **没有真实数据库。** 真实库 `data/mrisk.db`（约 228 MB，2015-01 至 2026-09 的行情，2024-09-02 至 2026-09-18 共 497 个持仓日的全部结果）只在用户本机。`data/` 整个目录在 `.gitignore` 里。
- **连不上 akshare 的数据源。** 不要尝试 `mrisk backfill` / `mrisk daily`，它们会联网拉数。
- 在 4B 的离线快照发布到 GitHub Release 之前，能用的数据只有测试里的假数据源（`tests/test_jobs.py` 的 `FakeSource` 和 `env` / `risk_env` fixture：约 80 个交易日、12 只股票、窗口 20 天）。
- 依赖用 `uv sync` 安装。
- **pre-commit hook**：每个 clone 要先执行一次 `git config core.hooksPath .githooks`。之后每次提交前会跑 `pytest -q -m "not network and not slow"`，不过就拒绝提交。`slow` 标记留给需要真实库的回归测试（4B），`network` 标记是联网测试。

### 0.2 工作规矩（详见 CLAUDE.md，这里只列最容易忘的）

- 一次只做一个模块：4A → 4B → 4C，每个做完停下等用户验收。
- 每个模块动手前，先用几句话说清怎么实现，等用户回"开始"。
- 小步提交，一个逻辑单元一次 commit。
- 收尾：全部测试通过 → 在 `docs/` 写一页设计说明 → 停下。
- 不许擅自改 SPEC.md；不确定就问，不要默默兜底。
- engine 层是纯函数，不碰数据库和文件；引擎里不许按日期循环。

---

## 1. 已拍板的决定（本周）

| # | 决定 |
|---|---|
| 1 | 顺序：4A 看板 → 4B 离线样例 + README → 4C docs 收尾 |
| 2 | 离线方案选 A：用真实行情快照 + 回放数据源。**快照不进 git**，放在 GitHub Release 附件里 |
| 3 | 下载快照单独做成一条命令，带 sha256 校验。`mrisk demo` 本身全程不联网 |
| 4 | 95% 下的回测红绿灯按同一套二项分布规则重算，250 天窗口：绿 ≤ 17、黄 18–26、红 ≥ 27（99% 是 ≤ 4 / 5–9 / ≥ 10）。做不到就在 95% 下隐藏红绿灯 |
| 5 | 干净克隆的计时实测放到云端 session 做。云端本身连不上 akshare，不需要模拟断网 |
| 6 | 默认日期 = 组合层 `risk_result` 在每个置信度上都有行，且每条 active 限额当天都有 `limit_check` 行 |
| 7 | 红绿灯：SPEC §5.4 同步为"边界由二项分布规则按 (p, 窗口长度) 推出"；参数法也写 zone |
| 8 | 其余不变：入库，不在看板现算。`scenario.json` 为每个行业冲击加 raw 版本（`IND_<行业>_-10_RAW`）；top N 标明模式 |
| 9 | 限额连续天数：遇到 no_data 的日子中断计数 |

4B 开工时（2026-10-18）补充拍板：

| # | 决定 |
|---|---|
| 10 | 第 4 周结束时把仓库改为公开。快照放在本仓库的 GitHub Release，不另建仓库 |
| 11 | `fetch-snapshot` 默认匿名下载；环境变量里有 `GITHUB_TOKEN` 时带上，只用于私有期间测试下载流程 |
| 12 | Release 由用户在网页上传：打好包后报 tag 名、文件名和 sha256，用户上传后在本地跑一遍 `fetch-snapshot` 验证 |
| 13 | 全市场名单和申万成分原样保留，不裁剪 |
| 14 | README 写明：行情经 akshare 取自公开网站，仅供学习研究，权利人要求即删除 |
| 15 | README 和 docs 提到项目来源时，只写"参考券商市场风险系统的典型功能，用公开方法论自己设计"，不提任何文件 |

规则核对：二项分布 B(250, p)，P(X ≤ k) < 95% 为绿，< 99.99% 为黄，其余为红。
- p = 0.01：P(X ≤ 4) = 0.892，P(X ≤ 5) = 0.959，P(X ≤ 9) = 0.99975，P(X ≤ 10) = 0.99995；
- p = 0.05：P(X ≤ 17) = 0.921，P(X ≤ 18) = 0.953，P(X ≤ 26) = 0.99984，P(X ≤ 27) = 0.99993。

两组边界都与上表一致。

---

## 2. 4A 看板（已完成，以下为当时的规划；实际落地见 `docs/week4a-dashboard.md`，其中的"动手前确认"均已按 §1 第 6–9 条拍板）

### 2.1 结构

```
src/mrisk/dashboard/
├── app.py        入口：st.navigation 组织 8 个页面；侧边栏放日期和置信度
├── queries.py    只读取数：每个页面一个函数，(engine, 组合, 日期, 置信度) -> DataFrame
├── charts.py     Plotly 作图：DataFrame -> Figure
└── pages/        8 个页面
```

- **只读库，不重算风险。** 所有数字来自已入库的结果表。数据库以只读方式连接（SQLite `mode=ro`），看板写不了库。
- **依赖**：新增 `streamlit` 和 `plotly`（SPEC §8 已列出），放主依赖。
- 启动：`uv run streamlit run src/mrisk/dashboard/app.py`。

### 2.2 侧边栏与横幅

- **默认日期**取"最新一个有完整风险结果的日期"，不取最新持仓日。
  - 建议的定义：组合层的 `risk_result` 在 config 里的每个置信度（95%、99%）上都有行。**动手前和用户确认。**
- **横幅**：如果最新交易日被 DQ 门禁阻断，在页面顶部显示横幅，写明哪天、哪条规则、原因。原因来自 `jobs/gate.py` 的 `dq_gate`。
- **置信度**：默认 99%，可切到 95%。

### 2.3 8 个页面（SPEC §7）及补充要求

1. **总览**：市值、VaR（历史模拟和参数法）、ES，以及与上一日的变动；5 条限额的状态灯。
2. **趋势**：VaR / ES / 市值，可选近 1 月、3 月、半年、1 年。
3. **敞口**：行业分布条形图，个股 top 10。
4. **回测**：例外日历、Kupiec 和 Christoffersen 检验、红绿灯，损益与 −VaR 的对照图。
   - 目前 `engine/backtest.py` 的 `traffic_light` 把 99% 的边界写死了，job 只给"历史模拟 + 99% + 250 天窗口"写 zone，95% 的行没有 zone。
   - 计划：把 `traffic_light` 改成按二项分布规则、由 (p, 窗口长度) 推出边界。测试要复现 4 / 9 和 17 / 26 两组边界。job 改为两个置信度都写 zone。
   - **动手前确认**：
     - SPEC §5.4 是否需要同步；
     - 参数法是否也写 zone（目前只有历史模拟写）。
5. **归因**：当日 VaR 变化的分解（瀑布图），数据来自 `var_attribution`。
6. **压力测试**：情景损失表；选一个情景，看行业分解。
   - **走代理路径的股票不提供单股下钻**（或者置灰），因为单只股票的代理结果不准，与 `docs/week3b-stress-limits.md` §5.5 a 的结论一致：代理平均无偏，但单段残差的标准差有 8%。判断依据是个股行的 `proxy_mv_unlisted` 或 `proxy_mv_suspended` 大于 0。
   - **假设情景**：展示"条件传导"（β 传导）结果的地方，要和"其余不变"的结果并排放，并标注"基于近 250 日相关性"。
     - "其余不变"就是同一个冲击定义改用 `raw` 模式：只有被冲击的行业或股票受冲击，其余为 0。
     - 建议的算法：看板对库里个股层的市值调用纯函数 `engine.stress.hypothetical_shocks({"raw": ...})` 现算，不写库。**动手前确认**：也可以改为在 job 里预先算好入库。
7. **限额**：规则列表、当日检查结果、breach 历史。
   - breach 红色高亮，warn 黄色，no_data 灰色（SPEC §6.3）。
   - 区分"今日新触发"和"已持续 N 天"。N = 截至所选日期、状态连续相同的持仓日数；N = 1 就是今日新触发。
   - **动手前确认**：no_data 的日子是中断连续计数，还是跳过不计。
8. **数据质量**：当日 12 条规则的通过情况，以及门禁状态。

### 2.4 测试

- **第一步**：确认装到的 Streamlit 版本里，`streamlit.testing.v1.AppTest` 支持 `st.navigation`。如果不支持，就让 `app.py` 尽量薄，用 `AppTest.from_file` 逐页测，把侧边栏状态通过 `session_state` 注入。
- `queries.py` 的每个函数配 pytest 用例，跑在测试里现成的假数据库上（`risk_env` 等 fixture）。
- 每个页面用 AppTest 跑一遍：断言不报错。**数字的一致性用 `st.metric` 和 `st.dataframe` 的值来断言**，因为 AppTest 读不到 Plotly 图里的数据。这就是 SPEC §10"8 个页面可点"的验收。
- **图表的正确性在 `charts.py` 的测试里检查 `Figure.data`**，例如序列的长度、取值、颜色映射。
- 开发时可以在用户本机的真实库上看效果。云端 session 只能用假数据库，或者等 4B 的快照。

---

## 3. 4B 离线样例 + README（已完成，以下为当时的规划；实际落地见 `docs/week4b-offline-demo.md`）

### 3.1 快照与回放

- **`SnapshotSource`**（放在 `data/sources.py`）：实现和 `AkshareSource` 相同的 6 个方法，从本地快照目录读数据：
  - `trade_calendar`、`listings`、`sw_l1_members`、`delisted`、`daily_raw`、`hfq_factor`。
- config 增加 `data.source: akshare | snapshot`，由 `etl.make_source` 选择。走快照时**完全不 import akshare**，akshare 改成延迟导入。
- **`mrisk export-snapshot`**：从用户本机的 akshare 原始缓存（`data/raw/`，约 30 MB，含重复区间）导出快照，打包并输出 sha256。由用户把包上传到 GitHub Release。
  - 回放的输入就是当时拉到的原始数据，所以离线跑出来的数应与 docs 一致。这一点要核对。
  - `data/raw` 里同一只股票有多个区间。**重叠部分的数值如果不一致，不静默挑一个，列出来报告给用户**；保留的区间要覆盖库里实际用到的全部日期。
  - 行情压缩后实测约 3.4 MB（81 只股票，2015-01 至 2026-09，22.9 万行）；加上复权因子和主数据，预计 5–10 MB。
- **`mrisk fetch-snapshot`**：唯一联网的命令。从 Release 地址下载快照，按仓库里记录的 sha256 校验，解压到 `data/snapshot/`（在 `.gitignore` 里）。
  - 如果云端的网络策略连 GitHub Release 也不让下载，要先告诉用户。
  - 默认匿名下载；有 `GITHUB_TOKEN` 时带上（§1 第 11 条）。
- **`config.demo.yaml` 和 `mrisk demo`**：用快照做 backfill → risk → pnl → backtest → attribution → stress → limits，写到 `data/demo.db`，最后打印启动看板的命令。全程不联网。
  - **耗时要实测。** 全量 497 天里光压测就要 130 秒。如果全流程超过约 5 分钟，demo 配置缩短持仓期，但至少保留约 260 天，否则回测窗口一直处于 warming up。
- **测试**：把 akshare 设为不可 import（`monkeypatch.setitem(sys.modules, "akshare", None)`），在一个极小快照上跑 `mrisk demo`，确认整条流程不联网。

- **回放核对**：所有结果表逐行比对，只排除 `run_id`、`created_at` 这类运行元数据。同样的代码加同样的输入，预期逐位相同；不同的话，报告每张表的差异行数和最大差异。

### 3.2 回归测试（slow）

把 docs 里的关键数字写成 `@pytest.mark.slow` 的回归测试，跑在 `mrisk demo` 用全量配置生成的库上（库的路径通过环境变量给出）：
- 2015 情景损失占市值的最大值为 41.51%（497 天，`docs/week3b-stress-limits.md` §5.3）；
- 限额 2,485 行，状态分布为 ok 2,055、warn 383、breach 47、no_data 0（§5）。其中 L_STRESS_2015 为 ok 459、warn 38、breach 0。

如果 demo 缩短了持仓期，这些测试要用全量配置单独生成库。

### 3.3 README

内容：
- 一段话说明这是什么（"虚构组合在真实市场里的真实风险"：行情是真的，持仓由 mock 生成器造）；
- 最上面放 2–3 张看板整页截图。截图由用户自己截，放进 `docs/figures/` 后告知；
- 10 分钟上手：`uv sync` → `mrisk fetch-snapshot` → `mrisk demo` → `mrisk dashboard`；
- `mrisk dashboard` 以仓库根目录为工作目录启动 Streamlit，并显式传 `--server.address localhost`。安全默认不能取决于从哪个目录启动；
- 需要在局域网访问看板时怎么改：单次运行加 `--server.address 0.0.0.0`；
- 数据流图；
- 模块与 docs 的对应关系；
- 如何切回 akshare 实时拉数；
- 已知局限；
- 数据来源说明：行情经 akshare 取自公开网站，仅供学习研究，权利人要求即删除；
- 项目来源只写"参考券商市场风险系统的典型功能，用公开方法论自己设计"，不提任何文件。

### 3.4 干净克隆计时

在云端 session 里做：新 clone → 按 README 从头执行并计时，记录每一步的耗时。

- **等仓库公开后再做，不给云端 `GITHUB_TOKEN`**（2026-10-18 拍板）。陌生人就是匿名下载，测试也该用匿名下载。
- 匿名 `fetch-snapshot` 从现在的 404 变成 200，本身就是一项验收。

---

## 4. 4C docs 收尾

- `docs/week4-dashboard.md`：4A 和 4B 的设计说明（做了什么、关键取舍、怎么验证的），格式同前几周。
- `docs/overview.md`：一页总览，包括架构、数据流和各周文档的索引。
- 在 `docs/week4a-dashboard.md` §5 补两点：
  - 全样本二项检验的单侧 p 值：HS99 ≈ 0.013，P99 ≈ 0.065，HS95 ≈ 0.12，P95（例外偏少的方向）≈ 0.14。
  - 滚动窗口高度重叠：496 个观测、窗口 250，信息量大约只相当于两个互不重叠的窗口，黄灯天数和 Kupiec 拒绝天数不能当作相互独立的证据。
- 不新写别的文档。

---

## 5. Backlog（第 4 周之后在云端做）

下面几条大多需要真实数据，要等 4B 的快照发布后才能做。

### 5.1 代理偏差的置信区间（docs §5.5 a）

- 现状：27 个停牌段的复牌日残差，均值 −1.17%，标准差 7.99%。按 t 分布算，95% 置信区间约为 −1.17% ± 2.056 × 7.99% / √27 = **[−4.3%, +2.0%]**。
- 但残差不独立：复牌日扎堆，27 段里有 20 段在 07-07、07-08 停牌，在 07-09 至 07-20 之间复牌，正赶上 07-09 开始的反弹，所以实际区间更宽。可以按停牌或复牌日期聚类，或者用分块 bootstrap。
- 再换算成对组合层面的最大影响：偏差的区间端点 × 停牌代理市值占比。停牌代理市值占比取 `stress_result` 的 `proxy_mv_suspended` / `market_value`，看 497 天里的最大值。注意 BJ920682 属于"未上市"代理，不在这项核对的覆盖范围内。

### 5.2 限额规则引用不存在的情景时的行为

**现在的实际行为**（2026-10-07 在真实库的副本上，对 2026-09-18 实测）：

| 情况 | 现在的行为 |
|---|---|
| 规则引用的情景在 `scenario` 表里根本不存在（例如拼写错误） | `run_limits` 抛出 `ValueError: limit on unknown scenario …`，**整次运行失败，所有规则都不写**，这几天原有的 `limit_check` 行保持旧值。在 `mrisk daily` 里，前面的风险、损益、回测、归因、压测都已入库，只有限额这一步失败退出 |
| 情景已从 `scenario.json` 删除，但 `scenario` 表里还留着旧行 | 压测不再计算它，限额写 no_data，原因是"not computed for this date"。**原因有误导性**，看起来像是漏算，而不是配置失效 |
| VaR / ES 规则的置信度不在 config 里（例如 `var_hs_98`） | 同上：no_data，原因"not computed for this date" |
| 完全未知的指标名（例如 `foo`） | `evaluate` 抛出 `ValueError: rule … no values for metric foo`，整次运行失败，所有规则都不写 |

另外，加载种子数据时（`jobs/limits.py` 的 `load_rules`）**没有任何引用完整性校验**；`limit_rule.metric` 是自由文本，没有指向 `scenario` 的外键。

**目标**：
- 引用有问题的规则单独记为 no_data 或配置错误，其他规则照常检查；
- 加载种子数据时做一次引用完整性校验：情景存在于种子、指标名能解析、置信度在 config 里。

**待用户决定**：
- "配置错误"用新状态 `config_error` 表示，还是用 no_data 加上 `detail.reason = "config_error: …"`？前者要改 `limit_check` 的约束和 SPEC §3.8。
- 种子校验失败时，是拒绝加载整张表，还是只停用出问题的规则？

### 5.3 docs §5.3 的"25% 不可行"论证

- 用户要求写明：同期指数跌了三成多，组合只做多头，没有对冲手段。
- **现状：没写全。** 现在只有"25% 相当于要求留约三分之一现金，是在否定策略本身"，没有"同期指数跌三成多"和"只做多头、无对冲"这两点。
- 指数跌幅需要数据来源：快照里只有股票池，可以算股票池等权收益，但要标明它不是指数。沪深 300 或上证指数的跌幅需要用户提供数字，或者另找离线来源。不要凭记忆写。

### 5.4 2015 情景 warn 段的持仓归因

- 2015 情景的路径对每只股票是固定的：情景收益 r_i 不随持仓日变化。所以损失占比 L_t = Σ w_{i,t} r_i，变化量 ΔL = Σ Δw_i · r_i 是**精确恒等式**，没有交叉项。
- 做法：对两段 warn（2026-01-05 至 02-04、02-06 至 03-06）各取一个基准日（warn 段开始前最后一个 ok 日）和峰值日，按行业汇总 Δw_i · r_i，看是哪些行业、哪些股票把损失推过了 40.5%。
- 结果补进 `docs/week3b-stress-limits.md` §5.3：目前那里写的是"这两段的成因没有单独分解"。

### 5.5 测试提速

- 整轮测试只建一次假数据库模板文件，每个测试复制一份，不再逐个重建。
- 只用假数据，适合云端做。

### 5.6 进阶：波动率调整的历史模拟

- 用 EWMA 波动率把历史收益缩放到当前水平，针对历史模拟 95%、99% 都偏激进的滞后问题。
- 用 `docs/week4a-dashboard.md` 里同一张四格表（方法 × 置信度）验证改进效果。

### 5.7 进阶：基于例外间隔的独立性检验（Christoffersen–Pelletier）

- 一阶马尔可夫检验只能看见连续两天的例外，06-08 至 08-19 那种隔几天一次的扎堆它看不见。
