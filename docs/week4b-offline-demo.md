# 第 4 周 B：离线样例、回放核对与 README

## 1. 做了什么

| 命令 | 作用 | 联网 |
|---|---|---|
| `mrisk export-snapshot` | 把 akshare 原始缓存 `data/raw/` 打成快照 zip，打印 tag、文件名、大小和 sha256 | 否 |
| `mrisk fetch-snapshot` | 从 GitHub Release 下载快照，按 `config.demo.yaml` 里记录的 sha256 校验，逐个文件核对清单后解压到 `data/snapshot/` | **是（唯一的一条）** |
| `mrisk demo` | 用快照在一个新库 `data/demo.db` 上跑完整条流程 | 否 |
| `mrisk dashboard` | 启动看板 | 否 |
| `mrisk diff-db A B` | 两个库逐表逐行比对，排除运行元数据；有差异时退出码为 2 | 否 |

新增代码：

- `data/sources.py` 的 `SnapshotSource`：与 `AkshareSource` 相同的 6 个方法。读的是快照里的原始 akshare 帧，经过同一组 `norm_*` 函数，所以回放的输入和当时拉到的完全一样。缺文件直接报错，不会退回联网。
- `config.yaml` 新增 `data.source: akshare | snapshot` 和 `data.snapshot_dir`，由 `etl.make_source` 选源。akshare 原本就只在 `_call` 里延迟导入，走快照时不会 import。
- `data/snapshot.py`：导出、打包、下载、校验、解压。
- `jobs/demo.py`：回灌 → risk → pnl → backtest → attribution → stress → limits，每一步记录耗时和各状态的天数。
- `db/compare.py`：回放核对。`readonly_engine` 从看板挪到 `db/repo.py`，CLI 和看板共用。
- `config.demo.yaml`：与 `config.yaml` 只差 4 处，都标了「demo」：库路径、`data.source`、持仓落地目录、`snapshot_release`。除了这 4 处，其余必须与 `config.yaml` 完全相同，由测试守着。

## 2. 快照

### 2.1 格式

- 一个原始缓存键对应一个 parquet，去掉文件名里的拉取日期：`calendar`、`list_sh_main`、`list_sh_star`、`list_sz`、`list_bj`、`delist_sh`、`delist_sz`、`sw_l1`、31 个 `sw_comp_<行业代码>`、81 个 `daily_<代码>`、81 个 `hfq_<代码>`，共 201 个，外加 `manifest.json`。
- `manifest.json` 记录每个文件的 sha256 和行数、最后价格日 `price_end`、拉取日期，以及数据来源声明。
- 全市场名单和申万成分原样保留，不裁剪（§1 第 13 条）。
- zip 是确定性的：文件名排序、时间戳固定。同样的缓存打两次，sha256 相同（已实测）。

### 2.2 同一只股票的多个区间

`data/raw` 里 81 只股票各有两个日线文件：`19901219_20260918` 和 `20141211_20260918`。导出时：

1. 按日期合并，重叠日期逐列比对（NaN 视为相等）。**只要有一个值不一致，就把日期、列、区间、取值写进 `snapshot_conflicts.csv`，拒绝打包**，不挑任何一个。
2. 用库里 `daily_price` 的每一行真实行情（`is_suspended = False`）核对合并结果：日期必须在，收盘价必须一模一样。缺的写进 `snapshot_coverage_gaps.csv`，同样拒绝打包。
3. 如果某个区间本身就等于合并结果，原样拷贝它的字节；否则写合并后的帧。

实测结果：

- 81 只股票，重叠部分**没有任何一个值不一致**；`20141211` 那个文件都是 `19901219` 那个文件的子集。
- 229,334 行行情的覆盖核对没有缺口。
- 快照里放的就是 `19901219` 那一份的原始字节。它也正是库当时实际用到的那一份：回灌时向数据源请求的区间从日历首日 1990-12-19 开始。

### 2.3 大小：15.7 MB，比规划估计的 5–10 MB 大

- 日线保留了原始文件的全部列（open/high/low/amount/turnover 等）和上市以来的全部历史。
- 规划里的 3.4 MB 是只算 2015 年以后、只留用到的几列的估计。
- 裁掉早年历史会改变回放的输入：回灌把窗口起点之前的历史当作上下文，用来给跨越起点的停牌延续最后真实收盘价。为了保证逐位相同，这里不裁。
- 16 MB 对 Release 附件不成问题。

## 3. 下载

- **默认匿名**：`https://github.com/<repo>/releases/download/<tag>/<asset>`。
- **环境变量里有 `GITHUB_TOKEN` 时**（只用于私有期间测试，§1 第 11 条）：走 GitHub API。先按 tag 查到附件，再带 `Accept: application/octet-stream` 下载。
  - token 用 `add_unredirected_header` 加：API 会重定向到存储域名，urllib 在重定向时不会把这类 header 带过去，所以 token 只发给 `api.github.com`。这一点有测试守着。
- **校验**：下载文件的 sha256 必须等于 `config.demo.yaml` 里记录的值。不等就改名为 `.bad` 并报错，不解压。
- **解压前逐个检查成员**：只允许纯文件名，不能带路径、不能有 `..`；必须都列在清单里；sha256 必须相符。全部通过后，才整体替换 `data/snapshot/`。
- 下载好的 zip 留在 `data/release/`。下次运行时，如果这里已有 sha256 相符的文件，就不再下载。

## 4. demo

- **耗时实测**（本机，akshare 被设为不可 import）：

  | 步骤 | 秒 | 状态 |
  |---|---|---|
  | backfill | 15.4 | ok |
  | risk | 19.7 | ok 497 |
  | pnl | 1.0 | ok 496 |
  | backtest | 0.2 | warming_up 249，ok 247 |
  | attribution | 33.5 | ok 496 |
  | stress | 97.1 | ok 497 |
  | limits | 13.7 | 按天汇总的最差状态：warn 239，ok 211，breach 47 |
  | **合计** | **181** | |

  不到规划里约 5 分钟的线，所以 **demo 不缩短持仓期**，用的是全量 497 天。docs 里的数字都能直接在 demo 库上复现，回归测试也不需要另建全量库。
- `data/demo.db` 已存在时拒绝运行，加 `--overwrite` 才删掉重建。
- **默认配置**：`demo` 和 `fetch-snapshot` 只对快照有意义，不给 `--config` 时默认用 `config.demo.yaml`（设了 `MRISK_CONFIG` 时用它）。其余命令仍默认 `config.yaml`，所以看板要写 `--config config.demo.yaml`。demo 结束时会打印这条命令。

## 5. 回放核对

`mrisk diff-db data/mrisk.db data/demo.db`：

- **范围**：schema 里全部 17 张表，结果表和参考表都在内。
- **匹配**：按主键匹配，主键里的运行元数据不算。
- **排除的列**：`run_id`、`created_at`、`ingested_at`。`load_type` 照常比对，两边都是 `backfill`。
- **比较方式**：浮点数按值精确相等，NaN 等于 NaN。JSON 解析后比较，不受键顺序影响。
- **dq_result 的特殊情况**：真实库在 09-19 留了三次回灌的 DQ 结果，第一次有 14 条 fail，后两次全过。门禁和看板读的都是每个键最新的那次，所以只比最新一次。旧运行的 68,352 行计入 `superseded_a`，不参与比对。

结果：**17 张表全部逐位相同。** 差异行 0，单边多出的行 0，最大差异 0。

| 表 | 行数 |
|---|---|
| daily_price | 229,334 |
| position | 19,880 |
| risk_result | 48,706 |
| pnl_hypo | 24,304 |
| backtest_result | 988 |
| var_attribution | 610,890 |
| stress_result | 511,413 |
| limit_check | 2,485 |
| dq_result（最新一次） | 34,176 |

其余参考表（日历、证券、映射、情景、限额规则、DQ 规则等）也相同。

## 6. 看板启动

`mrisk dashboard` 同时做两件事，所以安全默认不取决于从哪个目录启动：

- 以配置文件所在目录（即仓库根目录，`.streamlit/config.toml` 所在的地方）为工作目录；
- 显式传 `--server.address localhost` 和 `--browser.gatherUsageStats false`。

局域网访问加 `--address 0.0.0.0`。实测：从别的目录启动后，只监听 `127.0.0.1`，健康检查通过。这个命令不建库、不连库，库由看板自己以只读方式打开。

## 7. 怎么验证的

- `tests/test_snapshot.py`（15 个）：
  - 区间合并：重叠一致；不一致时逐值列出；NaN。
  - 覆盖核对。
  - 导出：清单、确定性；遇到冲突、缺口、缺文件时拒绝打包。
  - 解压：拒绝不安全路径和被篡改的成员。
  - 下载：匿名路径；token 路径与不重定向；sha256 不符；已有文件不重复下载。
  - **端到端**：假世界经真的 `AkshareSource`（假的 akshare 模块，缓存里人为放了两个区间）回灌并跑完各步 → 导出 → 解压 → 在 akshare 不可 import 的情况下跑 demo → `compare_databases` 逐位相同。
- `tests/test_compare.py`（3 个）：差异计数与最大差异；忽略运行元数据和 JSON 键顺序；DQ 只取最新一次。
- `tests/test_cli.py`（4 个）：demo 配置与主配置只差 demo 项；看板命令的工作目录、地址与"不开库"；局域网地址；`diff-db` 的退出码。
- `tests/test_regression.py`（slow，3 个，库路径由 `MRISK_REGRESSION_DB` 给出）：
  - 497 个持仓日；
  - 2015 情景损失占市值最大 41.51%；
  - 限额 2,485 行，ok 2,055 / warn 383 / breach 47；L_STRESS_2015 为 ok 459 / warn 38。
  - 在 demo 库和真实库上都通过。
- 全部测试：250 个通过（不含 slow 和 network）。

## 8. 还没做 / 待确认

1. ~~Release 未上传~~ 2026-10-18 已上传并验证：tag `snapshot-20260918`，文件 `mrisk-snapshot-20260918.zip`，sha256 `e55a3f1a6a31b752d7daec208e8eaf3497074bb732c37836e46acfa5adbaf3f8`。在本机先移走本地那份 zip，再跑 `fetch-snapshot`：
   - 匿名：HTTP 404（仓库仍私有），报错并提示设 `GITHUB_TOKEN`，退出码 2；
   - 带 `GITHUB_TOKEN`（只在该条命令的环境变量里）：经 API 下载，sha256 相符，与本地导出的 zip 逐字节相同，201 个文件核对清单后解压；
   - 再跑一次：已有校验过的文件，不再下载；
   - 检查过 `data/`、仓库文件和日志，token 不在任何文件里。
2. **README 截图**：等用户放进 `docs/figures/`。
3. **干净克隆计时**（规划 §3.4）：在云端做。要等 Release 上传；仓库仍是私有期间，需要给云端 `GITHUB_TOKEN`。
4. **默认配置**：`demo` / `fetch-snapshot` 默认用 `config.demo.yaml`、看板要显式传 `--config`（§4），这是我定的，可以改。
