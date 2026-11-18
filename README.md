# 茶叶拼配多表连接查询优化器

一个用 Python 写的多表连接查询优化器：输入一条 SQL，输出**连接顺序 + 物理执行计划 + 行数估算**，
并带一个可视化页面（查询编辑 / 连接图 / 计划树 / 行数估算对比 / 枚举过程）。

针对的业务背景：茶叶拼配数据分散在十几张表里（原料批次、产区、检测报告、库存、订单……），
分析师写的多表关联查询换个连接顺序可能从几十分钟变成几秒。

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 解析 &   │──▶│ 统计信息  │──▶│ 行数估算 │──▶│ 代价模型  │──▶│ 计划生成 │
│ 子查询改写│   │ 直方图/NDV│   │ 选择率   │   │ NL/Hash/ │   │ 左深树   │
└──────────┘   │ 多列统计  │   │ 来源标注 │   │ Merge    │   └────┬─────┘
               └───────────┘   └──────────┘   └──────────┘        │
                                                                  ▼
                                       ┌──────────────────────────────────┐
                                       │ 子集 DP（穷举）/ 贪心（超阈值）  │
                                       └──────────────┬───────────────────┘
                                                      ▼
                                 ┌────────────────────────────────────────┐
                                 │ SQLite 执行验证：MATERIALIZED CTE 链   │
                                 │ 强制连接顺序 · 每步估算 vs 实际行数     │
                                 └────────────────────────────────────────┘
```

## 快速开始

```bash
pip install flask sqlglot pytest    # 依赖（sqlglot 负责 SQL 解析）

# 1) 生成演示库（16 张表，含列间相关与数据倾斜，固定随机种子）
python3 -m teaopt.demo data/tea.db

# 2) 跑测试（含穷举对比 / SQLite 执行对比）
python3 -m pytest tests -q

# 3) 启动页面
python3 -m teaopt.webapp        # 或 flask --app teaopt.webapp run
# 打开 http://127.0.0.1:5000
```

页面首次启动会自动建库（`data/tea.db`），无需手动准备数据。

## 页面五块

| 区块 | 内容 |
| --- | --- |
| **1 查询编辑** | SQL 输入、示例查询、贪心阈值 / 算法 / 统计开关；快捷键 ⌘/Ctrl+Enter |
| **2 连接图** | 表节点（过滤后 / 基础行数）、边（连接选择率），外连接虚线、半连接点线；选中节点即最优顺序 |
| **3 计划树** | 每个 join 节点标注算法（seqscan / indexscan / nl / hash / merge），显示各算法候选代价以解释选择 |
| **4 行数估算对比** | 每个中间结果：估算行数、SQLite 实际行数、偏差倍数、偏差百分比；底部给出最优 / 最差顺序 / SQLite 自选顺序的实际耗时 |
| **5 枚举过程** | 本次查询的子问题数、转移数；链 / 星 / 环 / 团四种图形状的子问题数对比表；候选计划；与穷举所有顺序的对比徽标 |

另有「选择率来源」表：逐条列出谓词的选择率、来源（直方图 / 高频值 / NDV / 多列统计 / 默认值）与计算说明。

## 八项要求的实现位置

| # | 要求 | 实现 | 测试 |
| --- | --- | --- | --- |
| 1 | 每条连接谓词标注选择率来源 | `teaopt/estimate.py` 为每个 `Predicate` 写入 `source` + `note`；直方图 / 高频值(MCV) / NDV / 多列统计 / 默认值 | `tests/test_estimate.py::test_filter_selectivity_sources` 等 |
| 2 | 表多时自动退回贪心，阈值可配 | `OptimizerConfig.exhaustive_threshold`（默认 10），`Optimizer.optimize()`；页面上可实时改 | `tests/test_search.py::test_greedy_used_above_threshold` |
| 3 | 直方图 + NDV 估行，多列统计修正独立性 | `Histogram`、`ColumnStats`、`MultiColStats`；`Estimation._correct_filter_correlation` / `_edge_selectivity` | `tests/test_estimate.py::test_correlated_filters_are_corrected` |
| 4 | 代价模型区分 NL / Hash / Merge | `teaopt/cost.py`：三种算法的启动代价、总代价与选择；FT 外连接禁用 NL | `tests/test_cost.py`（9 项） |
| 5 | 外连接 / 半连接不能随意换序 | `teaopt/parser.py::_build_stages` 切分可重排阶段；WHERE 过滤可空侧时外连接退化；`_legal_order` 供穷举对比共用 | `tests/test_parser.py`、`tests/test_search.py::test_outer_join_nullable_side_stays_right` |
| 6 | 关联子查询改写成半连接并参与枚举 | `teaopt/rewrite.py`（EXISTS / NOT EXISTS / IN → semi / anti）；半连接作为可浮动原子参与 DP | `tests/test_search.py::test_semi_unit_is_placed_in_enumeration` |
| 7 | 每步估算 vs 实际行数偏差 | `teaopt/execute.py::measure_step_rows` 逐 CTE 前缀计数；`explain.compare_estimates` | `tests/test_acceptance.py::test_deviation_reported_for_each_step` |
| 8 | 链 / 星 / 环 / 团四种形状的子问题数 | `teaopt/explain.py::enumeration_table` 按连通子集与转移数计算 | `tests/test_explain.py`（10 项） |

## 核心设计

### 连接顺序枚举：子集 DP 与穷举等价

左深树 + 子集 DP（System-R 风格）。`DP[S]` 保存子集 `S` 的最便宜计划，按子集大小递推。
**关键前提**：子集的行数估算只依赖子集本身，不依赖连接顺序——

* 所有选择率在搜索前一次性算好，挂在谓词上；
* 中间结果行数由 `Estimator.subset_rows` 按**规范顺序**（过滤后行数升序、再按别名）传播 NDV 计算，
  因此同一子集的估算恒定；
* 连接代价只依赖两个子计划的成本 / 行数 / 结果大小。

满足这三条时，DP 与穷举所有合法顺序的结果**严格一致**。验收测试对 3–6 关系的多组查询
（链、星、交叉链，共 100+ 组合法排列）逐一对拍，DP 代价与暴力搜索误差 < 1e-9。

> 备注：上面是"合法顺序"的穷举——与 DP 受同样的连通性 / 外连接约束，
> 否则比较的是两种不同的计划空间。`teaopt/search.py::_legal_order` 定义这套约束。

### 行数估算：避免星形连接的经典低估

朴素公式 `|R|·|S|·sel` 在"事实表 + 多个维度表"上会严重低估：第二个维度表的 `1/ndv` 又把
已经除过的 NDV 除了一遍。本实现：

1. 单谓词选择率：等值 / 区间 / IN 分别用直方图、高频值(MCV)、NDV，附来源与说明；
2. 多列统计：`(origin_id, grade)`、`(channel, tier)` 等显式收集的联合 NDV / MCV，
   修正"两列相关时独立假设相乘"的偏差；
3. 连接行数：按规范顺序传播每一列过滤后的 NDV，新关系加入时用
   `rows / max(ndv_new, ndv_old)`，再把 NDV 截断到当前行数——这与 PostgreSQL 的
   `ndv * Π multiplicity` 思路一致，消除星形连接的系统性低估。

### 外连接 / 半连接合法换序

解析时把关系切成若干**阶段**，阶段之间必须按顺序求值：

* `LEFT JOIN` 的可空侧自成阶段，永远在保留侧之后（`RIGHT JOIN` 解析时归一化为 `LEFT JOIN`）；
* 外连接之后出现的内连接关系单独成阶段，跨阶段的内连接谓词作为"附加条件"使用；
* 若 `WHERE` 过滤了可空侧（`IS NULL` 除外），该外连接退化为内连接，重新获得换序自由；
* 半连接 / 反连接**不是阶段**：作为可浮动原子，在外引用全部可用后即可应用——
  所以子查询既参与了枚举，又不会破坏外连接的语义。

### 执行验证：在 SQLite 上强制连接顺序

SQLite 会自行重排内连接，直接按优化顺序写表名并不能验证计划。本实现把计划渲染成
**一串 `MATERIALIZED` CTE**，每个 CTE 正好对应一步连接，因此执行顺序就是计划顺序：

```sql
WITH s1 AS MATERIALIZED (SELECT o.* FROM origins o WHERE o.province = '福建'),
     s2 AS MATERIALIZED (SELECT s1.*, l.* FROM s1 JOIN lots l ON l.origin_id = s1.o__origin_id),
     s3 AS MATERIALIZED (SELECT s2.* FROM s2 WHERE EXISTS (...))
SELECT COUNT(*) FROM s3;
```

验证内容：结果与原始 SQL 逐行一致；每个 CTE 前缀用 `COUNT(*)` 得到**实际行数**，与估算对比；
最优 / 最差排列 / SQLite 自选顺序各跑一次计时。所有执行都带 `progress_handler` 超时保护，
病态查询返回"超时"而不是卡死页面。

## 演示数据

`teaopt/demo.py` 生成 16 张表（固定种子，可复现），刻意注入：

* **列间相关**：`lots.grade` 由 `origin_id` 决定；`reports.passed/score` 由批次的 `grade` 决定；
  `customers.channel → tier` 强相关；`origins.province → region`。这些列对的联合 NDV
  明显小于各自 NDV 之积，多列统计据此修正。
* **数据倾斜**：少数产区供应大部分批次（Zipf 分布）；`orders.status` 中 `created` 占绝大多数；
  少数检验员承担大部分检测。
* 子表外键接近均匀采样，避免多个子表连接时因同一处倾斜被平方放大——这会让演示查询
  本身变得不可执行，掩盖优化效果。
* 生产环境会有的连接键索引（`lots.origin_id`、`reports.lot_id` 等）已建好，
  这样"好的顺序"和"坏的顺序"在 SQLite 上的耗时差异才是真实的。

## 目录结构

```
teaopt/
  types.py       数据类：统计信息、Query、PlanNode、OrderResult、OptimizerConfig
  parser.py      SQL → Query：连接图、外连接阶段、派生表折叠
  rewrite.py     关联子查询 → 半连接 / 反连接
  stats.py       从 SQLite 采集行数、NDV、直方图、MCV、多列统计
  estimate.py    选择率与所有中间结果的行数
  cost.py        扫描代价 + NL / Hash / Merge 三种连接代价
  plan.py        左深物理计划构造（各搜索算法共用）
  search.py      子集 DP、贪心、合法顺序穷举参考实现
  execute.py     计划 → MATERIALIZED CTE 链；执行 / 计时 / 逐步行数
  validate.py    估算 vs 实际报告
  explain.py     计划序列化、四种图形状的枚举统计
  service.py     面向页面 / 测试的端到端服务
  webapp.py      Flask API
  demo.py        演示库生成器
web/
  templates/index.html
  static/{app.js,style.css}
tests/           114 项测试
```

## 已知边界

* 只支持单条 `SELECT`；`GROUP BY` / `HAVING` / `DISTINCT` 不参与连接顺序优化（在连接之后执行）。
* 只支持左深树（System-R 计划空间）；健壮地处理 bushiness 需要更复杂的代价比较。
* 派生表只折叠"单表 + 过滤"；含连接 / 聚合的派生表要求先展开。
* `RIGHT JOIN` 左侧多表时不支持（提示改写为 `LEFT JOIN`）。
* 直方图为等宽 64 桶；倾斜列的等值估算依赖 MCV 列表（取前 12 个高频值）。
* 执行验证用的 CTE 链会物化每一步，适合验证语义与中间结果规模，不等价于生产引擎的
  流水线执行；页面上的耗时对比用于说明"顺序影响中间结果大小"，而不是绝对性能基准。
