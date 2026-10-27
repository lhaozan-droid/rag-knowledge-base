# 命令行与 HTTP 接口

> 不想用网页界面时的两种用法：命令行，或者把你的系统接进来。

## 命令行

```
python -m app.cli ingest <目录或单文件> [--reset] [--force]   导入 / 增量更新
python -m app.cli ask "问题" [-c 会话ID] [--json]             提问并打印出处
python -m app.cli search "问题" [-k 6]                        只看检索结果，不调模型（调参必备）
python -m app.cli stats                                       文档数 / chunk 数 / 向量数 / BM25 条数
python -m app.cli unanswered [--limit 30]                     被拒答的问题清单（知识盲区）
python -m app.cli serve [--host 127.0.0.1] [--port 8080] [--reload]   启动 HTTP 服务
```

`--config` 可放在子命令前指定另一个配置文件：`python -m app.cli --config ./config.test.yaml ingest ./demo-docs`。

`search` 的输出是调参的主要依据：

```
best_dense=0.6313  命中 6 条
1. [员工手册（2026 版） / 第三章 年假与休假 > 3.2 年假申请流程] rrf=0.0325 dense=0.568 bm25=17.58
   年假需提前 3 个工作日在 OA 系统「请假申请」中提交，由直属主管审批；连续休假超过 5 天的，需部门负责人二次审批。年假期间工资照常发放。…
2. [员工手册（2026 版） / 第三章 年假与休假 > 3.1 年假天数] rrf=0.0320 dense=0.631 bm25=8.17
   年休假按累计工作年限计算： - 累计工作满 1 年不满 3 年：每年 5 天 - 累计工作满 3 年不满 5 年：每年 10 天 - 累计工作满 5 年不满 10 年：每年 15 天 - 累计工作满 10 年以上：每年 20 天 年假以自然…
3. [考勤与请假管理办法 / 3 审批流程 > 3.1 审批权限] rrf=0.0304 dense=0.343 bm25=11.55
4. [考勤与请假管理办法 / 2 请假类型与规则 > 2.2 事假] rrf=0.0292 dense=0.452 bm25=3.04
5. [差旅与费用报销制度 / 1 出差申请 > 1.2 审批权限] rrf=0.0270 dense=0.289 bm25=2.43
6. [差旅与费用报销制度 / 3 报销流程与时限 > 3.2 报销步骤] rrf=0.0269 dense=0.231 bm25=5.11
```

（`demo-docs/` + 默认 `fastembed` 模型的真实输出。每条命中下面还有一行片段摘要，这里只保留了前两条。）

注意第 1 名和第 2 名的 `dense` 与 `bm25` 排名正好相反：BM25 被「年假」的字面频次拉到 17.58，向量路更认语义（`dense=0.631` 是全场最高）。两条都会进上下文（父块补全后按块去重），所以答案不受这个排序影响；但这正是需要两路的原因——只看 BM25 容易答成「申请流程」，只看向量又会漏掉「提前 3 个工作日」这类字面条款。

`dense` 是向量余弦相似度，`bm25` 是关键词得分，`rrf` 是两者的排名融合分——**`rrf` 不是相似度，不要拿它跟 0.6、0.8 比**：同一行里 `dense=0.568` 与 `rrf=0.0325` 差两个数量级，就是这个意思（原因见技术要点）。拒答阈值比的是 `best_dense`。

---


## HTTP 接口

单机内网调试用，默认只监听本机。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 问答页（无有效会话时返回登录页） |
| POST | `/login` | 账号密码登录，成功后种 7 天有效的 HttpOnly Cookie |
| GET | `/health` | 服务与知识库状态，**需要登录**（无 Cookie / Bearer Token 返回 401） |
| POST | `/chat` | 同步问答，返回 `text` + `sources` + `conclusion/rules/notes` |
| POST | `/chat/stream` | SSE 流式问答，事件序列 `meta → sources → delta* → done` |
| POST | `/search` | 只检索不生成 |
| GET | `/conversations`、`/conversations/{id}` | 当前账号的历史会话 |
| GET | `/unanswered` | 未答问题清单（需管理员） |
| — | `/admin/*` | 账号管理、文档增删、分类树、入库报告、调用量统计（需管理员） |

`/health` 返回的是业务状态而不是裸探针，因此和其它业务接口一样需要登录态，用外部监控采集请带上 Token；如果只是想在 CI 或部署脚本里判断进程活没活，直接用 `curl -sf http://127.0.0.1:8080/` 更省事。

`/chat` 的 `sources[]` 里有两个分数，别混用：**`dense_score` 是向量余弦相似度**（0~1，语义稳定，界面上的「相关度」用它）；**`score` 是 RRF 排名融合分**（`rrf_k=60`、两路权重都是 1.0 时上界约 2/61≈0.033），一旦把 `rerank.provider` 开成 `cross_encoder`，`score` 又会变成 rerank 分值、量纲再变一次，所以它只适合排查排序，不适合当相似度展示、也不要拿去跟 0.6 比。

`/search` 返回的每个 `hit` 是 `score` / `dense` / `bm25` 三个并列字段（对应 CLI `search` 打印的 `rrf=` / `dense=` / `bm25=`）。这里的 **`dense` 与 `/chat` sources 的 `dense_score` 是同一个指标，只是字段名不同**：`/search` 是排查用接口，三个短名成组出现、语境自明；`/chat` 的 `sources` 是对外契约，字段名要能独立看懂，所以叫 `dense_score`。

---

