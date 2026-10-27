# 配置说明

> 全部配置项。默认值开箱即用，不改也能跑；改之前先看「改了有什么代价」那一列。

## 配置文件与插值

配置在 `config.yaml`，支持 `${VAR}` 与 `${VAR:-默认值}` 两种环境变量插值，所以 API Key 不需要写进文件。`.env` 由配置层自己加载（固定读项目根目录的 `.env`，不依赖工作目录）。

下面只列真正需要你改的项，其余保持默认即可。

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `app.name` | 见 `config.yaml` | 会写进系统提示词，是回答里的自称。改成你自己的公司名或部门名 |
| `app.data_dir` | `./data` | 索引、向量库、账号库、BM25 全在这里。换目录等于换一套独立知识库，可用来隔离测试环境 |
| `ingest.root` | `./demo-docs` | 知识库根目录。管理后台的「全部重新入库」会把它直接当作 `POST /ingest` 的 `path`，所以它**必须是存在且可读的目录**，否则那个按钮会失败。想临时入库别的目录，用 `python -m app.cli ingest <路径>` 或在接口里传 `path`，不必改这一项 |
| `embedding.provider` | `fastembed` | 本地 ONNX 推理。初始化失败会**自动降级为 hash 向量**（日志有 WARNING），此时召回质量不可用，务必先修这里 |
| `embedding.model` | `jinaai/jina-embeddings-v2-base-zh` | 768 维 / 8192 token。**换模型必须清空向量库重新入库**，并重测 `min_top_score`：维度和语义空间都变了，旧阈值不再成立 |
| `embedding.batch_size` | `64` | 调大快点但更吃内存；CPU 机器上 32~64 是安全区 |
| `embedding.cache` | `true` | 按段落哈希缓存向量。增量入库必开，重跑全库时能省掉绝大部分计算 |
| `vector_store.provider` | `chroma` | `memory` 仅供测试，进程退出数据即丢 |
| `llm.provider` / `base_url` / `model` | `openai_compat` / DeepSeek / `deepseek-v4-flash` | 任何 OpenAI 兼容端点都能用（DeepSeek、通义、混元、vLLM、One-API、Ollama 的 OpenAI 模式）。改成 `extractive` 即完全离线，代价是回答退化为摘录 |
| `llm.api_key` | `${LLM_API_KEY:-}` | 从环境变量取，不要写死 |
| `llm.max_tokens` | `2048` | 推理模型的「思考」和「答案」共用这个预算。给到 1200 以下时思考可能吃光预算、正文字段为空，引擎会判定失败并走摘录兜底。它只作用于生成答案：追问改写 / 纠错 / LLM 精排的预算在代码里各自写死（512 / 512 / 1024），同一个道理——给小了正文返回空，调用方只会静默回退，日志前缀分别是 `[rewrite] 改写未生效:` `[correct] 纠错未生效:` `[rerank] 重排未生效:` |
| `llm.temperature` | `0.1` | 制度问答要稳。调高会让措辞飘，数字和期限反而更容易被改写 |
| `retrieval.final_top_k` | `6` | 最终喂给模型的片段数。调大覆盖更全但更慢更贵，噪声也更多；调小容易漏掉关键条款 |
| `retrieval.max_per_doc` | `2` | 同一篇文档最多占几条候选。设为 0 关闭限流——一部长文档的碎片会把 top_k 全占满，答案只围绕一篇展开 |
| `retrieval.max_context_chars` | `6000` | 上下文总预算，按「去重后的实际块数」均分，不是按命中数均分。总需求不超预算时不截断 |
| `retrieval.min_top_score` | `0.30` | **拒答阈值**，比的是 `best_dense`（向量余弦），不是融合分。它只能挡**明显跨域**的问题（问天气、股票、写代码），挡不住「同域但文档没写」——后者靠生成层的 `NO_ANSWER` 判定兜底。设 0 表示关闭这道闸门。换语料或换 embedding 模型后必须重标，详见常见问题 |
| `retrieval.correct_trigger_score` | `0.42` | 低于此分先做错别字纠错重检，再决定是否拒答。必须高于 `min_top_score`，否则纠错分支永远走不到；设得过高会让大量正常问题都多打一次模型。0 表示关闭纠错重检 |
| `retrieval.use_parent_context` | `true` | 命中子块时换回父块喂模型。关掉后上下文更短更省 token，但答案容易缺前提条件 |
| `chunking.max_chars` / `overlap_chars` / `parent_max_chars` | `700` / `120` / `2200` | 子块宁小勿大（保精度），父块是喂模型的完整上下文上限。制度类文档改这三项前先跑 `eval/` |
| `conversation.max_turns` / `history_turns_for_llm` / `rewrite` | `6` / `3` / `true` | 保留几轮历史、真正送给模型几轮、是否开启追问改写。**标定 `min_top_score` 或跑召回评测时必须把它设成 `false`**：改写后的查询把主语补全了，`dense` 天然比原句高 0.15 以上，拿它标出来的阈值不可比 |
| `guard.redact_pii` / `require_citation` | `true` / `true` | 外发前脱敏；没有引用就降级为「未标注出处」提示。除非你在做对比实验，不要关 |

---

