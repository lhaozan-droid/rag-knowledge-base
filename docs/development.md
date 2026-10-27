# 目录结构与开发

> 想改代码、加功能、或者跑测试时看这份。

## 目录结构

```
rag-knowledge-base/
├── app/
│   ├── cli.py            命令行入口（ingest / ask / search / stats / unanswered / serve）
│   ├── config.py         YAML 加载、${ENV} 插值、类型化 dataclass
│   ├── models.py         Document / Chunk / RetrievedChunk / Source / Answer / IngestReport
│   ├── embed.py          本地向量化（fastembed / ollama / sentence-transformers）+ 向量缓存 + hash 降级
│   ├── auth.py           账号库（bcrypt 存密码哈希）+ 自签 HMAC token（签名不依赖第三方库）
│   ├── metrics.py        API 调用量统计，内存计数 + 定期落盘
│   ├── api/
│   │   ├── server.py     FastAPI 全部路由：登录、问答、流式、管理后台
│   │   ├── login.html    登录页
│   │   ├── console.html  问答页（单文件、无前端构建，图标内联在页面里）
│   │   └── admin.html    管理后台
│   ├── ingest/
│   │   ├── loaders.py    各格式解析 → 统一成「带 # 标题的文本」
│   │   ├── chunker.py    结构感知切分：标题 → 父块 → 子块
│   │   ├── pipeline.py   增量入库：mtime 快路径、内容哈希去重、批量写向量、重建 BM25
│   │   └── ocr.py        图片与扫描页 OCR（本地 ONNX）
│   ├── llm/
│   │   ├── client.py     OpenAI 兼容客户端 + 离线摘录降级（ExtractiveLLM）
│   │   └── prompts.py    系统提示词、答案判定与追问改写用的提示词
│   ├── rag/
│   │   ├── engine.py     把检索、生成、校验、落库串成一条链路（同步与流式共用）
│   │   ├── rewrite.py    追问改写（只对含指代词或「那…呢」的追问触发，带确定性护栏）
│   │   ├── correct.py    错别字纠错（确定性词表 + 阈值触发模型纠错）
│   │   ├── glossary.py   领域同义词扩展，只作用于 BM25
│   │   ├── guard.py      PII 脱敏
│   │   ├── verdict.py    答案判定：拒答 / 引用缺失 / 正常
│   │   └── parse.py      把自由文本回答拆成 结论 / 分条 / 补充说明
│   ├── retrieval/
│   │   ├── bm25.py       jieba 分词 + BM25，索引持久化为 pkl
│   │   ├── fusion.py     RRF 融合
│   │   └── rerank.py     重排：none / cross_encoder / llm
│   └── store/
│       ├── base.py       向量库抽象（换库只需实现 5 个方法）
│       ├── chroma_store.py
│       ├── memory_store.py
│       └── registry.py   SQLite 单一事实源：docs / chunks / turns / unanswered / categories
├── demo-docs/            5 篇示例制度文档，用于跑通流程
├── eval/                 黄金测试集与评测脚本（Recall@k / MRR / 拒答准确率 / P95）
├── tests/                pytest 单元与集成测试
│   └── conftest.py       测试夹具：make_cfg / RecordingLLM / engine（与根目录 conftest.py 不是一个文件、职责也不同）
├── scripts/              启动与入库脚本（Windows / macOS / Linux）
├── data/                 运行时生成：向量库、registry.db、账号库、BM25、日志。不入库
├── config.yaml
├── .env.example          环境变量样例（复制成 .env 再填 Key）
├── .gitignore
├── LICENSE               MIT
├── conftest.py           pytest 根级配置（把项目根加入 sys.path）
├── cloudflared-config.example.yml   Cloudflare Tunnel 配置样例（全占位符）
└── requirements.txt
```

---


## 开发与测试

```bash
pytest                       # 单元 + 集成测试，用 hash 向量和 memory 后端，不下载模型、不调外部 API
python -m eval.run_eval --ingest ./demo-docs --k 6          # 只评估检索
python -m eval.run_eval --k 6 --answer                       # 同时评估生成（需要配好模型）
python -m eval.run_eval --k 6 --data-dir ./data-eval         # 用独立数据目录，不污染正式库
python -m eval.compare_embedder                              # 对比不同 embedding 的语义召回能力
```

评测脚本在 Recall 低于 80% 时返回非 0，可以直接接 CI。测试用 `provider: hash` + `vector_store: memory`，所以不依赖网络也不依赖模型文件——这也意味着测试通过只说明链路正确，不代表检索质量。

---


---

## 辅助脚本

`scripts/` 下四个脚本，免去手敲命令：

| 脚本 | 作用 |
|---|---|
| `ingest_demo.sh` / `ingest_demo.bat` | 入库 `demo-docs/` |
| `run_server.sh` / `run_server.bat` | 启动服务 |

Windows 用 `.bat`（双击即可），macOS / Linux 用 `.sh`（Git Bash 里也能跑）。脚本会先 `cd` 回项目根目录，所以从任意目录调用都可以。

**解释器探测顺序**：`.venv/bin/python` → `.venv/Scripts/python.exe` → `python3` → `python`。选中后先自检 `chromadb` / `fastembed` 是否装在这个解释器里——没装就明确报错让你先激活虚拟环境，而不是抛一堆 ImportError 让你以为依赖装错了。

**依赖装在别的环境里**（不是项目内的 `.venv`）时，用 `RAG_PYTHON` 直接指定，优先级高于自动探测：

```bash
RAG_PYTHON=/path/to/python ./scripts/run_server.sh
```

```bat
set RAG_PYTHON=C:/path/to/python.exe
scripts\run_server.bat
```
