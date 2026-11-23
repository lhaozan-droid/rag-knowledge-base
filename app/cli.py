"""命令行入口：

  python -m app.cli ingest ./demo-docs
  python -m app.cli ask "年假有多少天"
  python -m app.cli search "差旅住宿标准" -k 5
  python -m app.cli stats
  python -m app.cli serve --port 8080
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:  # 可选：读取 .env
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from app.config import load_config
from app.rag.engine import RagEngine


def _setup_log(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S")


def _embedding_line(engine) -> str:
    """报告里的向量模型行。降级必须显式喊出来，不能只躺在日志里。"""
    desc = engine.embedder.describe()
    if getattr(engine.embedder, "name", "") == "hash":
        return f"{desc} —— 降级，召回基本不可用，原因见上方日志"
    return desc


def cmd_ingest(args) -> int:
    engine = RagEngine(load_config(args.config))
    # 报告第一行就回显真实路径：`ingest .`（把 data/、eval/ 一起扫）、
    # `ingest ./demo-docs`、`ingest ./demo-doc`（少打一个 s）三种情况
    # 在后面的统计行上几乎一样，路径打错的人会以为一切正常。
    try:
        root = str(Path(args.path).expanduser().resolve())
    except OSError:                      # 路径都不合法时交给 ingest 去报错
        root = args.path
    print(f"扫描目录：{root}")

    total = {"n": 0}
    # 进度行走 stderr，报告走 stdout：
    # 1) `ingest ./demo-docs > report.txt` 时进度不会混进报告；
    # 2) 进度用 \r 原地刷新，和 logging 抢同一行会把日志切碎——
    #    日志默认也走 stderr，两路交错时至少不再撕裂 stdio 的报告。
    # 非 TTY（管道 / CI）下不打印进度行，免得日志被刷屏。
    show_progress = sys.stderr.isatty()

    def progress(done: int, total_n: int, name: str) -> None:
        total["n"] = total_n
        if not show_progress:
            return
        sys.stderr.write(f"\r[{done}/{total_n}] {name[:70]:<70}")
        if done == total_n:
            # 最后一件：先把进度行擦掉再收工。后面 pipeline 还会打
            # 「BM25 索引重建完成」这类日志，不清的话它会糊在进度行尾巴上。
            sys.stderr.write("\r" + " " * 80 + "\r")
        sys.stderr.flush()

    report = engine.ingest(args.path, reset=args.reset, force=args.force,
                           progress=progress)
    print(f"向量模型：{_embedding_line(engine)}")
    print(f"新增 {report.added} / 更新 {report.updated} / 跳过 {report.skipped} "
          f"/ 失败 {report.failed}")
    print(f"本次写入 {report.chunks} 个 chunk；库内共 {report.docs_total} 篇文档、"
          f"{report.chunks_total} 个 chunk；耗时 {report.elapsed_s}s")
    # 这两类不算失败，但都意味着知识库里少了内容，必须显式提示
    if report.unsupported:
        print(f"\n有 {len(report.unsupported)} 个文件因格式不支持，未入库：", file=sys.stderr)
        for item in report.unsupported[:15]:
            print("  - " + item, file=sys.stderr)
        if len(report.unsupported) > 15:
            print(f"  … 另有 {len(report.unsupported) - 15} 个", file=sys.stderr)
    if report.empty_text:
        print(f"\n有 {len(report.empty_text)} 个文件读到了但没有可用文本：", file=sys.stderr)
        for item in report.empty_text[:15]:
            print("  - " + item, file=sys.stderr)
        if len(report.empty_text) > 15:
            print(f"  … 另有 {len(report.empty_text) - 15} 个", file=sys.stderr)
    for err in report.errors[:10]:
        print("  ! " + err, file=sys.stderr)
    engine.close()
    return 0 if report.failed == 0 else 1


def cmd_ask(args) -> int:
    engine = RagEngine(load_config(args.config))
    if args.conversation_id:
        # 支持一次会话内连续追问：同一 conversation_id 会带上历史
        for q in [args.query]:
            pass
    ans = engine.answer(args.query, conversation_id=args.conversation_id)
    print("\n" + ans.text + "\n")
    if args.json:
        print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"— 耗时 {ans.latency_ms}ms｜改写后查询：{ans.rewritten_query}")
        for s in ans.sources:
            loc = f"《{s.title}》" + (f" / {s.section}" if s.section else "")
            print(f"  [{s.index}] {loc}  ({s.path})  "
                  f"相关度={s.dense_score:.4f} 融合分={s.score:.4f}")
    engine.close()
    return 0


def cmd_search(args) -> int:
    engine = RagEngine(load_config(args.config))
    hits, best = engine.retrieve(args.query, top_k=args.k)
    print(f"best_dense={best:.4f}  命中 {len(hits)} 条")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h.chunk.title} / {h.chunk.section}] "
              f"rrf={h.score:.4f} dense={h.dense_score:.3f} bm25={h.sparse_score:.2f}")
        print("   " + h.chunk.text[:120].replace("\n", " ") + "…")
    engine.close()
    return 0


def cmd_stats(args) -> int:
    engine = RagEngine(load_config(args.config))
    print(json.dumps(engine.stats(), ensure_ascii=False, indent=2))
    engine.close()
    return 0


def cmd_unanswered(args) -> int:
    engine = RagEngine(load_config(args.config))
    rows = engine.registry.list_unanswered(args.limit)
    if not rows:
        print("暂无未答问题。")
    for r in rows:
        print(f"- {r['query']}  (best_dense={r['top_score']})")
    engine.close()
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("app.api.server:app", host=args.host, port=args.port,
                reload=args.reload, log_level=args.log_level)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-knowledge-base", description="私有化企业知识库 RAG 引擎")
    p.add_argument("--config", default=None, help="配置文件路径（默认 config.yaml）")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="导入/更新知识库")
    pi.add_argument("path", help="文档目录或单个文件")
    pi.add_argument("--reset", action="store_true", help="先清空知识库再导入")
    pi.add_argument("--force", action="store_true", help="忽略 mtime，强制重新解析")
    pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("ask", help="提问")
    pa.add_argument("query")
    pa.add_argument("-c", "--conversation-id", default=None, help="会话 ID（多轮追问用）")
    pa.add_argument("--json", action="store_true", help="输出完整 JSON")
    pa.set_defaults(func=cmd_ask)

    ps = sub.add_parser("search", help="只看检索结果，不调模型（调参必备）")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=6)
    ps.set_defaults(func=cmd_search)

    sub.add_parser("stats", help="查看知识库与配置状态").set_defaults(func=cmd_stats)

    pu = sub.add_parser("unanswered", help="查看答不上来的问题（培训盲区）")
    pu.add_argument("--limit", type=int, default=30)
    pu.set_defaults(func=cmd_unanswered)

    pv = sub.add_parser("serve", help="启动 HTTP 服务 + 调试台")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8080)
    pv.add_argument("--reload", action="store_true")
    pv.add_argument("--log-level", default="info")
    pv.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    _setup_log(cfg.app.log_level)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
