"""离线评估：召回质量 + 拒答准确率 + 延迟。

用法（先跑检索评估，调好参数再评估生成）：
  python -m eval.run_eval --ingest ./demo-docs --k 6
  python -m eval.run_eval --k 6 --answer

指标说明：
  Recall@k   检索结果里是否包含正确文档（知识库场景最关键：没召回就不可能答对）
  MRR        正确文档排在第几位的倒数均值，衡量排序质量
  CiteHit    回答里是否给出了引用编号
  Refusal    该拒答的问题是否真的拒答了（问知识库外的问题时不能硬编）
  P95        95 分位延迟，决定企微里"转圈"多久
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from app.config import Config, load_config
from app.rag.engine import RagEngine

ROOT = Path(__file__).resolve().parents[1]
SET_PATH = Path(__file__).with_name("eval_set.jsonl")

_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def cited_indices(text: str, flags: Dict | None = None) -> List[int]:
    """回答里「已引用」的编号 = 渲染出的 [n] ∪ flags.citations。

    两处取并集，是为了和界面的「已引用」标记同源：只数渲染出来的 [n] 会漏掉
    流式先给 flags 后补正文的情形；只数 flags 又会漏掉模型自己写的角标。
    同时这也让本指标的分母可以明确为「参与评估的可答题数」，不会再出现 >100%。
    """
    found = {int(m) for m in _CITE_RE.findall(text or "")}
    for c in (flags or {}).get("citations") or []:
        try:
            found.add(int(c))
        except (TypeError, ValueError):
            continue
    return sorted(found)



def load_set(path: Path) -> List[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", default=str(SET_PATH))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ingest", default=None, help="评估前先导入该目录（重置知识库）")
    ap.add_argument("--answer", action="store_true", help="同时评估生成（要配好模型）")
    ap.add_argument("--data-dir", default=None, help="覆盖 data_dir，避免污染正式库")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg.app.data_dir = args.data_dir
    engine = RagEngine(cfg)

    if args.ingest:
        rep = engine.ingest(args.ingest, reset=True)
        print(f"入库完成：{rep.docs_total} 篇 / {rep.chunks_total} chunks "
              f"（新增 {rep.added}，失败 {rep.failed}）")

    items = load_set(Path(args.set))
    ranks: List[int] = []
    cite_hits = 0
    cite_total = 0
    refusals_ok = 0
    refusal_total = 0
    keyword_ok = 0
    keyword_total = 0
    latencies: List[int] = []
    rows: List[dict] = []

    for it in items:
        q = it["question"]
        gold = it.get("gold_doc")
        t0 = time.perf_counter()
        hits, _ = engine.retrieve(q, top_k=args.k)
        paths = [h.chunk.path for h in hits]
        rank = 0
        if gold:
            for i, p in enumerate(paths, 1):
                if p.endswith(gold):
                    rank = i
                    break
            ranks.append(rank)

        row = {"question": q, "gold": gold, "rank": rank, "latency": 0}
        if args.answer:
            ans = engine.answer(q)
            row["latency"] = ans.latency_ms
            latencies.append(ans.latency_ms)
            if it.get("expect_no_answer"):
                refusal_total += 1
                ok = not ans.answered and not ans.sources
                refusals_ok += int(ok)
                row["refusal_ok"] = ok
            else:
                # 分母是「参与评估的可答题数」——和分子同一批题目，
                # 之前用 len(ranks)-refusal_total 会让分母少算负样本，
                # 分母被减到 11、分子仍是 13，算出 118.2% 这种不可能值。
                cite_total += 1
                cites = cited_indices(ans.text, ans.flags)
                cite_hits += int(bool(cites))
                row["citations"] = cites
                need = it.get("must_contain") or []
                if need:
                    keyword_total += 1
                    ok = all(n in ans.text for n in need)
                    keyword_ok += int(ok)
                    row["keyword_ok"] = ok
            row["answer"] = ans.text[:200]
        else:
            row["latency"] = int((time.perf_counter() - t0) * 1000)
            latencies.append(row["latency"])
        rows.append(row)

    n = len(items)
    recall = sum(1 for r in ranks if 0 < r <= args.k) / max(1, len(ranks))
    mrr = sum(1 / r for r in ranks if r > 0) / max(1, len(ranks))
    p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 3 else max(latencies)

    print("\n== 明细 ==")
    for r in rows:
        if r.get("refusal_ok") is not None:
            # 负样本：只有真的评过才给判定，否则明确写「未评测」。
            # 原来一律打 ✗，读者会把「没测」读成「失败」。
            flag = "拒答✓" if r["refusal_ok"] else "拒答✗"
        elif r["gold"] is None:
            flag = "—（未评测）"
        elif r["rank"] and r["rank"] <= args.k:
            flag = "✓"
        else:
            flag = "✗"
        print(f"{flag}  rank={r['rank'] or '-':<3} {r['question'][:32]:<34} {r['gold'] or '(无答案)'}")

    print("\n== 汇总 ==")
    print(f"样本数        : {n}")
    print(f"Recall@{args.k}      : {recall:.1%}  ({sum(1 for r in ranks if 0 < r <= args.k)}/{len(ranks)})")
    print(f"MRR           : {mrr:.3f}")
    if args.answer:
        print(f"引用覆盖率    : {cite_hits / max(1, cite_total):.1%}  ({cite_hits}/{cite_total})")
        if keyword_total:
            print(f"要点命中率    : {keyword_ok / keyword_total:.1%}")
        if refusal_total:
            print(f"拒答准确率    : {refusals_ok / refusal_total:.1%}")
    print(f"P95 延迟      : {p95:.0f} ms")
    engine.close()

    # 低于基线直接返回非 0，方便接 CI
    baseline_ok = recall >= 0.8
    if not baseline_ok:
        print("\n[!] Recall 低于 80% 基线，先别上线：检查分块参数、Embedding 模型与召回通道。")
    return 0 if baseline_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
