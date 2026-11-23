"""图片 OCR（本地 ONNX，不出网）。

为什么单独一个模块：OCR 引擎加载模型要 1~2 秒、占几百 MB 内存，
必须做成进程内单例，不能每张图 new 一次。PDF 扫描页与独立图片共用同一套逻辑。

设计上的两个约束：
1. **不静默失败**：识别不出字时返回置信度 0 并留痕，由调用方决定是"跳过"还是"报错"，
   而不是返回空字符串让上层以为"这份文件没内容"。
2. **保留阅读顺序**：OCR 返回的是散乱文本框，必须先按行聚类再排序，
   否则一句话会被拆成几段，检索时语义断裂。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Tuple

log = logging.getLogger(__name__)

_ENGINE: Any = None
_ENGINE_FAILED = False

# 同一行的判定阈值（像素）。OCR 文本框有轻微倾斜，不能按 y 完全相等聚类。
_ROW_TOL = 12


def available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _engine():
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None or _ENGINE_FAILED:
        return _ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
        log.info("OCR 引擎就绪（rapidocr-onnxruntime）")
    except Exception as exc:  # noqa: BLE001
        _ENGINE_FAILED = True
        log.warning("OCR 引擎初始化失败，图片/扫描页将无法识别: %s", exc)
    return _ENGINE


def _unwrap(res: Any) -> List[Tuple[List[List[float]], str, float]]:
    """兼容 rapidocr 各版本返回：tuple(result, elapse) / 对象 / 裸 list。"""
    if isinstance(res, tuple):
        res = res[0]
    if res is None:
        return []
    if hasattr(res, "txts"):          # 1.4+ 的 RapidOCROutput
        boxes = getattr(res, "boxes", None) or []
        txts = getattr(res, "txts", None) or []
        scores = getattr(res, "scores", None) or []
        return [(list(b), t or "", float(s or 0))
                for b, t, s in zip(boxes, txts, scores)]
    out = []
    for item in res:
        try:
            box, text, score = item[0], item[1], item[2]
            if text and str(text).strip():
                out.append((list(box), str(text).strip(), float(score or 0)))
        except (IndexError, TypeError):
            continue
    return out


def _reading_order(items: List[Tuple[List[List[float]], str, float]]) -> List[List[Tuple]]:
    """把散乱文本框聚成「行」，行内按 x 排序。返回的是行的列表，不是拍平的列表。

    行高才是不变的参照：36px 字号的行高约 45px，用固定 12px 阈值会把
    「报销流程：填单 -> 主管审批」切成三段，检索时语义就断了。
    """
    if not items:
        return []
    def top(item):
        return min(p[1] for p in item[0])
    def left(item):
        return min(p[0] for p in item[0])
    def height(item):
        ys = [p[1] for p in item[0]]
        return max(ys) - min(ys)

    heights = sorted(height(it) for it in items)
    median_h = heights[len(heights) // 2] if heights else 0
    tol = max(_ROW_TOL, median_h * 0.6)

    rows: List[List[Tuple]] = []
    for it in sorted(items, key=top):
        if rows and abs(top(it) - top(rows[-1][0])) <= tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    return [sorted(row, key=left) for row in rows]


def _needs_space(prev: str, nxt: str) -> bool:
    """中文片段直接拼，英文单词之间补空格——否则「RMB 600」会变成「RMB600」。"""
    return bool(prev) and bool(nxt) and prev[-1].isascii() and prev[-1].isalnum() \
        and nxt[0].isascii() and nxt[0].isalnum()


def _join_row(row: List[Tuple]) -> str:
    out = ""
    for _, text, _ in row:
        if out and _needs_space(out, text):
            out += " "
        out += text
    return out.strip()


def ocr(img, *, label: str = "") -> Tuple[str, float, int]:
    """识别一张图，返回 (按阅读顺序拼接的文本, 平均置信度, 行数)。

    `img` 可以是文件路径、PIL Image、numpy 数组。
    识别不出任何文字时返回 ("", 0.0, 0)——调用方必须据此区分
    「图上没字」和「OCR 坏了」。
    """
    eng = _engine()
    if eng is None:
        return "", 0.0, 0
    try:
        res = eng(img)
    except Exception as exc:  # noqa: BLE001
        log.warning("OCR 失败%s: %s", f"（{label}）" if label else "", exc)
        return "", 0.0, 0

    items = _unwrap(res)
    if not items:
        return "", 0.0, 0
    rows = _reading_order(items)
    text = "\n".join(_join_row(r) for r in rows)
    conf = sum(s for _, _, s in items) / max(1, len(items))
    return text, round(conf, 4), len(rows)


def ocr_path(path: str | Path) -> Tuple[str, float, int]:
    return ocr(str(path), label=Path(path).name)
