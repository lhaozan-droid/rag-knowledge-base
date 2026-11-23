"""结构感知分块 + 父子块（Parent-Child）。

为什么不用固定窗口切分（比如每 500 字一刀）：
1. 会把「请假流程」的步骤从「适用范围」那里切断，检索到的片段缺少前提条件；
2. 引用溯源只能给出「文档第 8 段」，无法给出「3.2 请假流程」，用户在培训场景里不认；
3. 上千份文档里，固定窗口会产生大量无意义碎片，直接拉低召回精度。

这里的做法：
- 以 # 标题为天然边界 → 保留章节语义，引用能精确到小节；
- 大章节再按段落/句子切成父块（≤parent_max_chars）；
- 父块内切成带重叠的子块（≤max_chars）参与向量检索；
- 命中子块后可以把父块一起喂给 LLM，兼顾「精确召回」和「上下文完整」。
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Tuple

from app.config import ChunkingConfig
from app.models import Chunk, Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENT_RE = re.compile(r"[^。！？；!?;\n]+[。！？；!?;\n]*")
PAGE_BREAK = "\f"
# 文档级 meta → chunk meta 的白名单（不要整体拷贝，避免把 pages/slides 这类
# 与检索无关的字段写进向量库 metadata）
# 需要带到每个 chunk 上的文档级元数据。
# content_type/source_url 决定"能不能展开讲"（正文不完整的资料只有标题与链接，不许编内容），
# published_at 用于回答时标注版本时效，所以必须随之进入检索结果。
_PASS_META = ("modality", "ocr", "ocr_confidence", "content_type", "source_url",
              "published_at", "category", "status", "article_id", "source")


def _hash(*parts: object, n: int = 20) -> str:
    h = hashlib.sha1("\u0001".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()[:n]


def _paragraphs(body: str) -> List[str]:
    """段落优先；PDF 抽出的文本常没有空行，退化到按行切。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if len(paras) <= 1:
        lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
        if len(lines) > 1:
            return lines
    return paras or ([body.strip()] if body.strip() else [])


def _units(text: str, max_chars: int) -> List[str]:
    """切成句子级单元；超长句强行截断，避免单句撑爆 chunk。"""
    out: List[str] = []
    for m in _SENT_RE.finditer(text):
        s = m.group(0)
        if not s.strip():
            continue
        if len(s) <= max_chars:
            out.append(s)
        else:
            out.extend(s[i:i + max_chars] for i in range(0, len(s), max_chars))
    return out


def _tail_overlap(text: str, overlap: int) -> str:
    """取尾部 overlap 字符作为下一块的开头，尽量对齐到完整的句子。"""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    # 若窗口里有完整句子，从第一个句末标点之后开始，避免重叠部分以半句话开头
    for i, ch in enumerate(tail):
        if ch in "。！？；!?;\n":
            rest = tail[i + 1:]
            if rest.strip():
                return rest
            break
    # 窗口本身就是一句完整的话（标点在末尾），整段复用
    return tail


def pack_units(text: str, max_chars: int, overlap: int) -> List[str]:
    """把一段文本按句装进 ≤max_chars 的块，块间保留 overlap 字符的重叠。"""
    overlap = max(0, min(overlap, max_chars // 2))  # 防止 overlap≥max_chars 导致死循环
    units = _units(text, max_chars)
    chunks: List[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + len(u) > max_chars:
            chunks.append(cur.strip())
            cur = _tail_overlap(cur, overlap) + u
            # 极端情况：重叠 + 新单元仍超限，硬切
            while len(cur) > max_chars:
                chunks.append(cur[:max_chars].strip())
                cur = cur[max_chars - overlap:] if overlap else cur[max_chars:]
        else:
            cur += u
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c.strip()]


def _sections(text: str) -> List[dict]:
    """按标题切章节，维护标题层级路径（如 A > B > C）。"""
    page = 1
    stack: List[Tuple[int, str]] = []
    sections: List[dict] = []
    cur = {"title": "", "level": 0, "path": "", "lines": [], "page": 1}

    for raw_line in text.split("\n"):
        page += raw_line.count(PAGE_BREAK)
        line = raw_line.replace(PAGE_BREAK, "")
        m = HEADING_RE.match(line)
        if m:
            if cur["lines"] or not sections:
                sections.append(cur)
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur = {"title": title, "level": level,
                   "path": " > ".join(t for _, t in stack),
                   "lines": [], "page": page}
        else:
            cur["lines"].append(line)
    sections.append(cur)
    return sections


def chunk_document(doc: Document, text: str, cfg: ChunkingConfig) -> List[Chunk]:
    """产出一个文档的全部父块与子块（is_parent 区分）。"""
    if not text.strip():
        return []

    # 从文档级 meta 带到 chunk 的白名单：检索时要靠它给 OCR/ASR 内容降权
    base_meta = {k: doc.meta[k] for k in _PASS_META if k in doc.meta}

    chunks: List[Chunk] = []
    ordinal = 0
    sections = _sections(text)

    for s_idx, sec in enumerate(sections):
        body = "\n".join(sec["lines"]).strip()
        sec_title = sec["title"] or doc.title
        sec_path = sec["path"] or doc.title
        # 去掉与文档标题重复的第一级（h1 通常就是文档名），让引用显示更干净
        if sec_path.startswith(doc.title + " > "):
            sec_path = sec_path[len(doc.title) + 3:]
        # 整行都是标题时 body 会为空。真标题不会写成一两百字的长句，出现长「标题」
        # 说明源文档把正文段落误标成了标题（富文本转 Markdown 的常见产物）。
        # 直接丢弃会让整篇文档 0 分块、连标题都搜不到，所以把长标题回收成正文。
        if not body:
            if len(sec_title) >= cfg.min_chars:
                body = sec_title
            else:
                continue

        # 1) 章节 → 父块（按段落聚合，不超过 parent_max_chars）
        parents: List[Tuple[str, int]] = []
        buf, buf_page, page = "", sec["page"], sec["page"]
        for para in _paragraphs(body):
            page += para.count(PAGE_BREAK)
            para = para.replace(PAGE_BREAK, "").strip()
            if not para:
                continue
            if buf and len(buf) + len(para) + 1 > cfg.parent_max_chars:
                parents.append((buf.strip(), buf_page))
                buf, buf_page = "", page
            if not buf:
                buf_page = page
            buf = (buf + "\n" + para).strip() if buf else para
        if buf.strip():
            parents.append((buf.strip(), buf_page))

        for p_idx, (p_text, p_page) in enumerate(parents):
            # s_idx 必须参与：同一篇文档里出现重名小标题（多个「附件」「说明」）时，
            # sec_path 相同、p_idx 又从 0 开始，parent_id 会撞车，
            # 子块 chunk_id 跟着重复，写向量库直接报 DuplicateIDError。
            parent_id = _hash(doc.doc_id, "p", s_idx, sec_path, p_idx)
            ordinal += 1
            chunks.append(Chunk(
                chunk_id=parent_id, doc_id=doc.doc_id, text=p_text, ordinal=ordinal,
                title=doc.title, section=sec_path, path=doc.path,
                parent_id="", is_parent=1, meta={**base_meta, "page": p_page}))

            # 2) 父块 → 子块（带重叠，参与向量/BM25 检索）
            pieces = pack_units(p_text, cfg.max_chars, cfg.overlap_chars) \
                if len(p_text) > cfg.max_chars else [p_text]

            # 末尾过短的碎片合并到前一块，避免产生"孤字块"
            if len(pieces) > 1 and len(pieces[-1]) < cfg.min_chars:
                pieces[-2] = (pieces[-2] + pieces[-1]).strip()
                pieces.pop()

            for c_idx, piece in enumerate(pieces):
                ordinal += 1
                chunks.append(Chunk(
                    chunk_id=_hash(doc.doc_id, "c", parent_id, c_idx, piece[:64]),
                    doc_id=doc.doc_id, text=piece, ordinal=ordinal, title=doc.title,
                    section=sec_path, path=doc.path, parent_id=parent_id,
                    is_parent=0, meta={**base_meta, "page": p_page}))

    # 兜底：整篇一个分块都没产出（每行都短标题、正文极碎），至少留一条，
    # 否则文档等于从知识库里消失——连「有这篇文档」都检索不到。
    if not chunks:
        flat = " ".join(ln.strip() for ln in text.split("\n") if ln.strip())
        flat = re.sub(r"^#{1,6}\s*", "", flat).strip()
        if not flat:
            return []
        parent_id = _hash(doc.doc_id, "p", "fallback")
        chunks.append(Chunk(
            chunk_id=parent_id, doc_id=doc.doc_id, text=flat, ordinal=1,
            title=doc.title, section=doc.title, path=doc.path,
            parent_id="", is_parent=1, meta={**base_meta, "page": 1}))
        for c_idx, piece in enumerate(pack_units(flat, cfg.max_chars, cfg.overlap_chars)):
            chunks.append(Chunk(
                chunk_id=_hash(doc.doc_id, "c", parent_id, c_idx, piece[:64]),
                doc_id=doc.doc_id, text=piece, ordinal=c_idx + 2, title=doc.title,
                section=doc.title, path=doc.path, parent_id=parent_id,
                is_parent=0, meta={**base_meta, "page": 1}))

    return chunks


def embedding_text(chunk: Chunk) -> str:
    """喂给 Embedding 的文本带上「文档标题 + 章节」，能显著提升短查询的召回。"""
    prefix = f"{chunk.title}"
    if chunk.section and chunk.section != chunk.title:
        prefix += f" / {chunk.section}"
    return f"{prefix}\n{chunk.text}"
