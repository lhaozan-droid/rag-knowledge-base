"""多格式入库的回归测试：静默失败、PDF 清洗、图片 OCR。

这些测试要守住的不是"功能有没有"，而是**失败有没有被看见**——
原实现里不支持的文件直接消失、扫描件被当成空文档，报告上一切正常。
"""
import pytest

from app.ingest.loaders import load_file, reject_reason, scan_dir

pymupdf = pytest.importorskip("pymupdf", reason="需要 pymupdf 生成 PDF 样例")
PIL = pytest.importorskip("PIL")


def _font(size=36):
    from PIL import ImageFont
    for name in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _text_image(path, lines, size=(900, 300)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    f = _font()
    y = 40
    for line in lines:
        d.text((40, y), line, fill="black", font=f)
        y += 90
    img.save(path)
    return path


def _two_col_pdf(path, pages=4):
    doc = pymupdf.open()
    for pno in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 40), "星海科技 · 内部资料", fontname="china-s", fontsize=9)
        page.insert_text((280, 810), f"- {pno + 1} -", fontname="helv", fontsize=9)
        page.insert_textbox(pymupdf.Rect(50, 100, 270, 400),
                            f"第 {pno + 1} 章 考勤\n左栏第一条 应按时打卡。",
                            fontname="china-s", fontsize=10)
        page.insert_textbox(pymupdf.Rect(320, 100, 545, 400),
                            "右栏第四条 加班需提前申请。",
                            fontname="china-s", fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def _scanned_pdf(path, image_path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=str(image_path))
    doc.save(str(path))
    doc.close()
    return path


# ---------------- 失败可见 ----------------

def test_reject_reason_is_explicit():
    from pathlib import Path
    assert reject_reason(Path("a.mp4")) and "音视频" in reject_reason(Path("a.mp4"))
    assert "旧版" in reject_reason(Path("a.doc"))
    assert "压缩包" in reject_reason(Path("a.zip"))
    assert "无扩展名" in reject_reason(Path("README"))
    assert reject_reason(Path("a.md")) is None
    assert reject_reason(Path("a.png")) is None


def test_scan_dir_reports_every_unsupported_file(tmp_path):
    (tmp_path / "制度.md").write_text("# 制度\n年假 15 天。", encoding="utf-8")
    (tmp_path / "录像.mp4").write_bytes(b"\x00")
    (tmp_path / "旧版.doc").write_bytes(b"\x00")
    (tmp_path / "资料.zip").write_bytes(b"\x00")

    files, rejected = scan_dir(tmp_path)

    assert [p.name for p in files] == ["制度.md"]
    names = {p.name for p, _ in rejected}
    assert names == {"录像.mp4", "旧版.doc", "资料.zip"}, "被拒绝的文件必须一个不漏地报出来"
    assert all(reason for _, reason in rejected)


def test_ingest_report_separates_unsupported_and_empty(tmp_path):
    from app.rag.engine import RagEngine
    from tests.conftest import make_cfg

    (tmp_path / "kb").mkdir()
    kb = tmp_path / "kb"
    (kb / "制度.md").write_text("# 制度\n\n年假 15 天。", encoding="utf-8")
    (kb / "录像.mp4").write_bytes(b"\x00")
    _text_image(kb / "空白图.png", [])  # 纯白图，OCR 识别不到字

    eng = RagEngine(make_cfg(tmp_path))
    report = eng.ingest(kb)

    assert report.added == 1
    assert any("录像.mp4" in u for u in report.unsupported)
    assert any("空白图.png" in e for e in report.empty_text), "无文字图片必须出现在 empty_text"
    assert report.skipped == 0, "「没内容」不能混进「没改动」的计数"
    eng.close()


# ---------------- 图片 OCR ----------------

def test_image_ocr_extracts_text(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime", reason="需要 OCR 依赖")
    img = _text_image(tmp_path / "截图.png", ["ROOM 600 RMB", "INVOICE LIMIT"])
    doc = load_file(img)

    assert doc.source_type == "image"
    assert doc.meta["ocr"] is True
    assert 0 < doc.meta["ocr_confidence"] <= 1
    assert "600" in doc.text, f"OCR 没读出关键数字：{doc.text!r}"


def test_image_without_text_is_flagged_not_silent(tmp_path):
    img = _text_image(tmp_path / "纯白.png", [])
    doc = load_file(img)
    assert doc.text.strip() == ""
    assert "empty_reason" in doc.meta, "无字图片必须给出原因，否则入库时会静默跳过"


# ---------------- PDF ----------------

def test_pdf_drops_running_heads_and_page_numbers(tmp_path):
    doc = load_file(_two_col_pdf(tmp_path / "双栏.pdf"))
    assert doc.source_type == "pdf"
    assert "内部资料" not in doc.text, "页眉应被识别并删除"
    assert "- 1 -" not in doc.text and "\n1\n" not in doc.text, "页码应被删除"
    assert doc.meta["pages"] == 4


def test_pdf_two_column_order(tmp_path):
    doc = load_file(_two_col_pdf(tmp_path / "双栏.pdf", pages=1))
    left = doc.text.index("左栏第一条")
    right = doc.text.index("右栏第四条")
    assert left < right, "双栏应按「先左栏后右栏」排序，否则读起来是断句"


def test_scanned_pdf_falls_back_to_ocr(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime", reason="需要 OCR 依赖")
    img = _text_image(tmp_path / "页.png", ["HOTEL 600", "DEADLINE 15 DAYS"], size=(1240, 1754))
    doc = load_file(_scanned_pdf(tmp_path / "扫描.pdf", img))

    assert doc.meta["ocr"] is True
    assert doc.meta["ocr_pages"] == 1
    assert "600" in doc.text, f"扫描页应通过 OCR 出文字：{doc.text!r}"


def test_scanned_pdf_without_text_reports_reason(tmp_path):
    """整页白图：OCR 也读不出字，必须给出原因而不是当成空文档。"""
    img = _text_image(tmp_path / "白页.png", [], size=(1240, 1754))
    doc = load_file(_scanned_pdf(tmp_path / "空白扫描.pdf", img))
    assert doc.text.strip() == ""
    assert "empty_reason" in doc.meta


# ---------------- 元数据传递 ----------------

def test_ocr_metadata_reaches_chunk(tmp_path):
    """OCR 置信度必须传到 chunk 的 meta —— 检索层要靠它给识别类内容降权。"""
    from app.ingest.chunker import chunk_document
    from app.models import Document
    from tests.conftest import make_cfg

    cfg = make_cfg(tmp_path)
    doc = Document(doc_id="d1", path="住宿.png", title="住宿", source_type="image",
                   content_hash="h", mtime=0.0,
                   meta={"modality": "image", "ocr": True, "ocr_confidence": 0.86})
    chunks = chunk_document(doc, "# 住宿\n\nHOTEL 600 元/晚，含早餐。", cfg.chunking)

    assert chunks
    for c in chunks:
        assert c.meta["ocr"] is True
        assert c.meta["ocr_confidence"] == 0.86
        assert c.meta["modality"] == "image"


def test_single_file_ingest(tmp_path):
    from app.rag.engine import RagEngine
    from tests.conftest import make_cfg

    f = tmp_path / "单文件.md"
    f.write_text("# 单文件\n\n年假 15 天。", encoding="utf-8")
    eng = RagEngine(make_cfg(tmp_path / "data"))
    report = eng.ingest(f)
    assert report.added == 1
    eng.close()
