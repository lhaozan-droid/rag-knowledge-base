"""HTTP 服务 + 内网调试台。

后端接口面向后续企业微信接入，所以全部设计成「无状态请求 + conversation_id」，
企微回调只需把 FromUserName 映射成 conversation_id 即可直接复用。
"""
from __future__ import annotations

import atexit
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Header, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.auth import AccountStore, init_secret, sign_token, verify_token
from app.config import load_config
from app.metrics import ApiMetrics
from app.rag.engine import ConversationAccessError, RagEngine

log = logging.getLogger(__name__)

_engine: Optional[RagEngine] = None
_lock = threading.Lock()
CONSOLE = Path(__file__).parent / "console.html"
LOGIN_PAGE = Path(__file__).parent / "login.html"
ADMIN_PAGE = Path(__file__).parent / "admin.html"

_config = load_config()
init_secret(_config.data_path)  # 启动即初始化签名密钥，避免首个受保护请求因空密钥 401
_auth: Optional[AccountStore] = None
_metrics = ApiMetrics(_config.data_path / "api_metrics.json")

# 进程退出兜底（非 FastAPI 优雅关闭路径）：覆盖 Ctrl+C、直接关闭窗口、uvicorn
# 进程被 SIGTERM 结束等解释器正常退出的场景。lifespan 只在 FastAPI graceful
# shutdown 时触发；atexit 则保证解释器正常退出时也补一次 flush。
# 硬杀（taskkill /F、kill -9）连 atexit 都不会执行，属物理极限，无法覆盖。
atexit.register(_metrics.flush)


def get_engine() -> RagEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = RagEngine(load_config())
    return _engine


def get_auth() -> AccountStore:
    global _auth
    if _auth is None:
        with _lock:
            if _auth is None:
                init_secret(_config.data_path)
                _auth = AccountStore(_config.data_path / "auth.json")
    return _auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：优雅关闭时兜底落盘调用量统计。

    指标平时按「累计 50 次 / 60 秒」批量落盘，若进程在阈值前被停止，
    最近一小段统计会留在内存里丢失。这里在 shutdown 阶段补一次 flush。
    注意：SIGKILL（kill -9）不经过任何退出钩子，此兜底无法覆盖，属正常取舍。
    """
    yield
    _metrics.flush()


app = FastAPI(title="rag-knowledge-base", version="0.1.0", lifespan=lifespan)


# ------------------------- 调用量统计中间件 -------------------------
def _metric_path(request: Request) -> str:
    """用路由模板（如 /conversations/{conversation_id}）代替带具体 id 的原始 path。

    防止把 conversation_id 这类会话标识写进 api_metrics.json 并展示在管理后台。
    未匹配到路由（404）时退回原始 path——那本来就是不存在的路径，不含敏感信息。
    """
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    """记录每个 HTTP 请求的调用量/耗时/状态码，供 API 调用量仪表盘使用。"""
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:  # noqa: BLE001 —— 统计不影响异常冒泡
        status = 500
        raise
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _metrics.record(request.method, _metric_path(request), status, elapsed_ms)
    return response


# ------------------------- 鉴权 -------------------------
def _current_account(request: Request, authorization: Optional[str] = Header(None)) -> dict:
    """解析当前账号。优先 Cookie（浏览器自动带），其次 Authorization: Bearer 头。

    无 token / 失效 / 账号被停用 → 401。
    """
    token = request.cookies.get("rag_token")
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token:
        raise HTTPException(401, "未登录")
    info = verify_token(token)
    if not info:
        raise HTTPException(401, "登录已失效，请重新登录")
    acct = get_auth().get(info["username"])
    if not acct or not acct.enabled:
        raise HTTPException(401, "账号已停用")
    # role 以账号库为唯一权威来源，token 内嵌 role 只用于身份识别（username）。
    # 否则管理员被降权后，旧 token 最长 7 天仍带 role=admin，可继续访问 /admin。
    return {"username": info["username"], "role": acct.role}


def _require_admin(user: dict = Depends(_current_account)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(403, "需要管理员权限")
    return user


def _cookie_token(request: Request) -> Optional[str]:
    return request.cookies.get("rag_token")


# ------------------------- schemas -------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


class AdminCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class AdminSetPasswordRequest(BaseModel):
    username: str
    password: str


class AdminSetEnabledRequest(BaseModel):
    username: str
    enabled: bool


class AdminSetRoleRequest(BaseModel):
    username: str
    role: str


class IngestRequest(BaseModel):
    path: str
    reset: bool = False
    force: bool = False


class DocDeleteRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=512)


class DocsDeleteRequest(BaseModel):
    doc_ids: List[str] = Field(min_length=1, max_length=2000)


class CategoryCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    parent_id: int = 0


class CategoryRenameRequest(BaseModel):
    cat_id: int
    name: str = Field(min_length=1, max_length=100)


class CategoryMoveRequest(BaseModel):
    cat_id: int
    parent_id: int


class CategoryDeleteRequest(BaseModel):
    cat_id: int
    force: bool = False


class CategoryOrderRequest(BaseModel):
    """[[cat_id, sort_order], ...]：只提交需要调整的条目。"""
    orders: List[List[int]] = Field(min_length=1, max_length=2000)


class DocCategoryRequest(BaseModel):
    doc_ids: List[str] = Field(min_length=1, max_length=2000)
    category_id: int = 0


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    where: Optional[dict] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 6
    where: Optional[dict] = None


# ------------------------- routes -------------------------
@app.get("/", response_class=HTMLResponse)
def console(request: Request) -> object:
    """根路径：有有效会话且账号未被停用则返回问答页，否则返回登录页。"""
    tok = _cookie_token(request)
    if tok:
        info = verify_token(tok)
        acct = info and get_auth().get(info["username"])
        if acct and acct.enabled:
            if CONSOLE.exists():
                return CONSOLE.read_text(encoding="utf-8")
            return "<h1>rag-knowledge-base</h1><p>console.html 缺失</p>"
    if LOGIN_PAGE.exists():
        return LOGIN_PAGE.read_text(encoding="utf-8")
    return "<h1>企业知识库</h1><p>登录页缺失</p>"


@app.post("/login")
def login(req: LoginRequest):
    """登录：校验账号密码，成功则种一个 7 天有效的 HttpOnly cookie。"""
    auth = get_auth()
    if not auth.verify(req.username, req.password):
        raise HTTPException(401, "账号或密码错误")
    acct = auth.get(req.username)
    token = sign_token(acct.username, acct.role)
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie("rag_token", token, httponly=True, samesite="lax",
                    max_age=7 * 24 * 3600, path="/")
    return resp


@app.post("/logout")
def logout():
    resp = RedirectResponse("/", status_code=303)
    # 不能用 delete_cookie()：它内部 expires=0 会被 Python http.cookies 当作
    # 「距现在的相对秒数」，渲染成 expires=<当前时间>（未来），而非 1970 过期时间；
    # 浏览器/部分客户端据此无法可靠删除 cookie，导致退出后跳 / 又被识别为有效会话，
    # 表现为「退不出去」。这里显式写一个过去时间点，确保 cookie 被彻底清除。
    resp.set_cookie("rag_token", "", max_age=0,
                    expires=datetime(1970, 1, 1, tzinfo=timezone.utc),
                    path="/", httponly=True, samesite="lax")
    return resp


@app.get("/me", dependencies=[Depends(_current_account)])
def me(user: dict = Depends(_current_account)) -> dict:
    return {"username": user["username"], "role": user["role"]}


# ================= 管理员 =================
@app.get("/admin", response_class=HTMLResponse)
def admin_page(user: dict = Depends(_require_admin)) -> str:
    if ADMIN_PAGE.exists():
        return ADMIN_PAGE.read_text(encoding="utf-8")
    return "<h1>账号管理</h1><p>admin.html 缺失</p>"


@app.get("/admin/api/accounts", dependencies=[Depends(_require_admin)])
def admin_list(user: dict = Depends(_require_admin)) -> dict:
    return {"accounts": get_auth().list_public(), "me": user["username"]}


@app.post("/admin/api/accounts", dependencies=[Depends(_require_admin)])
def admin_create(req: AdminCreateRequest, user: dict = Depends(_require_admin)) -> dict:
    err = get_auth().create(req.username, req.password, req.role)
    if err:
        raise HTTPException(400, err)
    return {"ok": True}


@app.post("/admin/api/password", dependencies=[Depends(_require_admin)])
def admin_set_password(req: AdminSetPasswordRequest, user: dict = Depends(_require_admin)) -> dict:
    err = get_auth().set_password(req.username, req.password)
    if err:
        raise HTTPException(400, err)
    return {"ok": True}


@app.post("/admin/api/enabled", dependencies=[Depends(_require_admin)])
def admin_set_enabled(req: AdminSetEnabledRequest, user: dict = Depends(_require_admin)) -> dict:
    err = get_auth().set_enabled(req.username, req.enabled)
    if err:
        raise HTTPException(400, err)
    return {"ok": True}


@app.post("/admin/api/role", dependencies=[Depends(_require_admin)])
def admin_set_role(req: AdminSetRoleRequest, user: dict = Depends(_require_admin)) -> dict:
    err = get_auth().set_role(req.username, req.role)
    if err:
        raise HTTPException(400, err)
    return {"ok": True}


@app.get("/admin/api/docs", dependencies=[Depends(_require_admin)])
def admin_list_docs(q: str = Query("", max_length=200),
                    type: str = Query("", max_length=32),
                    category_id: int = Query(0, ge=0),
                    include_sub: bool = Query(True),
                    unclassified: bool = Query(False),
                    limit: int = Query(50, ge=0, le=500),
                    offset: int = Query(0, ge=0)) -> dict:
    """知识库文档清单，支持关键字/类型/分类筛选与分页。

    limit=0 表示不分页（一次性返回），仅调试脚本使用；后台页面固定传分页参数。
    total 始终是与当前筛选条件匹配的总数（不是本页条数），否则前端翻页会算错页数。
    """
    eng = get_engine()
    docs = eng.list_docs(q=q.strip() or None, source_type=type.strip() or None,
                         category_id=category_id or None,
                         include_sub=include_sub, limit=limit, offset=offset,
                         unclassified=unclassified)
    total = eng.count_docs(q=q.strip() or None, source_type=type.strip() or None,
                           category_id=category_id or None,
                           include_sub=include_sub, unclassified=unclassified)
    return {"docs": docs, "total": total, "limit": limit, "offset": offset,
            "kb_root": eng.kb_root()}


@app.get("/admin/api/docs/{doc_id}/chunks", dependencies=[Depends(_require_admin)])
def admin_doc_chunks(doc_id: str) -> dict:
    """某篇文档的全部分块（子块+父块），供管理后台「查看切片」只读展示。"""
    try:
        chunks = get_engine().chunks_of_doc(doc_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"doc_id": doc_id, "chunks": chunks, "total": len(chunks)}


@app.post("/admin/api/docs/delete", dependencies=[Depends(_require_admin)])
def admin_delete_doc(req: DocDeleteRequest) -> dict:
    """单篇删除：按 doc_id（兼容 path）删除文档及其全部 chunk。"""
    try:
        result = get_engine().delete_doc(req.doc_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, **result}


@app.post("/admin/api/docs/delete-batch", dependencies=[Depends(_require_admin)])
def admin_delete_docs(req: DocsDeleteRequest) -> dict:
    """批量删除。逐篇删，单篇失败不影响其余，错误一并返回给前端提示。"""
    return {"ok": True, **get_engine().delete_docs(req.doc_ids)}


@app.post("/admin/api/docs/category", dependencies=[Depends(_require_admin)])
def admin_set_doc_category(req: DocCategoryRequest) -> dict:
    """批量给文档归类。category_id=0 表示置为「未分类」。"""
    eng = get_engine()
    if req.category_id and not eng.registry.get_category(req.category_id):
        raise HTTPException(404, f"分类不存在：{req.category_id}")
    n = eng.set_docs_category(req.doc_ids, req.category_id)
    return {"ok": True, "updated": n}


# ================= 分类管理 =================
@app.get("/admin/api/categories", dependencies=[Depends(_require_admin)])
def admin_list_categories() -> dict:
    """分类树（扁平返回，含文档数；前端按 parent_id 组装）。"""
    return {"categories": get_engine().list_categories()}


@app.post("/admin/api/categories", dependencies=[Depends(_require_admin)])
def admin_create_category(req: CategoryCreateRequest) -> dict:
    try:
        cid = get_engine().create_category(req.name, req.parent_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, "id": cid}


@app.post("/admin/api/categories/rename", dependencies=[Depends(_require_admin)])
def admin_rename_category(req: CategoryRenameRequest) -> dict:
    try:
        get_engine().rename_category(req.cat_id, req.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True}


@app.post("/admin/api/categories/move", dependencies=[Depends(_require_admin)])
def admin_move_category(req: CategoryMoveRequest) -> dict:
    try:
        get_engine().move_category(req.cat_id, req.parent_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True}


@app.post("/admin/api/categories/order", dependencies=[Depends(_require_admin)])
def admin_reorder_categories(req: CategoryOrderRequest) -> dict:
    """调整同级分类排序。前端上移/下移时只提交交换的两条。"""
    orders: List[tuple] = []
    for item in req.orders:
        if len(item) >= 2:
            orders.append((int(item[0]), int(item[1])))
    get_engine().registry.reorder_categories(orders)
    return {"ok": True, "updated": len(orders)}


@app.post("/admin/api/categories/delete", dependencies=[Depends(_require_admin)])
def admin_delete_category(req: CategoryDeleteRequest) -> dict:
    try:
        res = get_engine().delete_category(req.cat_id, force=req.force)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True, **res}


@app.post("/admin/api/categories/migrate", dependencies=[Depends(_require_admin)])
def admin_migrate_categories() -> dict:
    """把历史 meta.category 字符串迁移成分类树（幂等，可重复执行）。"""
    return {"ok": True, **get_engine().migrate_categories_from_meta()}


@app.post("/admin/api/docs/upload", dependencies=[Depends(_require_admin)])
async def admin_upload_doc(file: UploadFile = File(...),
                           category_id: int = Form(0)) -> dict:
    """上传单篇文档入库，并可选直接归到某个分类。

    存到 data/uploads/ 后再走正常入库流水线，而不是另写一条入库路径——
    二次加工的分块/向量/BM25 逻辑一旦分叉，检索质量就会出现两套标准。
    """
    eng = get_engine()
    if category_id and not eng.registry.get_category(category_id):
        raise HTTPException(404, f"分类不存在：{category_id}")
    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(400, "文件名不能为空")
    upload_dir = eng.cfg.data_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / name
    # 同名文件覆盖写：入库以内容哈希为身份，重传同一份文件只会命中跳过，
    # 不会像「自动改名成 xxx(1).md」那样悄悄多出一份重复文档。
    data = await file.read()
    dest.write_bytes(data)
    try:
        report = eng.ingest(str(dest), force=True)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    doc = eng.registry.get_doc_by_path(name)
    if doc is None:
        return {"ok": False, "report": report.to_dict(),
                "message": "文件已保存但未产生可用文本（格式不支持或内容为空）"}
    if category_id:
        eng.set_doc_category(doc.doc_id, category_id)
    return {"ok": True, "doc_id": doc.doc_id, "title": doc.title,
            "path": doc.path, "category_id": category_id,
            "chunks": eng.registry.count_chunks_of_doc(doc.doc_id),
            "report": report.to_dict()}


@app.post("/admin/api/bm25/rebuild", dependencies=[Depends(_require_admin)])
def admin_rebuild_bm25() -> dict:
    """重建 BM25 稀疏索引。

    分类改名或批量重新归类后必须跑一次：BM25 的索引文本里拼了所属分类名
    （为了让「分类名」这种用户口语也能命中），分类变了而索引没重建，
    检索仍在使用旧分类词，表现为「改了分类但搜不到」。
    """
    n = get_engine().pipeline.rebuild_bm25()
    return {"ok": True, "chunks": n}


@app.get("/admin/api/ingest/report", dependencies=[Depends(_require_admin)])
def admin_ingest_report() -> dict:
    """最近一次入库报告；从未入库时 report / last_run_at 均为 null。"""
    eng = get_engine()
    return {"report": eng.last_ingest_report(), "last_run_at": eng.last_ingest_at()}


@app.get("/admin/api/metrics", dependencies=[Depends(_require_admin)])
def admin_metrics(hours: int = Query(24, ge=1, le=720)) -> dict:
    """API 调用量仪表盘数据：总调用/错误/耗时、按接口分布、按小时趋势。"""
    return _metrics.summary(hours=hours)


# ================= 业务接口（全部需登录） =================
@app.get("/health", dependencies=[Depends(_current_account)])
def health() -> dict:
    try:
        return {"status": "ok", "stats": get_engine().stats()}
    except Exception as exc:  # noqa: BLE001
        return {"status": "degraded", "error": str(exc)}


@app.get("/stats", dependencies=[Depends(_current_account)])
def stats() -> dict:
    return get_engine().stats()


@app.post("/ingest", dependencies=[Depends(_require_admin)])
def ingest(req: IngestRequest) -> dict:
    try:
        report = get_engine().ingest(req.path, reset=req.reset, force=req.force)
        return report.to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/search", dependencies=[Depends(_current_account)])
def search(req: SearchRequest) -> dict:
    hits, best = get_engine().retrieve(req.query, where=req.where, top_k=req.top_k)
    return {"best_dense": round(best, 4), "hits": [
        {"chunk_id": h.chunk.chunk_id, "title": h.chunk.title,
         "section": h.chunk.section, "path": h.chunk.path,
         "score": round(h.score, 6), "dense": round(h.dense_score, 4),
         "bm25": round(h.sparse_score, 3), "text": h.chunk.text}
        for h in hits]}


@app.post("/chat", dependencies=[Depends(_current_account)])
def chat(req: ChatRequest, user: dict = Depends(_current_account)) -> dict:
    try:
        return get_engine().answer(req.query, conversation_id=req.conversation_id,
                                   where=req.where, username=user["username"]).to_dict()
    except ConversationAccessError as exc:
        raise HTTPException(404, "会话不存在") from exc


@app.post("/chat/stream", dependencies=[Depends(_current_account)])
def chat_stream(req: ChatRequest, user: dict = Depends(_current_account)) -> StreamingResponse:
    try:
        get_engine().check_conversation_access(req.conversation_id, user["username"])
    except ConversationAccessError as exc:
        raise HTTPException(404, "会话不存在") from exc

    def gen():
        try:
            for event in get_engine().answer_stream(
                    req.query, conversation_id=req.conversation_id, where=req.where,
                    username=user["username"]):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            log.exception("流式回答异常")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/conversations", dependencies=[Depends(_current_account)])
def list_conversations(user: dict = Depends(_current_account),
                       limit: int = Query(50, ge=1, le=200)) -> dict:
    """当前登录用户的历史会话列表，供客服工作台「历史对话」渲染。

    只返回归属当前账号的会话，按最近活跃倒序；账号之间互不可见。
    """
    conversations = get_engine().registry.list_conversations(user["username"], limit)
    return {"conversations": conversations}


@app.get("/conversations/{conversation_id}", dependencies=[Depends(_current_account)])
def conversation(conversation_id: str, limit: int = Query(50, ge=1, le=200),
                 user: dict = Depends(_current_account)) -> dict:
    turns = get_engine().registry.get_turns(conversation_id, limit,
                                            username=user["username"])
    if not turns:
        # 会话不存在、或不属于当前账号，统一 404：不泄露会话是否存在/归属何人。
        raise HTTPException(404, "会话不存在")
    return {"conversation_id": conversation_id,
            "turns": [{"role": t.role, "content": t.content, "ts": t.ts} for t in turns]}


@app.get("/unanswered", dependencies=[Depends(_require_admin)])
def unanswered(limit: int = Query(50, ge=1, le=500)) -> dict:
    """培训盲区清单：哪些问题知识库里没有答案，反向指导补文档。"""
    return {"items": get_engine().registry.list_unanswered(limit)}
