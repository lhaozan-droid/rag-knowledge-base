"""API 调用量统计：轻量、无外部依赖的内存计数器 + 落盘持久化。

设计取舍：
- 不引入 prometheus 等重依赖——知识库系统单机部署，一个线程安全的内存聚合器足够；
- 记录粒度为「路径 + 方法」，按小时分桶，既能看总量也能看趋势；
- 每 N 次落一次盘（+ 进程退出时兜底），避免每条请求都写磁盘拖慢响应；
- 数据存 data/api_metrics.json，重启后可继续累加（不丢历史）。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List

# 每小时一个桶，最多保留 90 天（防文件无限膨胀）
_RETENTION_DAYS = 90
_FLUSH_EVERY = 50          # 每累计 50 次调用落一次盘
_FLUSH_INTERVAL_S = 60.0   # 或每 60 秒落一次（取先到者）


class ApiMetrics:
    """线程安全的 API 调用量统计器。

    数据结构：_data = {
        "started_at": <首次启动时间戳>,
        "buckets": {
            "<epoch_hour>": {
                "<METHOD path>": {"count": N, "errors": N, "total_ms": N, "max_ms": N}
            }
        }
    }
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data: dict = {"started_at": time.time(), "buckets": {}}
        self._pending = 0
        self._last_flush = time.time()
        self._load()

    # ---------------- 持久化 ----------------
    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("buckets"), dict):
                    self._data = raw
                    self._prune()
        except Exception:  # noqa: BLE001 —— 统计文件损坏不影响主流程
            pass

    def _prune(self) -> None:
        """只保留最近 _RETENTION_DAYS 天的小时桶。"""
        cutoff = int(time.time() // 3600) - _RETENTION_DAYS * 24
        self._data["buckets"] = {
            h: v for h, v in self._data["buckets"].items()
            if int(h) >= cutoff
        }

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path)
        except Exception:  # noqa: BLE001
            pass

    # ---------------- 记录 ----------------
    def record(self, method: str, path: str, status_code: int, elapsed_ms: float) -> None:
        """记录一次调用。path 归一化：去掉 query string。"""
        key_path = path.split("?")[0]
        hour = int(time.time() // 3600)
        with self._lock:
            buckets = self._data.setdefault("buckets", {})
            hour_bucket = buckets.setdefault(str(hour), {})
            k = f"{method.upper()} {key_path}"
            item = hour_bucket.get(k)
            ms = int(elapsed_ms)
            if item is None:
                hour_bucket[k] = {"count": 1, "errors": 1 if status_code >= 400 else 0,
                                  "total_ms": ms, "max_ms": ms}
            else:
                item["count"] += 1
                if status_code >= 400:
                    item["errors"] += 1
                item["total_ms"] += ms
                item["max_ms"] = max(item["max_ms"], ms)
            self._pending += 1
            now = time.time()
            should_flush = self._pending >= _FLUSH_EVERY or (now - self._last_flush) >= _FLUSH_INTERVAL_S
            if should_flush:
                self._prune()
                self._save()
                self._pending = 0
                self._last_flush = now

    def flush(self) -> None:
        with self._lock:
            self._prune()
            self._save()
            self._pending = 0
            self._last_flush = time.time()

    # ---------------- 查询 ----------------
    def summary(self, hours: int = 24) -> dict:
        """最近 N 小时的调用量汇总，供仪表盘展示。

        返回：
        - total: 总调用次数
        - errors: 总错误数（status >= 400）
        - avg_ms / max_ms: 平均/最大耗时
        - by_endpoint: [{path, method, count, errors, avg_ms, max_ms}] 按调用次数降序
        - by_hour: [{hour, count, errors}] 时间趋势（按时间升序）
        """
        hours = max(1, min(int(hours), 24 * 30))  # 夹在 1 小时 ~ 30 天
        now_hour = int(time.time() // 3600)
        start_hour = now_hour - hours + 1

        with self._lock:
            buckets = self._data.get("buckets", {})
            endpoint_agg: Dict[str, dict] = {}
            hour_agg: Dict[int, dict] = {}

            for hstr, ep_map in buckets.items():
                h = int(hstr)
                if h < start_hour:
                    continue
                hour_item = hour_agg.setdefault(h, {"count": 0, "errors": 0})
                for k, item in ep_map.items():
                    method, _, path = k.partition(" ")
                    key = (path, method)
                    e = endpoint_agg.setdefault(key, {
                        "path": path, "method": method,
                        "count": 0, "errors": 0, "total_ms": 0, "max_ms": 0})
                    e["count"] += item["count"]
                    e["errors"] += item["errors"]
                    e["total_ms"] += item["total_ms"]
                    e["max_ms"] = max(e["max_ms"], item["max_ms"])
                    hour_item["count"] += item["count"]
                    hour_item["errors"] += item["errors"]

            total = sum(e["count"] for e in endpoint_agg.values())
            total_errors = sum(e["errors"] for e in endpoint_agg.values())
            total_ms = sum(e["total_ms"] for e in endpoint_agg.values())
            max_ms = max((e["max_ms"] for e in endpoint_agg.values()), default=0)

            by_endpoint = sorted(endpoint_agg.values(), key=lambda e: -e["count"])
            for e in by_endpoint:
                e["avg_ms"] = round(e["total_ms"] / e["count"], 1) if e["count"] else 0
                e.pop("total_ms", None)

            by_hour = [
                {"hour": h, "count": hour_agg[h]["count"], "errors": hour_agg[h]["errors"]}
                for h in sorted(hour_agg)
            ]

        return {
            "hours": hours,
            "total": total,
            "errors": total_errors,
            "avg_ms": round(total_ms / total, 1) if total else 0,
            "max_ms": max_ms,
            "by_endpoint": by_endpoint,
            "by_hour": by_hour,
        }
