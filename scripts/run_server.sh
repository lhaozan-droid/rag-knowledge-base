#!/usr/bin/env bash
# 启动私有化 RAG 服务（默认只监听本机）
#
# 用法：
#   ./scripts/run_server.sh                 # 127.0.0.1:8080
#   ./scripts/run_server.sh 0.0.0.0 8080    # 显式指定监听地址与端口
#   ./scripts/run_server.sh 127.0.0.1 8080 --reload
set -euo pipefail

# 回到项目根目录，不依赖调用时的工作目录
cd "$(dirname "$0")/.."

HOST="${1:-127.0.0.1}"
PORT="${2:-8080}"

# 优先用项目内虚拟环境，找不到再退回系统 python
# 注意：POSIX 布局是 .venv/bin/python，Windows 布局是 .venv/Scripts/python.exe。
# 只判断前者会在 Git Bash / MSYS 下永远落空，退回到没装依赖的系统 python。
# 解释器优先级：RAG_PYTHON 显式指定 > 项目内 .venv > 系统 python。
# 加 RAG_PYTHON 是因为"哪个 python 装了 chromadb"只有使用者知道：CI、conda、
# pyenv、Windows 上的 entrag 环境都不在 .venv 里，硬猜必然落到没装依赖的那个。
if [ -n "${RAG_PYTHON:-}" ]; then
  PY="$RAG_PYTHON"
elif [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif [ -x ".venv/Scripts/python.exe" ]; then
  PY=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "[错误] 找不到 python。请先安装 Python 3.10+ 并创建虚拟环境：" >&2
  echo "       python -m venv .venv && source .venv/bin/activate" >&2
  echo "       或显式指定：RAG_PYTHON=/path/to/python $0" >&2
  exit 1
fi

# 依赖自检：find_spec 只查有没有装，不真的 import，几十毫秒就返回。
# 不加这一步的话，选到没装依赖的解释器会直接抛 chromadb 的 ImportError 堆栈，
# 让人误以为是 pip install 没成功。
if ! "$PY" -c "import importlib.util as u,sys; sys.exit(0 if u.find_spec('chromadb') and u.find_spec('fastembed') else 1)" >/dev/null 2>&1; then
  echo "[错误] 这个解释器里找不到 chromadb / fastembed：" >&2
  echo "       $PY  ($("$PY" -V 2>&1))" >&2
  echo "       两种改法，任选其一：" >&2
  echo "       1) 装依赖：$PY -m pip install -r requirements.txt" >&2
  echo "       2) 换解释器：RAG_PYTHON=/path/to/python $(basename "$0")" >&2
  echo "       注意：依赖装在别的环境里很常见，这时要选 2 而不是 1。" >&2
  exit 1
fi

if [ ! -f ".env" ]; then
  echo "[提示] 未找到 .env，将使用 config.yaml 里的默认配置"
  echo "[提示] 没有配置 LLM_API_KEY 时，回答会退化为离线摘录模式"
fi

echo "解释器   : $PY"
echo "访问地址 : http://$HOST:$PORT"
echo "监听范围 : 默认只监听 $HOST，局域网内其他机器访问不到"
echo "首次访问 : 终端会打印管理员初始口令，只打印一次，登录后请立即修改"
echo

exec "$PY" -m app.cli serve --host "$HOST" --port "$PORT" "${@:3}"
