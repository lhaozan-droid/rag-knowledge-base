@echo off
rem 启动私有化 RAG 服务，默认只监听本机
rem 本文件必须保存为 GBK 编码加 CRLF 换行，否则中文 Windows 的 cmd 会解析成乱命令
chcp 936 >nul
setlocal

rem 回到项目根目录，不依赖双击时的当前目录
cd /d "%~dp0.."

set "HOST=127.0.0.1"
if not "%~1"=="" set "HOST=%~1"
set "PORT=8080"
if not "%~2"=="" set "PORT=%~2"

rem 优先用项目内虚拟环境，找不到再退回系统 python
rem 显式指定优先：先 set RAG_PYTHON=C:\path\to\python.exe 再运行本脚本
set "PY=python"
if not "%RAG_PYTHON%"=="" set "PY=%RAG_PYTHON%"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
rem 依赖自检：只查有没有装，不真的 import
"%PY%" -c "import importlib.util as u,sys;sys.exit(0 if u.find_spec('chromadb') and u.find_spec('fastembed') else 1)" >nul 2>&1
if errorlevel 1 (
  echo [错误] 这个解释器里找不到 chromadb / fastembed：%PY%
  echo        两种改法，任选其一：
  echo        1^) 装依赖：%PY% -m pip install -r requirements.txt
  echo        2^) 换解释器：set RAG_PYTHON=C:\path\to\python.exe 后重跑本脚本
  echo        注意：依赖装在别的环境里很常见，这时要选 2 而不是 1。
  pause
  endlocal
  exit /b 1
)

if not exist ".env" (
  echo [提示] 未找到 .env，将使用 config.yaml 里的默认配置
  echo [提示] 没有配置 LLM_API_KEY 时，回答会退化为离线摘录模式
  echo.
)

echo 解释器   : %PY%
echo 访问地址 : http://%HOST%:%PORT%
echo 监听范围 : 默认只监听 %HOST%，局域网内其他机器访问不到
echo 首次访问 : 终端会打印管理员初始口令，只打印一次，登录后请立即修改
echo.

"%PY%" -m app.cli serve --host %HOST% --port %PORT% %*
if errorlevel 1 (
  echo.
  echo [错误] 启动失败。请确认已安装依赖 pip install -r requirements.txt
  echo [错误] 并确认 Python 版本在 3.10 以上
  pause
)

endlocal
