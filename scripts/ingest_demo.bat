@echo off
rem 一键把 demo-docs 导入知识库
rem 本文件必须保存为 GBK 编码加 CRLF 换行，否则中文 Windows 的 cmd 会解析成乱命令
chcp 936 >nul
setlocal

rem 回到项目根目录，不依赖双击时的当前目录
cd /d "%~dp0.."

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

echo 解释器   : %PY%
echo 导入目录 : .\demo-docs
echo 首次运行 : 需要下载本地 embedding 模型到 data\models，模型约 640MB
echo.

"%PY%" -m app.cli ingest ./demo-docs %*
if errorlevel 1 (
  echo.
  echo [错误] 入库失败，请检查上面的输出
)

echo.
pause
endlocal
