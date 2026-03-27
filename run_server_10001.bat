@echo off
cd /d %~dp0

:: Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating .venv...
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    echo No .venv found, using system Python...
)

set PYTHONPATH=%~dp0
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONLEGACYWINDOWSSTDIO=0
echo Starting Open-AutoGLM Service on port 10001...
python -m uvicorn server.ai_service:app --host 0.0.0.0 --port 10001 --reload --log-level info
pause
