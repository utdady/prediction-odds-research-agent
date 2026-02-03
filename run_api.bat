@echo off
REM Run the FastAPI server
REM Usage: run_api.bat

set PYTHONPATH=src
.\.venv_win\Scripts\python.exe -m uvicorn src.app.api.main:app --reload --host 0.0.0.0 --port 8000

