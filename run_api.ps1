# Run the FastAPI server
# Usage: .\run_api.ps1

$env:PYTHONPATH = "src"
.\.venv_win\Scripts\python.exe -m uvicorn src.app.api.main:app --reload --host 0.0.0.0 --port 8000

