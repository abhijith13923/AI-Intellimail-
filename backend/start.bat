@echo off
echo Starting FastAPI Backend...
call venv\Scripts\activate
uvicorn app.main:app --reload
pause
