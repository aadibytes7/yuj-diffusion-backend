@echo off
title YUJ Diffusion Server
cd /d "%~dp0"
echo Activating virtual environment...
call .\.venv\Scripts\activate.bat
echo Starting FastAPI Diffuser Server on http://127.0.0.1:8000 ...
python server.py
pause
