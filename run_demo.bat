@echo off
REM COMP6713 bravo11234 - Gradio demo launcher (Windows).
REM Double-click or run from any terminal; opens http://127.0.0.1:7860 automatically.

setlocal
set HERE=%~dp0
set PY=python

if exist "%HERE%.venv\Scripts\python.exe" set PY="%HERE%.venv\Scripts\python.exe"

echo.
echo Starting Gradio demo on http://127.0.0.1:7860
echo (First launch may take 30s while transformers/gradio import.)
echo Close this window or press Ctrl+C to stop the server.
echo.

REM Launch the browser after a short delay, without blocking server startup.
start "" /min cmd /c "timeout /t 20 /nobreak >nul & start http://127.0.0.1:7860/"

%PY% "%HERE%CODE\app.py" %*
