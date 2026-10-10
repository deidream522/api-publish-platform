@echo off
REM COMP6713 bravo11234 - Command-line tester launcher (Windows).
REM Double-click or run from any terminal; no `cd` needed.

setlocal
set HERE=%~dp0
set PY=python

REM Prefer a local venv if the submitter shipped one.
if exist "%HERE%.venv\Scripts\python.exe" set PY="%HERE%.venv\Scripts\python.exe"

if "%~1"=="" (
    echo.
    echo Running a default demo prompt. Pass your own arguments to override.
    echo Example:  run_cli.bat --model roberta --text "you are an idiot"
    echo           run_cli.bat --file CODE\sample_inputs.txt
    echo.
    %PY% "%HERE%CODE\cli.py" --text "you are an absolute idiot and should shut up"
) else (
    %PY% "%HERE%CODE\cli.py" %*
)

echo.
pause
