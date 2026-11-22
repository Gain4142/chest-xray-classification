@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run setup_windows_rtx50.cmd first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m jupyterlab
if errorlevel 1 pause
