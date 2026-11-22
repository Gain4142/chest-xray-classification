@echo off
setlocal
cd /d "%~dp0"
echo Installing the CXR GPU environment. This may take several minutes.
echo Progress and errors are written to setup_gpu.log in this folder.
call :install > setup_gpu.log 2>&1
if errorlevel 1 (
    type setup_gpu.log
    echo.
    echo SETUP FAILED. Send setup_gpu.log for diagnosis.
    pause
    exit /b 1
)
type setup_gpu.log
echo.
echo Setup completed. Run start_jupyter.cmd next.
pause
exit /b 0

:install
echo Windows RTX 50 setup - PyTorch 2.14.1 CUDA 13.0
if not exist ".venv\Scripts\python.exe" (
    py -3.12 --version
    if errorlevel 1 (
        echo Python 3.12 with the Python launcher is required. Install it first.
        exit /b 1
    )
    py -3.12 -m venv .venv
    if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install --upgrade torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements-gpu-common.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m scripts.check_gpu
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m ipykernel install --user --name cxr-gpu --display-name "CXR GPU"
if errorlevel 1 exit /b 1
exit /b 0
