@echo off
title Terminal AI V14.0
color 0A
setlocal

set "ROOT=%~dp0"
set "BUILD_MARKER=%ROOT%.built"
set "VENV_DIR=%ROOT%venv"
set "CONTROL_PANEL=%ROOT%control_panel.py"
set "PATH_REPAIR_COUNT=0"

:Start
if not exist "%CONTROL_PANEL%" (
    echo.
    echo  [ERROR] control_panel.py not found.
    echo  Expected: %CONTROL_PANEL%
    pause & exit /b 1
)
if not exist "%BUILD_MARKER%" goto :NeedsSetup
if not exist "%VENV_DIR%\Scripts\python.exe" goto :NeedsSetup
goto :Ready


:NeedsSetup
echo.
echo  [INFO] First run — setting up. This runs BUILD.bat once.
echo.
set "TERMINAL_AI_LAUNCHER=1"
call "%ROOT%BUILD.bat"
set "BUILD_ERROR=%errorlevel%"
set "TERMINAL_AI_LAUNCHER="
if not exist "%VENV_DIR%\Scripts\python.exe" (
    if "%BUILD_ERROR%"=="2" goto :RecoverPath
    echo  [ERROR] Setup did not complete. Cannot launch.
    pause & exit /b 1
)

:Ready
echo  [OK] Ready: %VENV_DIR%\Scripts\python.exe

start "" "%VENV_DIR%\Scripts\pythonw.exe" "%CONTROL_PANEL%"
if errorlevel 1 (
    echo  [WARN] pythonw.exe launch failed, falling back to python.exe ^(a console window will also open^).
    start "" "%VENV_DIR%\Scripts\python.exe" "%CONTROL_PANEL%"
    if errorlevel 1 goto :RecoverPath
)
exit /b 0

:RecoverPath
echo.
echo  Checking for Python...
set "PYTHON_EXE="
set "PYTHON_DIR="
python --version >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=python"
if not defined PYTHON_EXE (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYTHON_EXE=%%P"
    if defined PYTHON_EXE for %%P in ("%PYTHON_EXE%") do set "PYTHON_DIR=%%~dpP"
)
if not defined PYTHON_EXE (
    echo  [ERROR] Python was not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)
if %PATH_REPAIR_COUNT% GEQ 2 (
    echo  [ERROR] PATH recovery was already attempted twice. Stopping to prevent a loop.
    pause
    exit /b 1
)

call "%ROOT%add_this_folder_to_path.bat" /quiet
if errorlevel 1 (
    echo  [ERROR] Could not add this folder to PATH.
    pause
    exit /b 1
)
if defined PYTHON_DIR set "PATH=%PYTHON_DIR%;%PATH%"
set "PATH=%ROOT%;%PATH%"
echo  Folder added to PATH.
set /a PATH_REPAIR_COUNT+=1
echo  Restarting setup and launch ^(PATH recovery %PATH_REPAIR_COUNT% of 2^)...
goto :Start
