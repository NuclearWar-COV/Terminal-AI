@echo off
title Terminal AI V14.0 - Setup
color 0B
setlocal

set "ROOT=%~dp0"
set "BUILD_MARKER=%ROOT%.built"
set "VENV_DIR=%ROOT%venv"
set "SERVER_PY=%ROOT%server\web_server.py"

echo.
echo  =====================================================
echo           Terminal AI V14.0 - Setup (Python)
echo  =====================================================
echo.

if not exist "%SERVER_PY%" (
    echo  [ERROR] server\web_server.py not found.
    echo  Expected: %SERVER_PY%
    pause & exit /b 1
)

echo  Checking for Python...
set "PYTHON_COMMAND=python"
python --version >nul 2>&1
if not errorlevel 1 goto :PythonFound

set "PYTHON_COMMAND="
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_COMMAND=py -3"
    goto :PythonFound
)

for /f "tokens=2,*" %%A in ('reg query "HKCU\Software\Python\PythonCore" /s /v ExecutablePath 2^>nul ^| findstr /i "ExecutablePath"') do set "PYTHON_COMMAND="%%B""
if not defined PYTHON_COMMAND (
    for /f "tokens=2,*" %%A in ('reg query "HKLM\Software\Python\PythonCore" /s /v ExecutablePath 2^>nul ^| findstr /i "ExecutablePath"') do set "PYTHON_COMMAND="%%B""
)
if defined PYTHON_COMMAND (
    call %PYTHON_COMMAND% --version >nul 2>&1
    if errorlevel 1 set "PYTHON_COMMAND="
)
if not defined PYTHON_COMMAND goto :PythonMissing

:PythonFound
for /f "tokens=*" %%v in ('%PYTHON_COMMAND% --version 2^>^&1') do set PYVER=%%v
echo  [OK] %PYVER%
goto :PythonReady

:PythonMissing
echo  [ERROR] Python is not installed or could not be found.
echo  Install Python 3.10+ from: https://python.org/downloads
echo  IMPORTANT: Check "Add Python to PATH" during install!
echo.
if defined TERMINAL_AI_LAUNCHER (
    exit /b 2
)
start https://python.org/downloads
pause & exit /b 2

:PythonReady
echo.
:ChooseInstallMode
echo.
echo  Choose how to install project dependencies:
echo   [1] Search the computer - reuse installed packages and pip's wheel cache (offline)
echo   [2] Download from the internet
choice /c 12 /n /m "Enter 1 or 2: "
if errorlevel 2 (
    set "INSTALL_MODE=DOWNLOAD"
) else (
    set "INSTALL_MODE=LOCAL"
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo  Creating virtual environment...
    if /i "%INSTALL_MODE%"=="LOCAL" (
        call %PYTHON_COMMAND% -m venv --system-site-packages "%VENV_DIR%"
    ) else (
        call %PYTHON_COMMAND% -m venv "%VENV_DIR%"
    )
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
) else (
    echo  [OK] Virtual environment already exists.
)

if /i "%INSTALL_MODE%"=="LOCAL" (
    call :SetupLocalSearch
    if errorlevel 1 (
        goto :LocalDependenciesMissing
    )
) else (
    "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 (
        echo  [ERROR] Could not upgrade pip. See the output above.
        pause & exit /b 1
    )
    echo  Downloading dependencies from the internet...
)

echo  Checking for an NVIDIA GPU...
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo  [OK] NVIDIA GPU detected - installing CUDA-enabled PyTorch ^(larger download, this can take a while^)...
    if /i "%INSTALL_MODE%"=="LOCAL" (
        call :PipInstall torch torchvision torchaudio
    ) else (
        call :PipInstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    )
    if errorlevel 1 (
        if /i "%INSTALL_MODE%"=="LOCAL" goto :LocalDependenciesMissing
        echo  [WARN] CUDA PyTorch install failed - falling back to CPU-only PyTorch.
        echo  ^(GPU acceleration won't work; re-run BUILD.bat after checking your internet connection to retry.^)
        call :PipInstall torch torchvision torchaudio
    )
) else (
    echo  No NVIDIA GPU detected - installing CPU-only PyTorch.
    call :PipInstall torch torchvision torchaudio
)
if errorlevel 1 (
    if /i "%INSTALL_MODE%"=="LOCAL" goto :LocalDependenciesMissing
    echo  [ERROR] PyTorch install failed. See the output above.
    pause & exit /b 1
)
echo  [OK] PyTorch installed.

echo  Installing remaining dependencies (this can take a few minutes the first time)...
call :PipInstall -r "%ROOT%requirements.txt"
if errorlevel 1 (
    if /i "%INSTALL_MODE%"=="LOCAL" goto :LocalDependenciesMissing
    echo  [ERROR] Dependency install failed. See the output above.
    pause & exit /b 1
)
echo  [OK] Dependencies installed.
if /i "%INSTALL_MODE%"=="LOCAL" rmdir /s /q "%LOCAL_WHEELHOUSE%"

cd /d "%ROOT%"
if not exist model_cache\       mkdir model_cache
if not exist model_archive\     mkdir model_archive
if not exist models\            mkdir models
if not exist training_data\     mkdir training_data
if not exist web_training_data\ mkdir web_training_data
if not exist logs\              mkdir logs
echo  Directories verified.

if not exist "%ROOT%config.json" (
    "%VENV_DIR%\Scripts\python.exe" -c "import json,pathlib; pathlib.Path('config.json').write_text(json.dumps({'host':'0.0.0.0','port':8080,'model':'Qwen/Qwen2.5-0.5B-Instruct','default_max_tokens':1024,'hard_max_tokens':8192,'default_priority':5,'default_tokens_per_min':6000,'guest_priority':10,'guest_tokens_per_min':1500,'allow_self_registration':True,'auto_discover_peers':True,'share_models_with_peers':True,'machine_role':'master'}, indent=2))"
    echo  Created default config.json.
)

echo V9-built > "%BUILD_MARKER%"

echo.
echo  =====================================================
echo   Setup complete!
echo.
echo   Run START.bat to launch Terminal AI.
echo   Nothing else is installed on this machine beyond what's
echo   listed in requirements.txt, inside its own venv\ folder.
echo  =====================================================
echo.
pause
endlocal
exit /b 0

:LocalDependenciesMissing
if defined LOCAL_WHEELHOUSE if exist "%LOCAL_WHEELHOUSE%" rmdir /s /q "%LOCAL_WHEELHOUSE%"
echo.
echo  [ERROR] The local dependency search or setup could not complete.
echo          Packages may be missing, or local search setup may have failed.
echo  Choose whether to search the computer again or download dependencies.
goto :ChooseInstallMode

:PipInstall
if /i "%INSTALL_MODE%"=="LOCAL" (
    "%VENV_DIR%\Scripts\python.exe" -m pip install --no-index --find-links "%LOCAL_WHEELHOUSE%" %*
) else (
    "%VENV_DIR%\Scripts\python.exe" -m pip install %*
)
exit /b %errorlevel%

:SetupLocalSearch
call %PYTHON_COMMAND% -c "import os; from pathlib import Path; p=Path(os.environ['VENV_DIR'])/'pyvenv.cfg'; lines=p.read_text().splitlines(); lines=[line for line in lines if not line.lower().startswith('include-system-site-packages')]; lines.append('include-system-site-packages = true'); p.write_text('\n'.join(lines)+'\n')"
if errorlevel 1 (
    echo  [ERROR] Could not enable locally installed Python packages.
    exit /b 1
)
if not defined TEMP (
    echo  [ERROR] The Windows TEMP folder is not defined.
    exit /b 1
)
set "LOCAL_WHEELHOUSE=%TEMP%\TerminalAI-wheelhouse-%RANDOM%"
mkdir "%LOCAL_WHEELHOUSE%" >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Could not create temporary package search folder: "%LOCAL_WHEELHOUSE%"
    exit /b 1
)
set "PIP_CACHE_DIR="
for /f "delims=" %%D in ('"%VENV_DIR%\Scripts\python.exe" -m pip cache dir 2^>nul') do set "PIP_CACHE_DIR=%%D"
if defined PIP_CACHE_DIR (
    if exist "%PIP_CACHE_DIR%" (
        for /r "%PIP_CACHE_DIR%" %%F in (*.whl) do copy /y "%%F" "%LOCAL_WHEELHOUSE%\" >nul
    )
)
echo  Searching installed packages and the local pip wheel cache...
exit /b 0
