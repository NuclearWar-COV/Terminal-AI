@echo off

set "FOLDER=%~dp0"
if "%FOLDER:~-1%"=="\" set "FOLDER=%FOLDER:~0,-1%"

echo.
echo Folder to add: %FOLDER%
echo.

for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "CURPATH=%%B"

echo ;%CURPATH%; | find /I ";%FOLDER%;" >nul
if %errorlevel%==0 (
    echo Already in PATH. Nothing to do.
    exit /b 0
)

setx PATH "%CURPATH%;%FOLDER%" >nul
if %errorlevel%==0 (
    echo Successfully added to USER PATH.
) else (
    echo Failed to update PATH. Try running as Administrator if needed.
    exit /b 1
)

if /i not "%~1"=="/quiet" (
    echo.
    echo Open a NEW terminal for the change to take effect.
    echo.
    pause
)
exit /b 0
