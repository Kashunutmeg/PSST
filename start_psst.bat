@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Run the following to set up PSST:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

:: Admin check — soft warning only (Python handles non-admin gracefully)
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [NOTE] Running without admin -- global hotkeys may not work in all windows.
)

call ".venv\Scripts\activate.bat"
python -m psst %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [PSST exited with an error - see above for details.]
    pause
)
