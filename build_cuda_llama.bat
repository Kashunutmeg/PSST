@echo off
REM ================================================================
REM  Build llama-cpp-python with CUDA support for PSST
REM
REM  PREREQUISITES:
REM    1. CUDA Toolkit 12.4 installed (nvcc must be in PATH)
REM    2. Run this from the VS Developer Command Prompt
REM       (Start > "Developer Command Prompt for VS 2022")
REM    3. The PSST venv must exist at .venv\
REM    4. A system CMake must be first on PATH (not Willow's bundled one)
REM
REM  This script:
REM    - Activates the venv
REM    - Uninstalls any existing llama-cpp-python
REM    - Purges pip cache
REM    - Builds llama-cpp-python 0.3.19 from source with CUDA enabled
REM      (newest known-good release that supports Qwen3.5
REM       ('qwen35' architecture) -- 0.3.16 is too old for Qwen3.5,
REM       and 0.3.20 (the current PyPI latest) has a broken llama.cpp
REM       submodule missing tools/mtmd/deprecation-warning.cpp)
REM    - Runs a quick import check
REM ================================================================

set LLAMA_CPP_VERSION=0.3.19

cd /d "%~dp0"

echo ============================================================
echo  PSST -- Build llama-cpp-python with CUDA
echo ============================================================

REM -- Check nvcc -----------------------------------------------
where nvcc >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvcc not found in PATH.
    echo         Install CUDA Toolkit 12.4 from:
    echo         https://developer.nvidia.com/cuda-12-4-0-download-archive
    echo         Then re-open the VS Developer Command Prompt and try again.
    pause
    exit /b 1
)
echo [OK] nvcc found:
nvcc --version | findstr /C:"release"

REM -- Check cl.exe ---------------------------------------------
where cl >nul 2>&1
if errorlevel 1 (
    echo [ERROR] cl.exe not found. Run this from the VS Developer Command Prompt.
    pause
    exit /b 1
)
echo [OK] cl.exe found

REM -- Check which CMake is first on PATH -----------------------
for /f "delims=" %%i in ('where cmake 2^>nul') do (
    set "CMAKE_PATH=%%i"
    goto :cmake_checked
)
:cmake_checked
if not defined CMAKE_PATH (
    echo [ERROR] cmake not found in PATH.
    echo         Install CMake from https://cmake.org/download/ and tick
    echo         "Add CMake to the system PATH for all users" during setup.
    pause
    exit /b 1
)
echo [OK] cmake found: %CMAKE_PATH%
echo %CMAKE_PATH% | findstr /I "WILLOW" >nul
if not errorlevel 1 (
    echo [ERROR] CMake is being picked up from the retired Project WILLOW directory.
    echo         Remove any entry containing WILLOW from your PATH environment variable
    echo         via "Edit the system environment variables", then open a NEW terminal
    echo         and re-run this script.
    pause
    exit /b 1
)

REM -- Activate venv --------------------------------------------
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] .venv not found. Create it first: python -m venv .venv
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
echo [OK] venv activated

REM -- Clean install --------------------------------------------
echo.
echo Uninstalling old llama-cpp-python...
pip uninstall -y llama-cpp-python 2>nul
pip cache purge

REM -- Set build flags ------------------------------------------
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=on
set PYTHONUTF8=1

echo.
echo Building llama-cpp-python==%LLAMA_CPP_VERSION% from source with CUDA...
echo   FORCE_CMAKE=1
echo   CMAKE_ARGS=%CMAKE_ARGS%
echo.
echo This will take several minutes. Watch for errors.
echo ============================================================

pip install llama-cpp-python==%LLAMA_CPP_VERSION% --force-reinstall --no-cache-dir --verbose

if errorlevel 1 (
    echo.
    echo [FAILED] Build failed. Check the output above for errors.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Build complete. Running quick verification...
echo ============================================================

python -c "from llama_cpp import llama_cpp; print('llama-cpp-python loaded OK, version:', __import__('llama_cpp').__version__)"

if errorlevel 1 (
    echo [WARNING] Import check failed. The DLL may have missing dependencies.
) else (
    echo [OK] llama-cpp-python imported successfully.
)

echo.
echo Next step: verify CUDA DLLs are present with:
echo   dir .venv\Lib\site-packages\llama_cpp\lib\ggml-cuda.dll
echo Then run PSST and watch for "offloaded X/Y layers to GPU" in the logs.
pause
