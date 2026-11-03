@echo off
setlocal
pushd "%~dp0.."
if errorlevel 1 exit /b 1

if exist ".venv\Scripts\python.exe" goto install
py -3 --version >nul 2>&1
if errorlevel 1 goto use_python
py -3 -m venv ".venv"
if errorlevel 1 goto failed
goto install

:use_python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Install Python 3 and enable its launcher or add Python to PATH.
    goto failed
)
python -m venv ".venv"
if errorlevel 1 goto failed

:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto failed

echo.
echo Setup complete. From the repository root, launch Jupyter with:
echo   .venv\Scripts\python.exe -m notebook
echo In VS Code, select .venv\Scripts\python.exe as the notebook kernel.
popd
exit /b 0

:failed
echo ERROR: Environment setup failed. See the message above for details.
popd
exit /b 1
