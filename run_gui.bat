@echo off
setlocal
cd /d %~dp0

set "PYTHON_EXE="
if defined VIRTUAL_ENV (
  if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
)
if not defined PYTHON_EXE if defined CONDA_PREFIX (
  if exist "%CONDA_PREFIX%\python.exe" set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
)
if not defined PYTHON_EXE (
  for /f "usebackq delims=" %%i in (`python -c "import sys; print(sys.executable)"`) do set "PYTHON_EXE=%%i"
)
if not defined PYTHON_EXE (
  for /f "usebackq delims=" %%i in (`python3 -c "import sys; print(sys.executable)"`) do set "PYTHON_EXE=%%i"
)
if not defined PYTHON_EXE (
  echo Failed to locate a Python interpreter.
  exit /b 1
)

"%PYTHON_EXE%" limix_gui.py
endlocal
