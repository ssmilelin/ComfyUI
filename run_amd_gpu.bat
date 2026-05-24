@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ========================================
echo   ComfyUI - AMD GPU / ROCm
echo   http://127.0.0.1:8188
echo ========================================
echo.
echo FIRST START with 79 custom nodes may take 10-30 minutes.
echo Wait until you see: Starting server
echo Then open the URL above in your browser.
echo.

if not exist ".venv\Scripts\python.exe" goto NO_VENV

".venv\Scripts\python.exe" main.py --preview-method auto
goto END

:NO_VENV
echo ERROR: .venv\Scripts\python.exe not found
pause
exit /b 1

:END
echo.
echo ComfyUI stopped.
pause
