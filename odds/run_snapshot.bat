@echo off
cd /d "%~dp0.."
"%~dp0..\venv\Scripts\python.exe" "%~dp0snapshot_lines.py"
exit /b %ERRORLEVEL%
