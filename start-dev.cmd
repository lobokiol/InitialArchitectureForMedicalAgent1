@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dev-services.ps1" %*
exit /b %ERRORLEVEL%
