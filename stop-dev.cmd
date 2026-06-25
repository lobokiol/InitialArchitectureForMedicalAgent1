@echo off
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dev-services.ps1" -Action stop
exit /b %ERRORLEVEL%
