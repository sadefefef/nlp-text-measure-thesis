@echo off
REM Push this repository to GitHub.
REM Usage:  push-to-github.bat your-github-username [repo-name]
setlocal
if "%~1"=="" (
  echo Usage: push-to-github.bat ^<github-username^> [repo-name]
  exit /b 1
)
set GHUSER=%~1
if "%~2"=="" (set REPO=nlp-text-measure-thesis) else (set REPO=%~2)

cd /d "%~dp0"
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/%GHUSER%/%REPO%.git
echo Pushing to https://github.com/%GHUSER%/%REPO%.git ...
git -c http.sslBackend=openssl push -u origin main
echo.
echo Done. Open: https://github.com/%GHUSER%/%REPO%
endlocal
