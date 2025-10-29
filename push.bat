@echo off
REM 一键上传（SSH 版，Windows BAT）
REM 第一次请先设置 origin： git remote add origin git@github.com:你的用户名/你的仓库.git

REM 1) 初始化/切到 main
git rev-parse --is-inside-work-tree >NUL 2>&1
if errorlevel 1 (
  echo 初始化 Git 仓库...
  git init
)
git branch -M main 2>NUL

REM 2) 提示远程地址
git remote get-url origin >NUL 2>&1
if errorlevel 1 (
  echo [错误] 未检测到 origin 远程地址。
  echo 请先执行：git remote add origin git@github.com:你的用户名/你的仓库.git
  pause
  exit /b 1
)

REM 3) 可选：停止追踪 .idea
IF EXIST ".idea" (
  findstr /R "^\.idea/$" .gitignore >NUL 2>&1 || echo .idea/>>.gitignore
  git add .gitignore
  git rm -r --cached .idea >NUL 2>&1
)

REM 4) .gitignore（若不存在则创建一个简单版）
IF NOT EXIST ".gitignore" (
  > .gitignore echo __pycache__/
  >> .gitignore echo *.py[cod]
  >> .gitignore echo .venv/
  >> .gitignore echo venv/
  >> .gitignore echo env/
  >> .gitignore echo .ipynb_checkpoints/
  >> .gitignore echo .idea/
  >> .gitignore echo .vscode/
  git add .gitignore
)

REM 5) 提交与推送
git add -A
git diff --cached --quiet
IF ERRORLEVEL 1 (
  for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set TODAY=%%a-%%b-%%c
  for /f "tokens=1-2 delims=: " %%a in ('time /t') do set NOW=%%a:%%b
  git commit -m "chore: autosync %TODAY% %NOW%"
) ELSE (
  echo 没有变化需要提交。
)

REM 判断是否已有上游跟踪
git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >NUL 2>&1
IF ERRORLEVEL 1 (
  git push -u origin main
) ELSE (
  git push
)

echo ✅ 已推送到 origin/main
pause
