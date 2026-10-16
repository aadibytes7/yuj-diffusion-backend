@echo off
title Share YUJ Diffusion Online
cd /d "%~dp0"
echo ========================================================
echo Generating a public link for your local GPU diffuser...
echo Share the generated URL with anyone on the internet!
echo Keep this window open while sharing.
echo ========================================================
npx --yes localtunnel --port 8000
pause
