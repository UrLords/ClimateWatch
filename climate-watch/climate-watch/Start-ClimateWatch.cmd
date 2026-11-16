@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:8766 in your browser. Keep this window open.
python scripts\serve_prototype.py
pause
