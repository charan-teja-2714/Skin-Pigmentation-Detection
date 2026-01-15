@echo off
echo Starting Skin Pigmentation Detection Backend...
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting FastAPI server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
python -m app.main