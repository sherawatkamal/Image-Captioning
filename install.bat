@echo off
echo Setting up Deep Learning ResNet Project Environment...
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher and add it to PATH.
    pause
    exit /b 1
)

echo Python detected ✓

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing project dependencies...
pip install -r requirements.txt

echo.
echo Installation completed successfully! 🎉
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the Jupyter notebooks:
echo   jupyter notebook
echo.
echo To deactivate the virtual environment:
echo   deactivate
echo.
pause
