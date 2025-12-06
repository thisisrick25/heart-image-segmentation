@echo off
REM Setup and Push to Kaggle Kernel
REM This script helps you push your code to Kaggle kernels

echo ========================================
echo Kaggle Kernel Setup and Push Script
echo ========================================
echo.

REM Check if Kaggle CLI is installed
where kaggle >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Kaggle CLI not found!
    echo Please install it with: pip install kaggle
    echo.
    echo After installation, you need to:
    echo 1. Go to https://www.kaggle.com/settings/account
    echo 2. Scroll to "API" section and click "Create New Token"
    echo 3. Place kaggle.json in: C:\Users\%USERNAME%\.kaggle\
    pause
    exit /b 1
)

echo Kaggle CLI detected!
echo.

REM Check if kernel-metadata.json exists
if not exist "kernel-metadata.json" (
    echo ERROR: kernel-metadata.json not found!
    echo Please ensure you're in the project directory.
    pause
    exit /b 1
)

echo Found kernel-metadata.json
echo.

REM Ask user what to do
echo What would you like to do?
echo 1. Push kernel to Kaggle (create or update)
echo 2. Check kernel status
echo 3. Pull latest kernel version
echo 4. Open kernel in browser
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Pushing kernel to Kaggle...
    echo This will upload: kaggle_train.py, requirements.txt, and kernel-metadata.json
    echo.
    kaggle kernels push
    echo.
    echo Kernel pushed successfully!
    echo You can view it at: https://www.kaggle.com/code/thisisrick25/heart-segmentation-unet
    echo.
) else if "%choice%"=="2" (
    echo.
    echo Checking kernel status...
    kaggle kernels status thisisrick25/heart-segmentation-unet
    echo.
) else if "%choice%"=="3" (
    echo.
    echo Pulling latest kernel version...
    kaggle kernels pull thisisrick25/heart-segmentation-unet
    echo.
) else if "%choice%"=="4" (
    echo.
    echo Opening kernel in browser...
    start https://www.kaggle.com/code/thisisrick25/heart-segmentation-unet
    echo.
) else (
    echo Invalid choice!
)

echo.
echo ========================================
echo Additional Commands:
echo ========================================
echo - View kernel output: kaggle kernels output thisisrick25/heart-segmentation-unet
echo - View all your kernels: kaggle kernels list --mine
echo.

pause
