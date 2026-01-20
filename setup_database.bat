@echo off
REM Setup PostgreSQL database for prediction-odds-research-agent

echo ========================================
echo PostgreSQL Database Setup
echo ========================================
echo.

REM Find psql.exe (avoid hardcoding PostgreSQL version)
set "PSQL_EXE="

REM Try common install location(s)
for /d %%D in ("%ProgramFiles%\PostgreSQL\*") do (
  if exist "%%~fD\bin\psql.exe" (
    set "PSQL_EXE=%%~fD\bin\psql.exe"
  )
)

REM Fallback: try PATH
for /f "delims=" %%F in ('where psql.exe 2^>nul') do (
  set "PSQL_EXE=%%F"
  goto :found
)

:found
if "%PSQL_EXE%"=="" (
  echo [ERROR] Could not find psql.exe.
  echo - Check that PostgreSQL is installed
  echo - Expected under: %ProgramFiles%\PostgreSQL\^\<version^\>\bin\psql.exe
  echo - Or add PostgreSQL bin folder to PATH
  echo.
  pause
  exit /b 1
)

echo Running database setup script...
echo You will be prompted for your PostgreSQL password.
echo.

"%PSQL_EXE%" -U postgres -f "%~dp0setup_database.sql"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Database setup complete!
    echo.
    echo Next steps:
    echo 1. Run migrations: alembic upgrade head
    echo 2. Run pipelines: python -m pipelines.run_all
) else (
    echo.
    echo [ERROR] Setup failed. Please check your PostgreSQL password.
    echo.
    echo If you don't remember your password, try:
    echo - Common passwords: postgres, admin, or your Windows password
    echo - Or use pgAdmin to reset it
)

pause


