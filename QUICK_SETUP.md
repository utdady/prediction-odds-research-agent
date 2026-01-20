# Quick Database Setup Guide

PostgreSQL is running but requires authentication. Here's the fastest way to set it up:

## Method 1: Using pgAdmin (Easiest - GUI)

1. **Open pgAdmin** (usually in Start Menu or Desktop)
2. **Connect to your PostgreSQL server** (you may need to enter your PostgreSQL password)
3. **Right-click on "Databases"** → **"Create"** → **"Database"**
   - Name: `pm_research`
   - Owner: Click the dropdown, then click the "+" to create a new user
     - In the new user dialog:
       - Name: `pm`
       - Password: `pm`
       - Can login: ✓ (checked)
       - Click "Save"
   - Click "Save" to create the database

4. **Done!** The database is ready.

## Method 2: Using Command Line (If you know your PostgreSQL password)

1. **Open PowerShell or Command Prompt**

2. **Find PostgreSQL bin directory** (common locations):
   ```
   C:\Program Files\PostgreSQL\15\bin
   C:\Program Files\PostgreSQL\16\bin
   C:\Program Files (x86)\PostgreSQL\15\bin
   ```

3. **Navigate to that directory** or add it to your PATH temporarily:
   ```powershell
   cd "C:\Program Files\PostgreSQL\16\bin"
   ```

4. **Run the setup script**:
   ```powershell
   .\psql.exe -U postgres -f "C:\Users\addyb\prediction-odds-research-agent\setup_database.sql"
   ```
   
   It will prompt for your PostgreSQL password. Enter the password you set when installing PostgreSQL.

5. **If that doesn't work**, try with your Windows username:
   ```powershell
   .\psql.exe -U addyb -f "C:\Users\addyb\prediction-odds-research-agent\setup_database.sql"
   ```

## Method 3: Manual SQL Commands

If you can connect to PostgreSQL (via pgAdmin or psql), run these commands:

```sql
-- Create the user
CREATE USER pm WITH PASSWORD 'pm';

-- Create the database
CREATE DATABASE pm_research OWNER pm;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE pm_research TO pm;

-- Connect to the new database
\c pm_research

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO pm;
```

## After Setup

Once the database is set up, run migrations:

```powershell
$env:PYTHONPATH="src"; .\.venv_win\Scripts\python.exe -m alembic upgrade head
```

Then run the pipelines:

```powershell
$env:PYTHONPATH="src"; .\.venv_win\Scripts\python.exe -m pipelines.run_all
```

## Need Help Finding Your PostgreSQL Password?

- Check if you wrote it down during installation
- If you forgot it, you may need to reset it or reinstall PostgreSQL
- Some installations use your Windows password for the postgres user


