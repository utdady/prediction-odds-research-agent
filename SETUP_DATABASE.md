# Database Setup Instructions

The pipelines require a PostgreSQL database. Here are several ways to set it up:

## Option 1: Use Docker (Recommended)

If you have Docker installed:

```bash
docker compose up -d
```

This will automatically create the database with the correct settings.

## Option 2: Manual Setup with psql

1. **Find your PostgreSQL installation:**
   - Common locations:
     - `C:\Program Files\PostgreSQL\<version>\bin`
     - `C:\Program Files (x86)\PostgreSQL\<version>\bin`
   - Or check: `Get-Command psql` in PowerShell

2. **Run the setup script:**
   ```bash
   psql -U postgres -f setup_database.sql
   ```
   
   If that doesn't work, try:
   ```bash
   psql -U <your-windows-username> -f setup_database.sql
   ```

3. **Or run the SQL commands manually:**
   ```sql
   CREATE USER pm WITH PASSWORD 'pm';
   CREATE DATABASE pm_research OWNER pm;
   GRANT ALL PRIVILEGES ON DATABASE pm_research TO pm;
   ```

## Option 3: Use pgAdmin

1. Open pgAdmin
2. Connect to your PostgreSQL server
3. Right-click on "Databases" → "Create" → "Database"
   - Name: `pm_research`
   - Owner: Create a new user `pm` with password `pm`, or select existing
4. If creating a new user:
   - Right-click on "Login/Group Roles" → "Create" → "Login/Group Role"
   - Name: `pm`
   - Password: `pm`
   - Privileges: Grant all necessary permissions

## Option 4: Find Your PostgreSQL Credentials

If you're not sure what username/password to use:

1. **Check pg_hba.conf** (usually in PostgreSQL data directory):
   - Look for authentication methods
   - `trust` means no password needed
   - `md5` or `scram-sha-256` means password required

2. **Common default credentials:**
   - Username: `postgres`
   - Password: The one you set during PostgreSQL installation
   - Or: Your Windows username (if using Windows authentication)

3. **Reset postgres password** (if needed):
   ```sql
   ALTER USER postgres WITH PASSWORD 'your_new_password';
   ```

## After Setup

Once the database is set up, run the migrations:

```bash
$env:PYTHONPATH="src"; .\.venv_win\Scripts\python.exe -m alembic upgrade head
```

Then you can run the pipelines:

```bash
$env:PYTHONPATH="src"; .\.venv_win\Scripts\python.exe -m pipelines.run_all
```


