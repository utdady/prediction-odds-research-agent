-- PostgreSQL Database Setup Script
-- Run this script as a PostgreSQL superuser (usually 'postgres' or your Windows username)
-- 
-- To run this script:
--   1. Open Command Prompt or PowerShell
--   2. Find your PostgreSQL bin directory (common locations):
--      - C:\Program Files\PostgreSQL\<version>\bin
--      - C:\Program Files (x86)\PostgreSQL\<version>\bin
--   3. Run: psql -U postgres -f setup_database.sql
--      (or use your PostgreSQL admin username)
--
-- Alternatively, use pgAdmin or any PostgreSQL client to run these commands

-- Create the user if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'pm') THEN
        CREATE USER pm WITH PASSWORD 'pm';
        RAISE NOTICE 'User pm created';
    ELSE
        RAISE NOTICE 'User pm already exists';
    END IF;
END
$$;

-- Create the database if it doesn't exist
SELECT 'CREATE DATABASE pm_research OWNER pm'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'pm_research')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE pm_research TO pm;

-- Connect to the new database and grant schema privileges
\c pm_research
GRANT ALL ON SCHEMA public TO pm;

\echo 'Database setup complete!'
\echo 'Database: pm_research'
\echo 'User: pm'
\echo 'Password: pm'


