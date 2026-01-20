"""Interactive script to set up the PostgreSQL database."""
from __future__ import annotations

import sys
import getpass

try:
    import psycopg
except ImportError:
    print("Error: psycopg is required. Install it with: pip install psycopg[binary]")
    sys.exit(1)

DB_HOST = "localhost"
DB_PORT = 5432
TARGET_DB = "pm_research"
TARGET_USER = "pm"
TARGET_PASSWORD = "pm"

def try_connection(username: str | None, password: str | None) -> psycopg.Connection | None:
    """Try to connect to PostgreSQL."""
    try:
        if username is None:
            # Try Windows authentication
            conn_string = f"postgresql:///{DB_HOST}:{DB_PORT}/postgres"
        elif password:
            conn_string = f"postgresql://{username}:{password}@{DB_HOST}:{DB_PORT}/postgres"
        else:
            conn_string = f"postgresql://{username}@{DB_HOST}:{DB_PORT}/postgres"
        
        return psycopg.connect(conn_string)
    except Exception:
        return None

def setup_database(conn: psycopg.Connection):
    """Set up database and user using an existing connection."""
    conn.autocommit = True
    with conn.cursor() as cur:
        # Check if user exists
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (TARGET_USER,))
        user_exists = cur.fetchone()
        
        if not user_exists:
            print(f"Creating user '{TARGET_USER}'...")
            cur.execute(f"CREATE USER {TARGET_USER} WITH PASSWORD %s", (TARGET_PASSWORD,))
            print(f"[OK] User '{TARGET_USER}' created")
        else:
            print(f"[OK] User '{TARGET_USER}' already exists")
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TARGET_DB,))
        db_exists = cur.fetchone()
        
        if not db_exists:
            print(f"Creating database '{TARGET_DB}'...")
            cur.execute(f'CREATE DATABASE {TARGET_DB} OWNER {TARGET_USER}')
            print(f"[OK] Database '{TARGET_DB}' created")
        else:
            print(f"[OK] Database '{TARGET_DB}' already exists")
            cur.execute(f'ALTER DATABASE {TARGET_DB} OWNER TO {TARGET_USER}')
        
        # Grant privileges
        print(f"Granting privileges to '{TARGET_USER}'...")
        cur.execute(f'GRANT ALL PRIVILEGES ON DATABASE {TARGET_DB} TO {TARGET_USER}')
        print("[OK] Privileges granted")
    
    print("\n[OK] Database setup complete!")
    print(f"  Database: {TARGET_DB}")
    print(f"  User: {TARGET_USER}")
    print(f"  Password: {TARGET_PASSWORD}")

def main():
    print("PostgreSQL Database Setup")
    print("=" * 40)
    print(f"\nWe need to connect to PostgreSQL as a superuser to create:")
    print(f"  - Database: {TARGET_DB}")
    print(f"  - User: {TARGET_USER}")
    print(f"  - Password: {TARGET_PASSWORD}")
    print("\nLet's try to connect...\n")
    
    # Try common methods first
    print("Trying common connection methods...")
    
    # Try 1: postgres user with no password
    print("1. Trying 'postgres' user with no password...")
    conn = try_connection("postgres", None)
    if conn:
        print("   [OK] Connected!")
        setup_database(conn)
        conn.close()
        return True
    
    # Try 2: Windows authentication
    print("2. Trying Windows authentication...")
    conn = try_connection(None, None)
    if conn:
        print("   [OK] Connected!")
        setup_database(conn)
        conn.close()
        return True
    
    # Try 3: Interactive prompts
    print("\nAutomatic connection failed. Let's try manual entry...")
    print("\nCommon PostgreSQL superuser names:")
    print("  - 'postgres' (most common)")
    print("  - Your Windows username")
    print("  - 'admin' or 'administrator'")
    
    for attempt in range(3):
        username = input(f"\nEnter PostgreSQL superuser name (attempt {attempt + 1}/3, or 'skip' to exit): ").strip()
        if username.lower() == 'skip':
            break
        
        if not username:
            username = "postgres"
        
        password = getpass.getpass(f"Enter password for '{username}' (or press Enter for no password): ").strip()
        
        print(f"\nTrying to connect as '{username}'...")
        conn = try_connection(username, password if password else None)
        if conn:
            print("   [OK] Connected!")
            setup_database(conn)
            conn.close()
            return True
        else:
            print("   [FAILED] Could not connect. Check your credentials.")
    
    print("\n[ERROR] Could not connect to PostgreSQL.")
    print("\nAlternative options:")
    print("1. Use pgAdmin to run the SQL commands in 'setup_database.sql'")
    print("2. Find your PostgreSQL bin directory and run:")
    print("   psql -U <your-username> -f setup_database.sql")
    print("3. Check SETUP_DATABASE.md for more detailed instructions")
    return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


