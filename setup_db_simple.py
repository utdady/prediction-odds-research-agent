"""Simple script to set up PostgreSQL database - tries multiple connection methods."""
from __future__ import annotations

import sys
import os

try:
    import psycopg
except ImportError:
    print("Error: psycopg is required.")
    sys.exit(1)

DB_HOST = "localhost"
DB_PORT = 5432
TARGET_DB = "pm_research"
TARGET_USER = "pm"
TARGET_PASSWORD = "pm"

# Try these connection methods in order
CONNECTION_ATTEMPTS = [
    ("postgres", ""),  # postgres user, no password
    ("postgres", "postgres"),  # postgres user, password "postgres"
    (os.getenv("USERNAME", ""), ""),  # Windows username, no password
    (os.getenv("USER", ""), ""),  # Alternative username env var
]

def setup_database(conn: psycopg.Connection):
    """Set up database and user."""
    conn.autocommit = True
    with conn.cursor() as cur:
        # Create user
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (TARGET_USER,))
        if not cur.fetchone():
            cur.execute(f"CREATE USER {TARGET_USER} WITH PASSWORD %s", (TARGET_PASSWORD,))
            print(f"[OK] Created user '{TARGET_USER}'")
        else:
            print(f"[OK] User '{TARGET_USER}' already exists")
        
        # Create database
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TARGET_DB,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE {TARGET_DB} OWNER {TARGET_USER}')
            print(f"[OK] Created database '{TARGET_DB}'")
        else:
            print(f"[OK] Database '{TARGET_DB}' already exists")
            cur.execute(f'ALTER DATABASE {TARGET_DB} OWNER TO {TARGET_USER}')
        
        # Grant privileges
        cur.execute(f'GRANT ALL PRIVILEGES ON DATABASE {TARGET_DB} TO {TARGET_USER}')
        print("[OK] Privileges granted")
    
    print(f"\n[SUCCESS] Database setup complete!")
    print(f"  Database: {TARGET_DB}")
    print(f"  User: {TARGET_USER}")
    print(f"  Password: {TARGET_PASSWORD}")

def main():
    print("PostgreSQL Database Setup")
    print("=" * 40)
    
    for username, password in CONNECTION_ATTEMPTS:
        if not username:
            continue
            
        try:
            if password:
                conn_str = f"postgresql://{username}:{password}@{DB_HOST}:{DB_PORT}/postgres"
            else:
                conn_str = f"postgresql://{username}@{DB_HOST}:{DB_PORT}/postgres"
            
            print(f"\nTrying: {username}" + (f" (password: {password})" if password else " (no password)"))
            conn = psycopg.connect(conn_str)
            print(f"[OK] Connected as '{username}'!")
            setup_database(conn)
            conn.close()
            return True
        except psycopg.OperationalError as e:
            if "password authentication failed" in str(e).lower():
                print(f"  [FAILED] Wrong password")
            elif "role" in str(e).lower() and "does not exist" in str(e).lower():
                print(f"  [FAILED] User '{username}' does not exist")
            else:
                print(f"  [FAILED] {e}")
            continue
        except Exception as e:
            print(f"  [FAILED] {e}")
            continue
    
    print("\n[ERROR] Could not connect with any automatic method.")
    print("\nPlease set up the database manually:")
    print("1. Open pgAdmin or any PostgreSQL client")
    print("2. Connect to your PostgreSQL server")
    print("3. Run these SQL commands:")
    print(f"   CREATE USER {TARGET_USER} WITH PASSWORD '{TARGET_PASSWORD}';")
    print(f"   CREATE DATABASE {TARGET_DB} OWNER {TARGET_USER};")
    print(f"   GRANT ALL PRIVILEGES ON DATABASE {TARGET_DB} TO {TARGET_USER};")
    print("\nOr use the SQL file: setup_database.sql")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


