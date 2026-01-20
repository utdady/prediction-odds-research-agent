"""Script to set up the PostgreSQL database and user."""
from __future__ import annotations

import sys

try:
    import psycopg
except ImportError:
    print("Error: psycopg is required. Install it with: pip install psycopg[binary]")
    sys.exit(1)

# Try to connect as postgres superuser (default user)
# You may need to adjust these credentials based on your PostgreSQL setup
DB_HOST = "localhost"
DB_PORT = 5432
SUPERUSER_OPTIONS = [
    ("postgres", ""),  # Try postgres user with no password
    ("postgres", "postgres"),  # Try postgres user with password "postgres"
    (None, None),  # Try Windows authentication (current user)
]

TARGET_DB = "pm_research"
TARGET_USER = "pm"
TARGET_PASSWORD = "pm"

def setup_database():
    """Create database and user if they don't exist."""
    # Try different connection methods
    for superuser, password in SUPERUSER_OPTIONS:
        try:
            if superuser is None:
                # Try Windows authentication (trust method)
                conn_string = f"postgresql:///{DB_HOST}:{DB_PORT}/postgres"
            elif password:
                conn_string = f"postgresql://{superuser}:{password}@{DB_HOST}:{DB_PORT}/postgres"
            else:
                conn_string = f"postgresql://{superuser}@{DB_HOST}:{DB_PORT}/postgres"
            
            print(f"Trying to connect as '{superuser or 'Windows user'}'...")
            with psycopg.connect(conn_string) as conn:
                return _setup_with_connection(conn, superuser or "current_user")
        except psycopg.OperationalError as e:
            if "password authentication failed" in str(e).lower():
                continue  # Try next option
            elif "connection refused" in str(e).lower() or "could not connect" in str(e).lower():
                print(f"\n[ERROR] Could not connect to PostgreSQL at {DB_HOST}:{DB_PORT}")
                print("  Make sure PostgreSQL is running.")
                return False
            else:
                continue  # Try next option
        except Exception as e:
            continue  # Try next option
    
    print("\n[ERROR] Could not connect to PostgreSQL with any of the common methods.")
    print("  Please run this script interactively or set up the database manually:")
    print(f"  - Create user: CREATE USER {TARGET_USER} WITH PASSWORD '{TARGET_PASSWORD}';")
    print(f"  - Create database: CREATE DATABASE {TARGET_DB} OWNER {TARGET_USER};")
    return False

def _setup_with_connection(conn, user_info):
    """Set up database and user using an existing connection."""
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Check if user exists
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (TARGET_USER,))
            user_exists = cur.fetchone()
            
            if not user_exists:
                print(f"Creating user '{TARGET_USER}'...")
                cur.execute(
                    f"CREATE USER {TARGET_USER} WITH PASSWORD %s",
                    (TARGET_PASSWORD,)
                )
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
                # Ensure user owns the database
                cur.execute(f'ALTER DATABASE {TARGET_DB} OWNER TO {TARGET_USER}')
            
            # Grant privileges
            print(f"Granting privileges to '{TARGET_USER}'...")
            cur.execute(f'GRANT ALL PRIVILEGES ON DATABASE {TARGET_DB} TO {TARGET_USER}')
            print("[OK] Privileges granted")
        
        print("\n[OK] Database setup complete!")
        print(f"  Database: {TARGET_DB}")
        print(f"  User: {TARGET_USER}")
        print(f"  Password: {TARGET_PASSWORD}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Error during setup: {e}")
        return False

if __name__ == "__main__":
    print("PostgreSQL Database Setup")
    print("=" * 40)
    success = setup_database()
    sys.exit(0 if success else 1)

