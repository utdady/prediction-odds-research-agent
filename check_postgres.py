"""Check PostgreSQL installation and provide connection info."""
from __future__ import annotations

import os
import sys
import subprocess

print("PostgreSQL Installation Check")
print("=" * 50)

# Check if PostgreSQL is running
print("\n1. Checking if PostgreSQL is running...")
try:
    result = subprocess.run(
        ["netstat", "-ano"], 
        capture_output=True, 
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    )
    if ":5432" in result.stdout:
        print("   [OK] PostgreSQL is running on port 5432")
    else:
        print("   [WARNING] PostgreSQL doesn't appear to be running on port 5432")
except Exception as e:
    print(f"   [ERROR] Could not check: {e}")

# Check common installation paths
print("\n2. Checking common PostgreSQL installation paths...")
common_paths = [
    r"C:\Program Files\PostgreSQL",
    r"C:\Program Files (x86)\PostgreSQL",
    os.path.expanduser(r"~\AppData\Local\Programs\PostgreSQL"),
]

found_installations = []
for base_path in common_paths:
    if os.path.exists(base_path):
        print(f"   [FOUND] {base_path}")
        # Look for version directories
        try:
            versions = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            for version in versions:
                full_path = os.path.join(base_path, version)
                bin_path = os.path.join(full_path, "bin")
                if os.path.exists(bin_path):
                    psql_path = os.path.join(bin_path, "psql.exe")
                    if os.path.exists(psql_path):
                        print(f"      -> Version: {version}")
                        print(f"      -> psql.exe: {psql_path}")
                        found_installations.append((version, psql_path))
        except Exception as e:
            print(f"      [ERROR] Could not list: {e}")

if not found_installations:
    print("   [NOT FOUND] PostgreSQL not found in common locations")

# Check for pgAdmin
print("\n3. Checking for pgAdmin...")
pgadmin_paths = [
    r"C:\Program Files\pgAdmin 4",
    r"C:\Program Files (x86)\pgAdmin 4",
    os.path.expanduser(r"~\AppData\Local\Programs\pgAdmin 4"),
    os.path.expanduser(r"~\AppData\Roaming\pgAdmin"),
]

pgadmin_found = False
for path in pgadmin_paths:
    if os.path.exists(path):
        print(f"   [FOUND] {path}")
        pgadmin_found = True
        # Look for pgAdmin executable
        for root, dirs, files in os.walk(path):
            if "pgAdmin4.exe" in files:
                print(f"      -> Executable: {os.path.join(root, 'pgAdmin4.exe')}")
                break

if not pgadmin_found:
    print("   [NOT FOUND] pgAdmin not found in common locations")
    print("   You can download it from: https://www.pgadmin.org/download/")

# Instructions
print("\n" + "=" * 50)
print("NEXT STEPS:")
print("=" * 50)

if found_installations:
    version, psql_path = found_installations[0]
    print(f"\n1. To use command line (psql):")
    print(f"   Navigate to: {os.path.dirname(psql_path)}")
    print(f"   Run: psql.exe -U postgres")
    print(f"   (It will prompt for your PostgreSQL password)")
    print(f"\n   Or run the setup script:")
    print(f"   psql.exe -U postgres -f setup_database.sql")

print("\n2. To use pgAdmin (GUI - Recommended):")
if pgadmin_found:
    print("   - Open pgAdmin from Start Menu")
else:
    print("   - Download from: https://www.pgadmin.org/download/")
    print("   - Install and open pgAdmin")
print("   - Connect to your PostgreSQL server")
print("   - Create database 'pm_research' with owner 'pm'")

print("\n3. If you don't remember your PostgreSQL password:")
print("   - Check if you wrote it down during installation")
print("   - Try common passwords: 'postgres', 'admin', your Windows password")
print("   - Or reset it using pgAdmin or command line")

print("\n4. Once database is set up, run:")
print("   $env:PYTHONPATH='src'; .\\.venv_win\\Scripts\\python.exe -m alembic upgrade head")


