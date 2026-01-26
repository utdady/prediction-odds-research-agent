# Quick Start Guide

## Prerequisites

- Python 3.11+
- PostgreSQL 18+ (or Docker)
- Git

## Step 1: Clone and Setup

```bash
# Clone the repository (if not already done)
git clone https://github.com/utdady/prediction-odds-research-agent.git
cd prediction-odds-research-agent

# Create virtual environment
python -m venv .venv_win

# Activate virtual environment
# On Windows PowerShell:
.\.venv_win\Scripts\Activate.ps1
# On Windows CMD:
.\.venv_win\Scripts\activate.bat
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements-core.txt
```

## Step 2: Database Setup

### Option A: Using Docker (Recommended)

```bash
# Start PostgreSQL and MLflow
docker compose up -d

# Wait a few seconds for services to start, then run migrations
make migrate
# Or manually:
alembic upgrade head
```

### Option B: Using Local PostgreSQL

```bash
# Make sure PostgreSQL is running
# Then run database setup
make db_setup
# Or manually:
python setup_db_interactive.py

# Run migrations
make migrate
```

## Step 3: Run the Application

### Option 1: Run Full Pipeline

```bash
# Set PYTHONPATH (Windows PowerShell)
$env:PYTHONPATH="src"

# Run all pipelines
python -m pipelines.run_all

# Or using make (if on Linux/Mac with make installed)
make run_all
```

### Option 2: Run Individual Pipelines

```bash
# Set PYTHONPATH first
$env:PYTHONPATH="src"

# Ingest markets
python -m pipelines.ingest_markets

# Ingest ticks
python -m pipelines.ingest_ticks

# Build features
python -m pipelines.build_features

# Train model
python -m pipelines.train_model

# Run inference (generate signals)
python -m pipelines.run_inference

# Run backtest
python -m pipelines.run_backtest

# Run walk-forward backtest
python -m pipelines.run_backtest_walkforward
```

### Option 3: Run API Server

```bash
# Set PYTHONPATH
$env:PYTHONPATH="src"

# Start API server
uvicorn src.app.api.main:app --reload --host 0.0.0.0 --port 8000

# Or using make
make api
```

Then visit: http://localhost:8000
API docs: http://localhost:8000/docs

### Option 4: Run Dashboard

```bash
# Set PYTHONPATH
$env:PYTHONPATH="src"

# Start Streamlit dashboard
streamlit run src.app.dashboard.Home --server.port 8501

# Or using make
make dashboard
```

Then visit: http://localhost:8501

## Step 4: View Results

### Dashboard
- Open http://localhost:8501
- View markets, signals, backtest results
- Use interactive backtesting tool
- Check advanced features (regime detection, arbitrage, sentiment)

### API
- Open http://localhost:8000/docs
- Interactive API documentation
- Test endpoints directly

## Common Commands

```bash
# Run tests
pytest -q

# Check database connection
python -c "from pm_agent.db import get_session; import asyncio; asyncio.run(get_session().__aenter__())"

# Generate enhanced mock data
python -m pm_agent.connectors.mock_generator

# Detect arbitrage opportunities
python -m pipelines.detect_arbitrage
```

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
# Windows: Check Services or Task Manager
# Or: pg_ctl status -D "C:\Program Files\PostgreSQL\18\data"

# Reset database password
psql -U postgres -c "ALTER USER pm WITH PASSWORD 'pm';"
```

### Import Errors
```bash
# Make sure PYTHONPATH is set
$env:PYTHONPATH="src"  # Windows PowerShell
export PYTHONPATH="src"  # Linux/Mac
```

### Port Already in Use
```bash
# Change port in command:
streamlit run src.app.dashboard.Home --server.port 8502
uvicorn src.app.api.main:app --port 8001
```

## Next Steps

1. **Explore Dashboard**: Open http://localhost:8501
2. **Run Full Pipeline**: `python -m pipelines.run_all`
3. **Check Results**: View signals and backtest metrics in dashboard
4. **Try Advanced Features**: Use regime detection, arbitrage scanner, sentiment analysis

## Full Example Workflow

```bash
# 1. Setup
python -m venv .venv_win
.\.venv_win\Scripts\Activate.ps1
pip install -r requirements-core.txt

# 2. Start database
docker compose up -d

# 3. Setup database
$env:PYTHONPATH="src"
make migrate

# 4. Run pipeline
python -m pipelines.run_all

# 5. Start dashboard
streamlit run src.app.dashboard.Home --server.port 8501

# 6. Open browser
# Visit http://localhost:8501
```

Enjoy! 🚀

