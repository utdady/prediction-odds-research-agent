# Agent Orchestrator Usage

## What is the Agent?

The **Agent Orchestrator** (`agent/orchestrator.py`) is the main coordinator that:
- Runs all pipelines in sequence
- Manages state persistence (tracks what's been run)
- Prevents unnecessary recomputation
- Supports one-time or scheduled execution

## How to Run the Agent

### Option 1: Run Once (Recommended)

```powershell
$env:PYTHONPATH="src"
python -m agent.orchestrator --mode once
```

This will:
1. Load current state from database
2. Run all pipelines (`pipelines.run_all`)
3. Save updated state to database

### Option 2: Scheduled Mode

```powershell
$env:PYTHONPATH="src"
python -m agent.orchestrator --mode scheduled --interval-sec 3600
```

This runs continuously, executing all pipelines every hour (3600 seconds).

### Option 3: Via Dashboard

1. Open the dashboard: http://localhost:8501
2. Go to the **"Agent"** tab
3. Click **"Run Agent Once"**

## Agent State Management

The agent tracks state in the `orchestrator_state` table:

- **component**: Pipeline component name (markets, ticks, features, model, backtest)
- **last_run_at**: Timestamp of last successful run
- **is_dirty**: Whether component needs recomputation
- **run_count**: Number of times component has run
- **failure_count**: Number of failures

## Viewing Agent State

### Via Dashboard
- Go to **"Agent"** tab
- See current state in the table

### Via Database Query
```sql
SELECT component, last_run_at, is_dirty, run_count, failure_count
FROM orchestrator_state
ORDER BY component;
```

## Agent vs Direct Pipeline Execution

**Using Agent (Recommended):**
- ✅ State management
- ✅ Prevents duplicate work
- ✅ Tracks execution history
- ✅ Supports scheduling

**Direct Pipeline Execution:**
- ✅ Simpler for testing
- ✅ No state overhead
- ❌ No state tracking
- ❌ May recompute unnecessarily

## Example Workflow

```powershell
# 1. Run agent once to process all data
$env:PYTHONPATH="src"
python -m agent.orchestrator --mode once

# 2. Check state
# (via dashboard or SQL query)

# 3. Run again (agent will use state to optimize)
python -m agent.orchestrator --mode once
```

## Troubleshooting

### Agent not found
Make sure `PYTHONPATH` is set:
```powershell
$env:PYTHONPATH="src"
```

### State table missing
Run migrations:
```powershell
$env:PYTHONPATH="src"
alembic upgrade head
```

### Agent hangs
Check if pipelines are running correctly:
```powershell
python -m pipelines.run_all
```

## Architecture

```
agent/orchestrator.py
    ├── load_state()      # Load from database
    ├── save_state()      # Save to database
    ├── run_once()        # Run all pipelines once
    └── run_scheduled()   # Run continuously
            │
            └──> pipelines.run_all()
                    ├── ingest_markets()
                    ├── ingest_ticks()
                    ├── build_features()
                    ├── train_model()
                    ├── run_inference()
                    ├── run_backtest()
                    └── publish_artifacts()
```

The agent is the **recommended way** to run the full pipeline system!

