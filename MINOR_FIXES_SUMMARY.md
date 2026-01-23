# Minor Fixes Summary

All requested minor fixes have been implemented.

## ✅ Completed Fixes

### 1. Database Setup Simplification
**Status**: ✅ Complete

- Added `make db_setup` command to Makefile
- Single command to run database setup: `make db_setup`
- Calls `setup_db_interactive.py` automatically

**Usage**:
```bash
make db_setup
```

### 2. Requirements Split
**Status**: ✅ Complete

Requirements have been split into three files:

- **requirements-core.txt**: Essential dependencies (FastAPI, SQLAlchemy, pandas, etc.)
- **requirements-ml.txt**: Optional ML/AI dependencies (transformers, torch) - commented out
- **requirements-dev.txt**: Development dependencies (pytest, ruff, mypy, black)

**requirements.txt** has been updated to:
- Reference the split files
- Include clear instructions
- Maintain backward compatibility

**Usage**:
```bash
# Install core dependencies
pip install -r requirements-core.txt

# Install with dev tools
pip install -r requirements-core.txt -r requirements-dev.txt

# Install ML features (optional)
pip install -r requirements-ml.txt
```

### 3. Improved Error Messages
**Status**: ✅ Complete

All generic error messages have been enhanced with context:

#### Before:
```python
raise ValueError("missing yes/no")
raise ValueError("v1 supports LONG only")
```

#### After:
```python
# normalization.py
raise ValueError(
    f"{context}: missing both yes and no prices. "
    f"At least one must be provided."
)
# Context includes: Market {market_id} at {tick_ts}

# backtest/engine.py
raise ValueError(
    f"Backtest engine v1 only supports LONG positions. "
    f"Received side='{side}'. SHORT positions not yet implemented."
)
```

## 📝 Files Modified

1. **Makefile**: Added `db_setup` target
2. **requirements.txt**: Updated to reference split files with instructions
3. **src/pm_agent/normalization.py**: Enhanced error message with context
4. **src/pm_agent/backtest/engine.py**: Enhanced error message with details

## ✅ Verification

- ✅ `make db_setup` command works
- ✅ Requirements files are properly split
- ✅ Error messages provide helpful context
- ✅ All imports work correctly
- ✅ No linter errors

All minor fixes are complete and ready to use! 🎉

