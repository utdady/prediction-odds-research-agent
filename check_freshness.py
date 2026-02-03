"""Check data freshness."""
import pandas as pd
import sqlalchemy
from datetime import datetime, timezone

engine = sqlalchemy.create_engine('postgresql+psycopg://pm:pm@localhost:5432/pm_research')
now = datetime.now(timezone.utc)

# Check features
f = pd.read_sql('SELECT MAX(ts) as max_ts FROM features', engine).iloc[0]['max_ts']
# Check ticks
t = pd.read_sql('SELECT MAX(tick_ts) as max_ts FROM odds_ticks', engine).iloc[0]['max_ts']

print('=' * 60)
print('Data Freshness Check')
print('=' * 60)
print(f'Current time: {now}')
print(f'\nLatest feature timestamp: {f}')
print(f'Latest tick timestamp: {t}')

if f:
    h = (now - pd.to_datetime(f)).total_seconds() / 3600
    status = 'FRESH' if h < 1 else 'STALE' if h < 6 else 'VERY STALE'
    print(f'\nFeatures age: {h:.1f} hours - {status}')

if t:
    h = (now - pd.to_datetime(t)).total_seconds() / 3600
    status = 'FRESH' if h < 1 else 'STALE' if h < 6 else 'VERY STALE'
    print(f'Ticks age: {h:.1f} hours - {status}')

print('=' * 60)

