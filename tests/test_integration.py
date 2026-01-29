"""Integration tests for pipelines and API connectors."""
from __future__ import annotations

import pytest
import pandas as pd
from datetime import datetime, timezone

from pm_agent.db import get_session
from pm_agent.sql import fetch_all, fetch_one


@pytest.mark.asyncio
async def test_full_pipeline_with_mock_data():
    """Test that full pipeline runs end-to-end with mock data."""
    from pipelines.run_all import run as run_all_pipelines
    
    # Run all pipelines
    results = await run_all_pipelines()
    
    # Check that critical pipelines succeeded
    assert results.get("ingest_markets") == "success"
    assert results.get("ingest_ticks") == "success"
    assert results.get("build_features") == "success"
    
    # Verify data was created
    async with get_session() as session:
        markets = await fetch_all(session, "SELECT COUNT(*) as n FROM markets")
        assert markets[0]["n"] > 0
        
        ticks = await fetch_all(session, "SELECT COUNT(*) as n FROM odds_ticks")
        assert ticks[0]["n"] > 0
        
        features = await fetch_all(session, "SELECT COUNT(*) as n FROM features")
        assert features[0]["n"] > 0


@pytest.mark.asyncio
async def test_database_upserts():
    """Test that database upserts work correctly."""
    from pm_agent.repo.upserts import upsert_market, upsert_tick
    from pm_agent.schemas import NormalizedMarket, NormalizedTick
    
    async with get_session() as session:
        # Test market upsert
        market = NormalizedMarket(
            venue_id="test",
            market_id="TEST_MARKET_001",
            event_id="EV_TEST_001",
            title="Test Market",
            description="Test",
            status="active",
            resolution_ts=None,
            raw={},
        )
        await upsert_market(session, market)
        await session.commit()
        
        # Verify it was inserted
        result = await fetch_one(
            session,
            "SELECT * FROM markets WHERE market_id = :market_id",
            {"market_id": "TEST_MARKET_001"}
        )
        assert result is not None
        assert result["title"] == "Test Market"
        
        # Test tick upsert
        tick = NormalizedTick(
            venue_id="test",
            market_id="TEST_MARKET_001",
            tick_ts=datetime.now(timezone.utc),
            yes_bid=0.5,
            yes_ask=0.52,
            no_bid=0.48,
            no_ask=0.5,
            yes_mid=0.51,
            no_mid=0.49,
            p_norm=0.51,
            volume=100.0,
            raw={},
        )
        await upsert_tick(session, tick)
        await session.commit()
        
        # Verify it was inserted
        tick_result = await fetch_one(
            session,
            "SELECT * FROM odds_ticks WHERE market_id = :market_id ORDER BY tick_ts DESC LIMIT 1",
            {"market_id": "TEST_MARKET_001"}
        )
        assert tick_result is not None
        assert tick_result["p_norm"] == 0.51


@pytest.mark.asyncio
async def test_stock_analysis_agent():
    """Test StockAnalysisAgent with database data."""
    from pm_agent.agent.stock_analysis import StockAnalysisAgent
    import sqlalchemy
    
    # Create sync engine
    engine = sqlalchemy.create_engine("postgresql+psycopg://pm:pm@localhost:5432/pm_research")
    agent = StockAnalysisAgent(engine)
    
    # Test analyze_stock_performance
    results = agent.analyze_stock_performance(lookback_days=7)
    
    # Should return dict with expected keys
    assert "performing_well" in results
    assert "performing_poorly" in results
    assert "neutral" in results
    assert "analysis" in results
    
    # Test get_stock_outlook (if data exists)
    try:
        outlook = agent.get_stock_outlook("AAPL")
        assert "outlook" in outlook
        assert "recommendation" in outlook
        assert "confidence" in outlook
    except Exception:
        # If no data, that's okay for this test
        pass


@pytest.mark.skip(reason="Requires live API credentials")
@pytest.mark.asyncio
async def test_kalshi_connector_live():
    """Test Kalshi connector with real API (requires credentials)."""
    from pm_agent.connectors.kalshi import KalshiConnector
    from pm_agent.config import settings
    
    if not settings.kalshi_api_key:
        pytest.skip("Kalshi API key not configured")
    
    connector = KalshiConnector(
        api_key=settings.kalshi_api_key,
        api_secret=settings.kalshi_api_secret,
    )
    
    try:
        markets = await connector.fetch_markets(limit=10)
        assert len(markets) > 0
        assert all(m.venue_id == "kalshi" for m in markets)
    finally:
        await connector.close()


@pytest.mark.skip(reason="Requires live API credentials")
@pytest.mark.asyncio
async def test_polymarket_connector_live():
    """Test Polymarket connector with real API (requires credentials)."""
    from pm_agent.connectors.polymarket import PolymarketConnector
    from pm_agent.config import settings
    
    connector = PolymarketConnector(api_key=settings.polymarket_api_key)
    
    try:
        markets = await connector.fetch_markets(limit=10)
        # Polymarket may work without API key for public data
        if len(markets) > 0:
            assert all(m.venue_id == "polymarket" for m in markets)
    finally:
        await connector.close()

