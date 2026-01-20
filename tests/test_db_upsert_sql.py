import pytest

from pm_agent.schemas import NormalizedMarket


@pytest.mark.asyncio
async def test_market_schema_validation():
    m = NormalizedMarket(venue_id="kalshi", market_id="m1", title="t", status="active", raw={})
    assert m.market_id == "m1"

