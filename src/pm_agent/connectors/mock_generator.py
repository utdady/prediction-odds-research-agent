"""Enhanced mock data generator with realistic scenarios."""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np


class MockDataGenerator:
    """Generates realistic mock data with various market conditions."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def generate_markets(
        self,
        venue_id: str,
        n_markets: int = 10,
        scenarios: list[Literal["trending", "choppy", "volatile", "low_liquidity", "failed"]] | None = None,
    ) -> list[dict]:
        """
        Generate markets with various scenarios.
        
        Args:
            venue_id: Venue identifier
            n_markets: Number of markets to generate
            scenarios: List of scenario types to include
        """
        if scenarios is None:
            scenarios = ["trending", "choppy", "volatile", "low_liquidity", "failed"]

        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
        markets = []

        for i in range(n_markets):
            ticker = tickers[i % len(tickers)]
            scenario = scenarios[i % len(scenarios)]
            market_id = f"{venue_id.upper()}_{ticker}_EVENT_{i+1:03d}"
            event_id = f"EV_{ticker}_{i+1:03d}"

            # Base market data
            base_date = datetime(2024, 1, 1) + timedelta(days=i * 2)
            resolution_date = base_date + timedelta(days=30)

            market = {
                "market_id": market_id,
                "event_id": event_id,
                "title": f"Will {ticker} close above $150 by {resolution_date.strftime('%Y-%m-%d')}?",
                "description": f"Prediction market for {ticker} stock price",
                "status": "active",
                "resolution_ts": resolution_date.isoformat() + "Z",
            }

            # Scenario-specific modifications
            if scenario == "failed":
                market["status"] = "failed"
            elif scenario == "low_liquidity":
                market["title"] += " [LOW LIQUIDITY]"

            markets.append(market)

        return markets

    def generate_ticks(
        self,
        venue_id: str,
        market_ids: list[str],
        base_date: datetime | None = None,
        days: int = 10,
        scenario: Literal["trending", "choppy", "volatile", "low_liquidity"] = "trending",
    ) -> list[dict]:
        """
        Generate ticks with realistic price movements.
        
        Args:
            venue_id: Venue identifier
            market_ids: List of market IDs to generate ticks for
            base_date: Starting date (defaults to 2024-01-01)
            days: Number of days of ticks
            scenario: Market condition scenario
        """
        if base_date is None:
            base_date = datetime(2024, 1, 1)

        all_ticks = []

        for market_id in market_ids:
            # Initial probability
            p0 = self.rng.uniform(0.3, 0.7)

            # Scenario-specific parameters
            if scenario == "trending":
                drift = self.rng.uniform(0.01, 0.03)  # Upward trend
                volatility = 0.02
                volume_base = 100
            elif scenario == "choppy":
                drift = 0.0  # No trend
                volatility = 0.015
                volume_base = 80
            elif scenario == "volatile":
                drift = 0.0
                volatility = 0.05  # High volatility
                volume_base = 150
            elif scenario == "low_liquidity":
                drift = self.rng.uniform(-0.01, 0.01)
                volatility = 0.02
                volume_base = 20  # Low volume
            else:
                drift = 0.0
                volatility = 0.02
                volume_base = 100

            # Generate time series
            p = p0
            for day in range(days):
                tick_date = base_date + timedelta(days=day)

                # Random walk with drift
                p += drift + self.rng.normal(0, volatility)
                p = np.clip(p, 0.01, 0.99)  # Keep in valid range

                # Generate bid/ask spread
                spread = 0.02 if scenario != "low_liquidity" else 0.05
                yes_bid = max(0.01, p - spread / 2)
                yes_ask = min(0.99, p + spread / 2)

                # Volume with some randomness
                volume = max(10, int(volume_base * self.rng.uniform(0.7, 1.3)))

                # Edge cases
                if scenario == "low_liquidity" and day % 3 == 0:
                    # Missing ticks (low liquidity)
                    continue

                tick = {
                    "market_id": market_id,
                    "tick_ts": tick_date.isoformat() + "Z",
                    "yes_bid": round(yes_bid, 4),
                    "yes_ask": round(yes_ask, 4),
                    "yes_mid": round(p, 4),
                    "no_bid": round(1 - yes_ask, 4),
                    "no_ask": round(1 - yes_bid, 4),
                    "no_mid": round(1 - p, 4),
                    "p_norm": round(p, 4),
                    "volume": volume,
                }

                all_ticks.append(tick)

        return all_ticks

    def generate_all_scenarios(self, output_dir: str = "data/mock") -> None:
        """Generate mock data files for all scenarios."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        venues = ["kalshi", "polymarket"]

        for venue in venues:
            # Generate markets
            markets = self.generate_markets(venue, n_markets=15)
            market_ids = [m["market_id"] for m in markets]

            # Generate ticks for each scenario
            scenarios = ["trending", "choppy", "volatile", "low_liquidity"]
            all_ticks = []

            for scenario in scenarios:
                scenario_ticks = self.generate_ticks(
                    venue,
                    market_ids[:4],  # 4 markets per scenario
                    scenario=scenario,
                    days=12,
                )
                all_ticks.extend(scenario_ticks)

            # Save files
            markets_file = output_path / f"{venue}_markets.json"
            ticks_file = output_path / f"{venue}_ticks.json"

            with open(markets_file, "w") as f:
                json.dump(markets, f, indent=2)

            with open(ticks_file, "w") as f:
                json.dump(all_ticks, f, indent=2)

            print(f"Generated {len(markets)} markets and {len(all_ticks)} ticks for {venue}")


if __name__ == "__main__":
    generator = MockDataGenerator(seed=42)
    generator.generate_all_scenarios()

