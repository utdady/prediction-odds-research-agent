"""Cross-venue arbitrage detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pm_agent.schemas import Market

log = structlog.get_logger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity."""

    event_id: str
    venue_1: str
    venue_2: str
    prob_1: float
    prob_2: float
    spread: float
    expected_profit: float
    action: str  # e.g., "buy_low_sell_high"


class ArbitrageDetector:
    """Detects arbitrage opportunities between venues."""

    def __init__(self, min_spread: float = 0.03):  # 3% minimum spread
        self.min_spread = min_spread

    def find_arbitrage_opportunities(self, markets: list[dict]) -> list[ArbitrageOpportunity]:
        """
        Find same event priced differently across venues.
        
        Args:
            markets: List of market dicts with keys: event_id, venue_id, probability
        
        Returns:
            List of arbitrage opportunities
        """
        # Group markets by event
        event_groups: dict[str, list[dict]] = {}
        for market in markets:
            event_id = market.get("event_id")
            if not event_id:
                continue
            if event_id not in event_groups:
                event_groups[event_id] = []
            event_groups[event_id].append(market)

        arbitrage_opps = []

        for event_id, markets_list in event_groups.items():
            if len(markets_list) < 2:
                continue

            # Get probabilities from each venue
            venue_probs: dict[str, float] = {}
            for m in markets_list:
                venue_id = m.get("venue_id")
                prob = m.get("probability")
                if venue_id and prob is not None:
                    venue_probs[venue_id] = float(prob)

            # Check for arbitrage between known venues
            if "kalshi" in venue_probs and "polymarket" in venue_probs:
                kalshi_p = venue_probs["kalshi"]
                poly_p = venue_probs["polymarket"]

                spread = abs(kalshi_p - poly_p)

                if spread >= self.min_spread:
                    # Determine action
                    if kalshi_p < poly_p:
                        action = "buy_kalshi_sell_polymarket"
                        expected_profit = spread * 0.5  # Simplified: assume 50% of spread is profit
                    else:
                        action = "buy_polymarket_sell_kalshi"
                        expected_profit = spread * 0.5

                    arbitrage_opps.append(
                        ArbitrageOpportunity(
                            event_id=event_id,
                            venue_1="kalshi",
                            venue_2="polymarket",
                            prob_1=kalshi_p,
                            prob_2=poly_p,
                            spread=spread,
                            expected_profit=expected_profit,
                            action=action,
                        )
                    )

            # Check other venue pairs
            venue_list = list(venue_probs.keys())
            for i in range(len(venue_list)):
                for j in range(i + 1, len(venue_list)):
                    v1, v2 = venue_list[i], venue_list[j]
                    p1, p2 = venue_probs[v1], venue_probs[v2]
                    spread = abs(p1 - p2)

                    if spread >= self.min_spread:
                        action = f"buy_{v1}_sell_{v2}" if p1 < p2 else f"buy_{v2}_sell_{v1}"
                        arbitrage_opps.append(
                            ArbitrageOpportunity(
                                event_id=event_id,
                                venue_1=v1,
                                venue_2=v2,
                                prob_1=p1,
                                prob_2=p2,
                                spread=spread,
                                expected_profit=spread * 0.5,
                                action=action,
                            )
                        )

        return arbitrage_opps
