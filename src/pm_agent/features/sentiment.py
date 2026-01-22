"""Sentiment analysis from news headlines."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import numpy as np
import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment from news headlines."""

    def __init__(self, use_finbert: bool = False):
        self.use_finbert = use_finbert
        self.sentiment_model = None

        if use_finbert:
            try:
                from transformers import pipeline

                # Use FinBERT (BERT trained on financial news)
                self.sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                )
                log.info("finbert_loaded")
            except ImportError:
                log.warning("transformers_not_installed", suggestion="pip install transformers torch")
            except Exception as e:
                log.warning("finbert_load_failed", error=str(e))

    async def get_news_sentiment(
        self, ticker: str, lookback_hours: int = 24
    ) -> float:
        """
        Fetch recent news and compute sentiment score.
        
        Args:
            ticker: Stock ticker symbol
            lookback_hours: Hours to look back for news
        
        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        # Fetch news (using Google News RSS as example)
        try:
            feed_url = f"https://news.google.com/rss/search?q={ticker}+stock"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(feed_url)
                response.raise_for_status()

            # Parse RSS feed
            import feedparser

            feed = feedparser.parse(response.text)

            # Filter recent articles
            from datetime import datetime, timedelta

            cutoff = datetime.now() - timedelta(hours=lookback_hours)
            recent_articles = []
            for entry in feed.entries:
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                    if pub_date >= cutoff:
                        recent_articles.append(entry.title)
                except Exception:
                    pass

            if not recent_articles:
                return 0.0

            # Analyze sentiment
            sentiments = []
            for article_title in recent_articles[:10]:  # Limit to 10 articles
                if self.sentiment_model:
                    try:
                        result = self.sentiment_model(article_title)[0]
                        # FinBERT outputs: positive, negative, neutral
                        score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                        sentiment_score = score_map.get(result["label"], 0.0) * result["score"]
                        sentiments.append(sentiment_score)
                    except Exception as e:
                        log.warning("sentiment_analysis_failed", error=str(e))
                else:
                    # Simple keyword-based sentiment (fallback)
                    title_lower = article_title.lower()
                    positive_words = ["up", "gain", "rise", "surge", "rally", "beat", "strong"]
                    negative_words = ["down", "fall", "drop", "crash", "miss", "weak", "loss"]

                    pos_count = sum(1 for w in positive_words if w in title_lower)
                    neg_count = sum(1 for w in negative_words if w in title_lower)

                    if pos_count > neg_count:
                        sentiments.append(0.5)
                    elif neg_count > pos_count:
                        sentiments.append(-0.5)
                    else:
                        sentiments.append(0.0)

            # Aggregate
            if not sentiments:
                return 0.0

            return float(np.mean(sentiments))

        except Exception as e:
            log.warning("news_sentiment_fetch_failed", ticker=ticker, error=str(e))
            return 0.0
