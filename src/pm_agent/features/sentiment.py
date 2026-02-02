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
            # Try multiple news sources for better coverage
            feed_urls = [
                f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
            ]
            
            recent_articles = []
            import feedparser
            from datetime import datetime, timedelta
            
            cutoff = datetime.now() - timedelta(hours=lookback_hours)
            
            for feed_url in feed_urls:
                try:
                    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                        response = await client.get(
                            feed_url, 
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                        )
                        response.raise_for_status()
                    
                    feed = feedparser.parse(response.text)
                    
                    # Extract articles - be more lenient with date parsing
                    for entry in feed.entries[:30]:  # Check up to 30 articles
                        try:
                            title = entry.get('title', '')
                            if not title:
                                continue
                                
                            # Try to parse date, but don't require it
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                try:
                                    pub_date = datetime(*entry.published_parsed[:6])
                                    if pub_date >= cutoff:
                                        recent_articles.append(title)
                                except (ValueError, TypeError):
                                    # If date parsing fails, include anyway (might be recent)
                                    recent_articles.append(title)
                            else:
                                # No date available, include it (assume recent)
                                recent_articles.append(title)
                        except Exception as e:
                            log.debug("article_parse_error", error=str(e))
                            continue
                    
                    # If we got articles, break
                    if recent_articles:
                        break
                except Exception as e:
                    log.warning("news_feed_failed", url=feed_url, error=str(e))
                    continue

            if not recent_articles:
                log.warning("no_recent_articles", ticker=ticker, lookback_hours=lookback_hours, feed_entries=len(feed.entries) if 'feed' in locals() else 0)
                # Return 0.0 if no articles found
                return 0.0
            
            log.info("articles_found", ticker=ticker, n_articles=len(recent_articles))

            log.info("articles_found", count=len(recent_articles), ticker=ticker)

            # Analyze sentiment
            sentiments = []
            for article_title in recent_articles[:20]:  # Limit to 20 articles
                if self.sentiment_model:
                    try:
                        result = self.sentiment_model(article_title)[0]
                        # FinBERT outputs: positive, negative, neutral
                        score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                        sentiment_score = score_map.get(result["label"], 0.0) * result["score"]
                        sentiments.append(sentiment_score)
                    except Exception as e:
                        log.warning("sentiment_analysis_failed", error=str(e))
                        # Fall back to keyword-based
                        title_lower = article_title.lower()
                        positive_words = ["up", "gain", "rise", "surge", "rally", "beat", "strong", "profit", "growth", "bullish"]
                        negative_words = ["down", "fall", "drop", "crash", "miss", "weak", "loss", "decline", "bearish", "worry"]
                        
                        pos_count = sum(1 for w in positive_words if w in title_lower)
                        neg_count = sum(1 for w in negative_words if w in title_lower)
                        
                        if pos_count > neg_count:
                            sentiments.append(0.6)
                        elif neg_count > pos_count:
                            sentiments.append(-0.6)
                        else:
                            sentiments.append(0.0)
                else:
                    # Enhanced keyword-based sentiment (fallback)
                    title_lower = article_title.lower()
                    positive_words = ["up", "gain", "rise", "surge", "rally", "beat", "strong", "profit", "growth", "bullish", "soar", "jump", "climb", "advance", "boost"]
                    negative_words = ["down", "fall", "drop", "crash", "miss", "weak", "loss", "decline", "bearish", "worry", "plunge", "tumble", "slump", "dip", "concern"]

                    pos_count = sum(1 for w in positive_words if w in title_lower)
                    neg_count = sum(1 for w in negative_words if w in title_lower)

                    # Weighted scoring based on word count - more sensitive
                    if pos_count > neg_count:
                        # More positive words = higher score
                        # Start at 0.4 instead of 0.3 for better sensitivity
                        score = min(0.9, 0.4 + (pos_count - neg_count) * 0.15)
                        sentiments.append(score)
                        log.debug("positive_sentiment", title=article_title[:50], score=score, pos=pos_count, neg=neg_count)
                    elif neg_count > pos_count:
                        # More negative words = lower score
                        # Start at -0.4 instead of -0.3 for better sensitivity
                        score = max(-0.9, -0.4 - (neg_count - pos_count) * 0.15)
                        sentiments.append(score)
                        log.debug("negative_sentiment", title=article_title[:50], score=score, pos=pos_count, neg=neg_count)
                    else:
                        sentiments.append(0.0)
                        log.debug("neutral_sentiment", title=article_title[:50], pos=pos_count, neg=neg_count)

            # Aggregate
            if not sentiments:
                log.warning("no_sentiments_calculated", ticker=ticker, n_articles=len(recent_articles))
                return 0.0

            avg_sentiment = float(np.mean(sentiments))
            log.info("sentiment_calculated", ticker=ticker, score=avg_sentiment, n_sentiments=len(sentiments), n_articles=len(recent_articles))
            return avg_sentiment

        except Exception as e:
            log.warning("news_sentiment_fetch_failed", ticker=ticker, error=str(e))
            return 0.0
