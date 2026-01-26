# Search Agent Guide

## What is the Search Agent?

The **Search Agent** is an interactive conversational interface that lets you ask questions about companies, signals, backtests, markets, and features. Think of it as a "person search" for financial data - you can ask it anything and it will find the relevant information.

## How to Use

### Via Dashboard

1. Open the dashboard: http://localhost:8501
2. Go to the **"Agent"** tab
3. Type your question in the chat input
4. Get instant answers with data visualizations

### Example Queries

#### Company Information
- **"show AAPL"** - Get all information about Apple
- **"company information for MSFT"** - Get Microsoft details
- **"what companies do we have?"** - List all companies

#### Trading Signals
- **"signals for TSLA"** - Get trading signals for Tesla
- **"show signals"** - Get all recent signals
- **"buy signals"** - Get all buy recommendations

#### Backtest Performance
- **"backtest results"** - See strategy performance
- **"show backtest performance"** - Performance metrics
- **"what's our Sharpe ratio?"** - Get performance stats

#### Markets
- **"markets for AAPL"** - List markets for Apple
- **"show all markets"** - List all prediction markets
- **"what markets do we track?"** - Market overview

#### Features
- **"features for AAPL"** - Show feature data for Apple
- **"show feature data"** - Recent features
- **"delta_p_1d for MSFT"** - Specific feature for Microsoft

## Features

### 1. **Natural Language Queries**
Ask questions in plain English - no need to know SQL or specific commands.

### 2. **Smart Ticker Detection**
Automatically extracts ticker symbols (AAPL, MSFT, TSLA, etc.) from your queries.

### 3. **Comprehensive Company Views**
When you ask about a company, you get:
- Company information (ticker, name, sector)
- Related markets
- Trading signals
- Feature data

### 4. **Interactive Chat Interface**
- Chat history preserved during session
- Quick action buttons for common queries
- Clear chat option

### 5. **Data Visualizations**
Results are displayed as:
- Tables for structured data
- Summary text for quick answers
- Detailed views for company information

## Quick Actions

The dashboard includes quick action buttons:
- 📊 **Show All Companies** - List all tracked companies
- 📈 **Recent Signals** - Show latest trading signals
- 📉 **Backtest Results** - Show performance metrics
- ❓ **Help** - Show help message

## Architecture

```
User Query
    ↓
SearchAgent.search()
    ↓
Query Classification
    ├── Company Query → _handle_company_query()
    ├── Signal Query → _handle_signal_query()
    ├── Backtest Query → _handle_backtest_query()
    ├── Market Query → _handle_market_query()
    └── Feature Query → _handle_feature_query()
    ↓
Database Query
    ↓
Formatted Response (answer + data)
```

## Example Conversation

```
User: "show AAPL"
Agent: **AAPL** (Apple Inc.)
        Sector: Technology
        
        **Markets:** 2 found
        **Signals:** 0 found
        **Features:** 3 found
        
        [Shows tables with company info, markets, signals, features]

User: "signals for TSLA"
Agent: Signals for TSLA:
        [Shows table with all signals for Tesla]

User: "backtest performance"
Agent: **Backtest Performance Summary:**
        - Average Sharpe Ratio: 1.25
        - Best Sharpe Ratio: 1.50
        - Total Runs: 5
        [Shows backtest results table]
```

## Technical Details

- **Location**: `src/pm_agent/agent/search_agent.py`
- **Integration**: Dashboard tab in `src/app/dashboard/Home.py`
- **Database**: Uses SQLAlchemy engine for queries
- **Response Format**: Returns structured dict with answer, data, and type

## Future Enhancements

Potential improvements:
- Add visualization generation (charts, graphs)
- Support for more complex queries
- Natural language to SQL conversion
- Multi-turn conversations with context
- Export results to CSV/PDF

The Search Agent makes your data accessible through natural conversation! 🚀

