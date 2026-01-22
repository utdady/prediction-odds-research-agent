"""Generate professional PDF tear sheets."""
from __future__ import annotations

from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_tear_sheet(
    backtest_run_id: str,
    output_path: str | None = None,
    metrics: dict | None = None,
    trades: list | None = None,
) -> str:
    """
    Create professional backtest report PDF.
    
    Args:
        backtest_run_id: ID of the backtest run
        output_path: Output file path (if None, uses artifacts/)
        metrics: Dictionary of backtest metrics
        trades: List of trade dictionaries
    
    Returns:
        Path to generated PDF
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for tear sheet generation")

    if output_path is None:
        Path("artifacts").mkdir(exist_ok=True)
        output_path = f"artifacts/tearsheet_{backtest_run_id}.pdf"

    pdf = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Page 1: Summary
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(1 * inch, height - 1 * inch, "Backtest Tear Sheet")

    pdf.setFont("Helvetica", 12)
    y = height - 1.5 * inch

    pdf.drawString(1 * inch, y, f"Run ID: {backtest_run_id}")
    y -= 0.25 * inch

    if metrics:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(1 * inch, y, "Performance Metrics")
        y -= 0.3 * inch
        pdf.setFont("Helvetica", 10)

        metric_labels = {
            "sharpe": "Sharpe Ratio",
            "max_drawdown": "Max Drawdown",
            "cagr": "CAGR",
            "total_return": "Total Return",
            "win_rate": "Win Rate",
        }

        for key, label in metric_labels.items():
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    if "drawdown" in key or "return" in key or "rate" in key:
                        value_str = f"{value:.2%}"
                    else:
                        value_str = f"{value:.2f}"
                else:
                    value_str = str(value)

                pdf.drawString(1.2 * inch, y, f"{label}: {value_str}")
                y -= 0.2 * inch

    # Trade Summary
    if trades:
        y -= 0.2 * inch
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(1 * inch, y, f"Trade Summary ({len(trades)} trades)")
        y -= 0.3 * inch
        pdf.setFont("Helvetica", 10)

        if len(trades) > 0:
            winning_trades = [t for t in trades if t.get("pnl_pct", 0) > 0]
            losing_trades = [t for t in trades if t.get("pnl_pct", 0) <= 0]

            pdf.drawString(1.2 * inch, y, f"Winning Trades: {len(winning_trades)}")
            y -= 0.2 * inch
            pdf.drawString(1.2 * inch, y, f"Losing Trades: {len(losing_trades)}")
            y -= 0.2 * inch

            if winning_trades:
                avg_win = sum(t.get("pnl_pct", 0) for t in winning_trades) / len(
                    winning_trades
                )
                pdf.drawString(1.2 * inch, y, f"Avg Win: {avg_win:.2%}")
                y -= 0.2 * inch

            if losing_trades:
                avg_loss = sum(t.get("pnl_pct", 0) for t in losing_trades) / len(
                    losing_trades
                )
                pdf.drawString(1.2 * inch, y, f"Avg Loss: {avg_loss:.2%}")

    pdf.save()

    return output_path

