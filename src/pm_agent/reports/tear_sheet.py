"""Generate professional PDF tear sheets for backtest results."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def generate_tear_sheet(
    run_id: str,
    output_path: str = "artifacts",
    metrics: dict | None = None,
    trades_df=None,
    equity_curve_df=None,
) -> str:
    """
    Create professional backtest report PDF.
    
    Args:
        run_id: Backtest run ID
        output_path: Directory to save PDF
        metrics: Dictionary with metrics (sharpe, max_drawdown, cagr, etc.)
        trades_df: DataFrame with trade details
        equity_curve_df: DataFrame with equity curve
    
    Returns:
        Path to generated PDF
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError:
        # Fallback to text-based report if reportlab not installed
        return _generate_text_tear_sheet(run_id, output_path, metrics, trades_df, equity_curve_df)

    output_file = Path(output_path) / f"tearsheet_{run_id}.pdf"
    doc = SimpleDocTemplate(str(output_file), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph(f"<b>Backtest Tear Sheet</b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))

    # Run Info
    story.append(Paragraph(f"<b>Run ID:</b> {run_id}", styles["Normal"]))
    story.append(Spacer(1, 0.1 * inch))

    # Metrics Table
    if metrics:
        metrics_data = [
            ["Metric", "Value"],
            ["Sharpe Ratio", f"{metrics.get('sharpe', 0.0):.2f}"],
            ["Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}"],
            ["CAGR", f"{metrics.get('cagr', 0.0):.2%}"],
            ["Total Return", f"{metrics.get('total_return', 0.0):.2%}"],
            ["Win Rate", f"{metrics.get('win_rate', 0.0):.2%}"],
        ]

        metrics_table = Table(metrics_data)
        metrics_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 14),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )
        story.append(metrics_table)
        story.append(Spacer(1, 0.3 * inch))

    # Trades Summary
    if trades_df is not None and len(trades_df) > 0:
        story.append(Paragraph("<b>Trade Summary</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))

        # Summary stats
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df.get("pnl_pct", 0) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        story.append(Paragraph(f"Total Trades: {total_trades}", styles["Normal"]))
        story.append(Paragraph(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    # Build PDF
    doc.build(story)

    return str(output_file)


def _generate_text_tear_sheet(
    run_id: str,
    output_path: str,
    metrics: dict | None,
    trades_df,
    equity_curve_df,
) -> str:
    """Fallback text-based tear sheet if reportlab not available."""
    output_file = Path(output_path) / f"tearsheet_{run_id}.txt"

    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BACKTEST TEAR SHEET\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Run ID: {run_id}\n\n")

        if metrics:
            f.write("METRICS\n")
            f.write("-" * 60 + "\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    if "drawdown" in key.lower() or "return" in key.lower() or "rate" in key.lower():
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")

        if trades_df is not None and len(trades_df) > 0:
            f.write("TRADE SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Trades: {len(trades_df)}\n")
            winning = len(trades_df[trades_df.get("pnl_pct", 0) > 0])
            f.write(f"Winning Trades: {winning} ({winning/len(trades_df)*100:.1f}%)\n")
            f.write("\n")

    return str(output_file)

