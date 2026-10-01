"""Self-contained HTML dashboard for paper trading: one file, viewable by
opening it directly in any browser (no server, no Python needed to view
it) -- built specifically because checking progress so far has meant
manually copy-pasting CSVs into a chat for review, which does not scale
past a few weeks of accumulated data.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paper_trading.report import TickerSummary


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _equity_chart_base64(ticker: str, log: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    normalized = log["equity"] / log["equity"].iloc[0]
    ax.plot(log.index, normalized.to_numpy(), linewidth=1.5, color="#2563eb")

    breaker_rows = log[log["circuit_breaker_triggered"]]
    if len(breaker_rows) > 0:
        first_trigger_date = breaker_rows["circuit_breaker_date"].dropna().iloc[0]
        ax.axvline(first_trigger_date, color="#dc2626", linestyle="--", alpha=0.6)

    ax.set_title(f"{ticker} -- capital normalizado (inicio = 1.0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _status_badge(summary: TickerSummary) -> str:
    if summary.stale:
        return '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;">DADO PARADO</span>'
    if summary.circuit_breaker_triggered:
        return '<span style="background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:4px;">CIRCUIT BREAKER</span>'
    return '<span style="background:#dcfce7;color:#166534;padding:2px 8px;border-radius:4px;">OK</span>'


def generate_dashboard(
    summaries: list[TickerSummary],
    logs: dict[str, pd.DataFrame],
    consistency_results: dict,
    output_path: Path,
    generated_at: pd.Timestamp,
) -> None:
    rows_html = []
    for summary in summaries:
        badge = _status_badge(summary)
        consistency = consistency_results.get(summary.ticker)
        if consistency is not None:
            consistency_html = (
                '<span style="color:#166534;">bate com o backtest</span>' if consistency.matches
                else f'<span style="color:#991b1b;">DIVERGE do backtest '
                     f'({consistency.max_relative_equity_diff:.4%})</span>'
            )
        else:
            consistency_html = '<span style="color:#6b7280;">nao verificado</span>'

        rows_html.append(f"""
        <tr>
          <td><strong>{summary.ticker}</strong></td>
          <td>{badge}</td>
          <td>{summary.days_logged}</td>
          <td>{summary.first_date.date()} a {summary.last_date.date()}</td>
          <td>{summary.total_return:+.2%}</td>
          <td>{summary.current_equity:,.2f}</td>
          <td>{summary.current_position:.4f}</td>
          <td>{consistency_html}</td>
        </tr>""")

    charts_html = []
    for ticker, log in logs.items():
        chart_b64 = _equity_chart_base64(ticker, log)
        charts_html.append(
            f'<div style="margin-bottom:24px;"><img src="data:image/png;base64,{chart_b64}" '
            f'style="max-width:100%;"/></div>'
        )

    html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Paper Trading -- Dashboard</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; color: #1f2937; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; }}
  h1 {{ font-size: 20px; }}
  .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>Paper Trading -- Dashboard (Etapa 6)</h1>
<p class="meta">Gerado em {generated_at.strftime('%Y-%m-%d %H:%M UTC')}</p>

<table>
  <tr><th>Ativo</th><th>Status</th><th>Dias</th><th>Periodo</th><th>Retorno</th><th>Equity</th><th>Posicao atual</th><th>Consistencia c/ backtest</th></tr>
  {''.join(rows_html)}
</table>

{''.join(charts_html)}

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
