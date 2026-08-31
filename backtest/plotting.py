"""Equity curve plotting.

Kept deliberately simple: one function, saves a PNG. The one thing this
must get right (found as a real risk during Stage 4 review): a strategy
halted by the circuit breaker produces a flat line for the rest of the
period, which can look visually identical to "the strategy just stopped
making money" instead of "the strategy was deliberately shut off". Every
circuit-breaker trigger gets an explicit vertical marker so this is never
ambiguous on the chart itself.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(
    curves: dict[str, pd.Series],
    circuit_breaker_dates: dict[str, pd.Timestamp | None],
    title: str,
    output_path: Path,
) -> None:
    """Plot multiple equity curves on one chart, normalized to start at 1.0
    (so strategies with different initial capital, or none at all, are
    directly comparable), with a vertical dashed line + label at each
    strategy's circuit breaker trigger date, if any.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, curve in curves.items():
        normalized = curve / curve.iloc[0]
        line = ax.plot(normalized.index, normalized.to_numpy(), label=name, linewidth=1.5)[0]

        trigger_date = circuit_breaker_dates.get(name)
        if trigger_date is not None:
            ax.axvline(trigger_date, color=line.get_color(), linestyle="--", alpha=0.5)
            ax.annotate(
                f"{name}: circuit breaker",
                xy=(trigger_date, 1.0),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=7,
                color=line.get_color(),
                rotation=90,
                va="bottom",
            )

    ax.set_title(title)
    ax.set_ylabel("Capital normalizado (inicio = 1.0)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
