"""Plotting tools for agent-driven sensor data visualisation.

Generates Matplotlib figures from SQL query results, saves them as PNGs to the
backend's /plots static directory, and returns a markdown image reference that
Chainlit renders inline.

Two tools:
- plot_timeseries: time on x-axis, one or more numeric series as lines.
- plot_barchart:   categorical x-axis, numeric y-axis (aggregates, comparisons).
"""

import os
import time
import uuid
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from langchain.tools import tool

from timescaledb import get_connection

matplotlib.use("Agg")  # non-interactive backend — safe for server use

_PLOTS_DIR = Path(__file__).parent.parent / "plots"
_PLOTS_DIR.mkdir(exist_ok=True)

_BACKEND_PUBLIC_URL = os.getenv("BACKEND_PUBLIC_URL", "http://localhost:8000")

# Delete PNGs older than this many seconds to avoid unbounded disk growth
_MAX_PLOT_AGE_S = 3600


def _purge_old_plots() -> None:
    """Remove plot files older than _MAX_PLOT_AGE_S."""
    now = time.time()
    for f in _PLOTS_DIR.glob("*.png"):
        if now - f.stat().st_mtime > _MAX_PLOT_AGE_S:
            f.unlink(missing_ok=True)


def _save_fig(fig) -> str:
    """Save a matplotlib figure to the plots dir and return its public URL."""
    _purge_old_plots()
    filename = f"{uuid.uuid4().hex}.png"
    fig.savefig(_PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"{_BACKEND_PUBLIC_URL}/plots/{filename}"


@tool
def plot_timeseries(sql: str, title: str, y_label: str = "value") -> str:
    """Execute a SQL query and render the result as a time-series line chart.

    The query must return columns in one of these shapes:
      - (timestamp, value)                → single line
      - (timestamp, value, series_name)   → one line per distinct series_name

    The timestamp column must be the first column.
    Use time_bucket() or similar to downsample large ranges before plotting.

    Args:
        sql:     SELECT query returning (time, value) or (time, value, series).
        title:   Chart title shown above the plot.
        y_label: Label for the y-axis (include unit, e.g. "Speed (km/h)").

    Returns:
        Markdown image string that Chainlit renders inline.
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return "Query returned no data."

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        if len(cols) >= 3:
            # Multi-series: group by the third column
            from itertools import groupby
            series: dict = {}
            for row in rows:
                key = row[2]
                series.setdefault(key, ([], []))
                series[key][0].append(row[0])
                series[key][1].append(row[1])
            for name, (xs, ys) in series.items():
                ax.plot(xs, ys, label=str(name), linewidth=1.2)
            ax.legend(fontsize=8)
        else:
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            ax.plot(xs, ys, linewidth=1.2, color="#2596be")

        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.2)

        url = _save_fig(fig)
        return f"![{title}]({url})"

    except Exception as e:
        return f"Plot error: {e}"


@tool
def plot_barchart(sql: str, title: str, y_label: str = "value") -> str:
    """Execute a SQL query and render the result as a bar chart.

    The query must return exactly two columns: (label, value).
    Useful for comparing aggregates across channels, runs, or time buckets.

    Args:
        sql:     SELECT query returning (label, value).
        title:   Chart title shown above the plot.
        y_label: Label for the y-axis (include unit).

    Returns:
        Markdown image string that Chainlit renders inline.
    """
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return "Query returned no data."

        labels = [str(r[0]) for r in rows]
        values = [float(r[1]) if r[1] is not None else 0.0 for r in rows]

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4))
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        bars = ax.bar(labels, values, color="#2596be", width=0.6)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylabel(y_label, fontsize=10)
        plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.grid(True, axis="y", alpha=0.2)

        url = _save_fig(fig)
        return f"![{title}]({url})"

    except Exception as e:
        return f"Plot error: {e}"
