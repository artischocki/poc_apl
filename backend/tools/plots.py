"""Plotting tools for agent-driven sensor data visualisation.

Generates interactive Plotly figures from SQL query results, saves them as
JSON to the backend's /plots directory, and returns a PLOT:{uuid} token.
The streaming layer in main.py intercepts this token, reads the JSON, and
emits a typed SSE event so the frontend can render a cl.Plotly element.

Two tools:
- plot_timeseries: time on x-axis, one or more numeric series as lines.
- plot_barchart:   categorical x-axis, numeric y-axis (aggregates/comparisons).
"""

import time
import uuid
from pathlib import Path

import plotly.graph_objects as go
from langchain.tools import tool

from timescaledb import get_connection

_PLOTS_DIR = Path(__file__).parent.parent / "plots"
_PLOTS_DIR.mkdir(exist_ok=True)

_MAX_PLOT_AGE_S = 3600


def _purge_old_plots() -> None:
    """Remove plot JSON files older than _MAX_PLOT_AGE_S."""
    now = time.time()
    for f in _PLOTS_DIR.glob("*.json"):
        if now - f.stat().st_mtime > _MAX_PLOT_AGE_S:
            f.unlink(missing_ok=True)


def _save_fig(fig: go.Figure) -> str:
    """Persist figure as JSON and return its UUID."""
    _purge_old_plots()
    uid = uuid.uuid4().hex
    (_PLOTS_DIR / f"{uid}.json").write_text(fig.to_json())
    return uid


@tool
def plot_timeseries(sql: str, title: str, y_label: str = "value") -> str:
    """Execute a SQL query and render the result as an interactive time-series line chart.

    The query must return columns in one of these shapes:
      - (timestamp, value)              → single line
      - (timestamp, value, series_name) → one line per distinct series_name

    The timestamp column must be the first column.
    Use time_bucket() to downsample large ranges before plotting.

    Args:
        sql:     SELECT query returning (time, value) or (time, value, series).
        title:   Chart title shown above the plot.
        y_label: Label for the y-axis (include unit, e.g. "Speed (km/h)").

    Returns:
        A plot reference token (PLOT:uuid). The chart is rendered automatically in the UI.
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

        fig = go.Figure()

        if len(cols) >= 3:
            # Multi-series: group by the third column
            series: dict = {}
            for row in rows:
                key = str(row[2])
                series.setdefault(key, ([], []))
                series[key][0].append(row[0])
                series[key][1].append(row[1])
            for name, (xs, ys) in series.items():
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name))
        else:
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=y_label,
                line=dict(color="#2596be"),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=y_label,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=60, r=20, t=50, b=40),
        )

        return f"PLOT:{_save_fig(fig)}"

    except Exception as e:
        return f"Plot error: {e}"


@tool
def plot_barchart(sql: str, title: str, y_label: str = "value") -> str:
    """Execute a SQL query and render the result as an interactive bar chart.

    The query must return exactly two columns: (label, value).
    Useful for comparing aggregates across channels, runs, or time buckets.

    Args:
        sql:     SELECT query returning (label, value).
        title:   Chart title shown above the plot.
        y_label: Label for the y-axis (include unit).

    Returns:
        A plot reference token (PLOT:uuid). The chart is rendered automatically in the UI.
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

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker_color="#2596be",
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title=title,
            yaxis_title=y_label,
            template="plotly_dark",
            margin=dict(l=60, r=20, t=50, b=80),
        )

        return f"PLOT:{_save_fig(fig)}"

    except Exception as e:
        return f"Plot error: {e}"
