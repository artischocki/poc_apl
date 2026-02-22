"""TimescaleDB tools for agent-driven sensor data storage and exploration.

Provides two tools:
- ingest_mdf_to_db: reads an MDF file and bulk-inserts all channels into TimescaleDB.
- query_sensor_data: executes a read-only SQL query against TimescaleDB.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import psycopg2.extras
from langchain.tools import tool

from timescaledb import get_connection

# Maximum rows returned to the agent per query to protect context window size
_MAX_ROWS = 500

# Rows per INSERT batch during ingestion
_BATCH_SIZE = 10_000


@tool
def ingest_mdf_to_db(file_path: str) -> str:
    """Ingest all channels from an MDF file into TimescaleDB.

    Each channel becomes a time-series of rows (time, file_name, channel, value, unit).
    The file name is used as the measurement identifier for later SQL queries.
    Re-ingesting the same file is safe — duplicate rows are skipped.

    Args:
        file_path: Absolute or relative path to the .mdf or .mf4 file.

    Returns:
        JSON string with a summary of channels ingested and total rows inserted.
    """
    try:
        from asammdf import MDF

        path = Path(file_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {file_path}"})

        file_name = path.name
        channels_ingested = []
        total_rows = 0

        conn = get_connection()
        try:
            with MDF(str(path)) as mdf:
                # start_time is a datetime; make it tz-aware (treat as UTC if naive)
                start_time: datetime = mdf.start_time
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)

                for channel_name in mdf.channels_db.keys():
                    try:
                        sig = mdf.get(channel_name)
                        if len(sig.samples) == 0:
                            continue

                        # Build (time, file_name, channel, value, unit) tuples
                        rows = []
                        for ts, raw_val in zip(sig.timestamps, sig.samples):
                            abs_time = start_time + timedelta(seconds=float(ts))
                            try:
                                val = float(raw_val)
                                val = val if np.isfinite(val) else None
                            except (TypeError, ValueError):
                                val = None
                            rows.append((abs_time, file_name, channel_name, val, sig.unit or ""))

                        # Bulk insert in batches
                        with conn.cursor() as cur:
                            for i in range(0, len(rows), _BATCH_SIZE):
                                psycopg2.extras.execute_values(
                                    cur,
                                    """
                                    INSERT INTO measurements (time, file_name, channel, value, unit)
                                    VALUES %s
                                    ON CONFLICT DO NOTHING
                                    """,
                                    rows[i : i + _BATCH_SIZE],
                                )
                        conn.commit()

                        channels_ingested.append(channel_name)
                        total_rows += len(rows)

                    except Exception as e:
                        channels_ingested.append(f"{channel_name} (error: {e})")

        finally:
            conn.close()

        return json.dumps(
            {
                "file": file_name,
                "channels_ingested": len(channels_ingested),
                "total_rows": total_rows,
                "channels": channels_ingested,
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def query_sensor_data(sql: str) -> str:
    """Execute a read-only SQL SELECT query against the TimescaleDB measurements database.

    The measurements table has columns:
        time (TIMESTAMPTZ), file_name (TEXT), channel (TEXT),
        value (DOUBLE PRECISION), unit (TEXT)

    TimescaleDB time-series functions (time_bucket, first, last, etc.) are available.
    Results are capped at 500 rows — use aggregation for large time ranges.

    Example queries:
        SELECT DISTINCT file_name FROM measurements;
        SELECT DISTINCT channel, unit FROM measurements WHERE file_name = 'run1.mf4';
        SELECT time_bucket('1 second', time) AS bucket, AVG(value)
            FROM measurements WHERE channel = 'Speed' GROUP BY bucket ORDER BY bucket;

    Args:
        sql: A SELECT SQL query string.

    Returns:
        JSON string with column names and result rows.
    """
    sql_stripped = sql.strip().upper()
    if not sql_stripped.startswith("SELECT") and not sql_stripped.startswith("WITH"):
        return json.dumps({"error": "Only SELECT queries are allowed."})

    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM ({sql}) _q LIMIT {_MAX_ROWS}")
                columns = [desc[0] for desc in cur.description]
                rows = [list(row) for row in cur.fetchall()]

            # Convert non-serialisable types (datetime, Decimal, …) to strings
            for row in rows:
                for i, cell in enumerate(row):
                    if not isinstance(cell, (int, float, str, bool, type(None))):
                        row[i] = str(cell)

            return json.dumps({"columns": columns, "rows": rows, "count": len(rows)}, indent=2)
        finally:
            conn.close()
    except Exception as e:
        return json.dumps({"error": str(e)})
