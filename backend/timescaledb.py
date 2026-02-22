"""TimescaleDB connection and schema initialisation.

The measurements hypertable stores every MDF channel as a time-series of rows:
    (time, file_name, channel, value, unit)

Call ensure_schema() once at application startup to create the extension,
table, and indexes if they don't already exist.
"""

import os

import psycopg2
from psycopg2.extensions import connection


_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

CREATE TABLE IF NOT EXISTS measurements (
    time        TIMESTAMPTZ     NOT NULL,
    file_name   TEXT            NOT NULL,
    channel     TEXT            NOT NULL,
    value       DOUBLE PRECISION,
    unit        TEXT            NOT NULL DEFAULT ''
);

SELECT create_hypertable('measurements', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_measurements_channel
    ON measurements (channel, time DESC);

CREATE INDEX IF NOT EXISTS idx_measurements_file
    ON measurements (file_name, time DESC);
"""


def get_connection() -> connection:
    """Return a new psycopg2 connection using the TIMESCALEDB_URL env var."""
    return psycopg2.connect(os.environ["TIMESCALEDB_URL"])


def ensure_schema() -> None:
    """Create the measurements hypertable and indexes if they don't exist.

    Safe to call multiple times — every statement is idempotent.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
