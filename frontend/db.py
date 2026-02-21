"""SQLite table initialisation for Chainlit's SQLAlchemyDataLayer.

Chainlit does not auto-create tables, so we do it here with a plain synchronous
sqlite3 connection before the async event loop starts.  Safe to call multiple
times — every statement uses IF NOT EXISTS.
"""

import os
import sqlite3


def _db_path_from_url(url: str) -> str:
    """Convert a SQLAlchemy URL like sqlite+aiosqlite:///./foo.db to a file path."""
    # Four slashes → absolute path; three slashes → relative
    if url.startswith("sqlite+aiosqlite:////"):
        return "/" + url.removeprefix("sqlite+aiosqlite:////")
    return url.removeprefix("sqlite+aiosqlite:///")


_CREATE_STATEMENTS = """
CREATE TABLE IF NOT EXISTS users (
    id           TEXT PRIMARY KEY,
    identifier   TEXT UNIQUE NOT NULL,
    createdAt    TEXT,
    metadata     TEXT,
    updatedAt    TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    id             TEXT PRIMARY KEY,
    userId         TEXT REFERENCES users(id),
    userIdentifier TEXT,
    name           TEXT,
    createdAt      TEXT,
    updatedAt      TEXT,
    deletedAt      TEXT,
    tags           TEXT,
    metadata       TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id            TEXT PRIMARY KEY,
    threadId      TEXT REFERENCES threads(id),
    parentId      TEXT,
    name          TEXT,
    type          TEXT,
    input         TEXT,
    output        TEXT,
    metadata      TEXT,
    generation    TEXT,
    createdAt     TEXT,
    start         TEXT,
    end           TEXT,
    streaming     INTEGER DEFAULT 0,
    waitForAnswer INTEGER DEFAULT 0,
    isError       INTEGER DEFAULT 0,
    defaultOpen   INTEGER DEFAULT 0,
    showInput     TEXT    DEFAULT '0',
    tags          TEXT,
    language      TEXT
);

CREATE TABLE IF NOT EXISTS feedbacks (
    id      TEXT PRIMARY KEY,
    forId   TEXT REFERENCES steps(id),
    value   INTEGER,
    comment TEXT
);

CREATE TABLE IF NOT EXISTS elements (
    id           TEXT PRIMARY KEY,
    threadId     TEXT REFERENCES threads(id),
    forId        TEXT,
    type         TEXT,
    name         TEXT,
    display      TEXT,
    url          TEXT,
    objectKey    TEXT,
    chainlitKey  TEXT,
    props        TEXT,
    mime         TEXT,
    language     TEXT,
    page         INTEGER,
    size         INTEGER,
    autoPlay     INTEGER,
    playerConfig TEXT
);
"""


_MIGRATIONS = [
    # Added in Chainlit ≥ latest: defaultOpen column on steps
    "ALTER TABLE steps ADD COLUMN defaultOpen INTEGER DEFAULT 0",
]


def ensure_tables(db_url: str) -> None:
    """Create all Chainlit tables and apply additive migrations if needed."""
    path = _db_path_from_url(db_url)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_CREATE_STATEMENTS)
        for sql in _MIGRATIONS:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.commit()
    finally:
        conn.close()
