import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import config


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_used TEXT,
                search_queries_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS factors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                category TEXT,
                description TEXT,
                sentiment TEXT,
                magnitude INTEGER,
                recency_days REAL,
                raw_score REAL,
                affected_tickers TEXT,
                evidence TEXT,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ticker TEXT,
                current_score REAL,
                forward_score REAL,
                composite_score REAL,
                signal TEXT,
                confidence REAL,
                top_factors TEXT,
                price_14d_change REAL,
                FOREIGN KEY(run_id) REFERENCES runs(run_id)
            );
        """)


def save_run(run_id: str, timestamp: str, model_used: str, search_count: int):
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?)",
            (run_id, timestamp, model_used, search_count),
        )


def save_factors(run_id: str, factors: list[dict]):
    with _conn() as conn:
        conn.executemany(
            """INSERT INTO factors
               (run_id,category,description,sentiment,magnitude,recency_days,raw_score,affected_tickers,evidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (
                    run_id,
                    f.get("category"),
                    f.get("description"),
                    f.get("sentiment"),
                    f.get("magnitude"),
                    f.get("recency_days"),
                    f.get("raw_score"),
                    json.dumps(f.get("affected_tickers", [])),
                    json.dumps(f.get("evidence", [])),
                )
                for f in factors
            ],
        )


def save_recommendations(run_id: str, recs: list[dict]):
    with _conn() as conn:
        conn.executemany(
            """INSERT INTO recommendations
               (run_id,ticker,current_score,forward_score,composite_score,signal,confidence,top_factors,price_14d_change)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (
                    run_id,
                    r["ticker"],
                    r.get("current_score"),
                    r.get("forward_score"),
                    r.get("composite_score"),
                    r.get("signal"),
                    r.get("confidence"),
                    json.dumps(r.get("top_factors", [])),
                    r.get("price_14d_change"),
                )
                for r in recs
            ],
        )


def get_recent_factors(n_runs: int = 3) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT f.* FROM factors f
               JOIN runs r ON f.run_id = r.run_id
               WHERE r.run_id IN (
                   SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT ?
               )
               ORDER BY r.timestamp DESC""",
            (n_runs,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["affected_tickers"] = json.loads(d.get("affected_tickers") or "[]")
        d["evidence"] = json.loads(d.get("evidence") or "[]")
        result.append(d)
    return result


def get_recent_recommendations(n_runs: int = 10) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            """SELECT r.timestamp, rec.*
               FROM recommendations rec
               JOIN runs r ON rec.run_id = r.run_id
               WHERE r.run_id IN (
                   SELECT run_id FROM runs ORDER BY timestamp DESC LIMIT ?
               )
               ORDER BY r.timestamp DESC""",
            (n_runs,),
        ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["top_factors"] = json.loads(d.get("top_factors") or "[]")
        result.append(d)
    return result


def purge_old_data():
    cutoff = (datetime.utcnow() - timedelta(days=config.HISTORY_RETENTION_DAYS)).isoformat()
    with _conn() as conn:
        old_runs = conn.execute(
            "SELECT run_id FROM runs WHERE timestamp < ?", (cutoff,)
        ).fetchall()
        for row in old_runs:
            rid = row["run_id"]
            conn.execute("DELETE FROM factors WHERE run_id=?", (rid,))
            conn.execute("DELETE FROM recommendations WHERE run_id=?", (rid,))
            conn.execute("DELETE FROM runs WHERE run_id=?", (rid,))
