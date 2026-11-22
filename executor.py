"""executor.py: safe SQL execution for the self-correction loop (Part 3).

Builds on the shared helpers in data_utils.py / evaluate.py, so Part 3 opens the
Spider databases and compares results exactly like every other part.

    - read-only SQLite connection (the model can never change a database)
    - query timeout (the query is interrupted after ``timeout_s`` seconds)
    - SQLite error capture (the message is fed back to the model)
    - execution-time measurement (reported as part of the extra cost)
    - result comparison helper (same rules as the official evaluation)
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from data_utils import connect_readonly
from evaluate import exec_match

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_ROWS = 10_000


@dataclass
class ExecResult:
    rows: list | None       # None when the query failed
    error: str | None       # SQLite message, "timeout", or a short reason
    seconds: float          # wall-clock time spent executing
    truncated: bool = False  # True if more than max_rows rows were produced

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def empty(self) -> bool:
        """The query ran without error but returned no rows."""
        return self.error is None and not self.rows


def run_sql(
    db_file: str | Path,
    sql: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> ExecResult:
    """Run one SELECT query read-only. Never raises for a bad query: the error is returned.

    Why it returns an ExecResult instead of raising: the self-correction loop must keep
    going after a bad query, and it needs the error text to give back to the model.
    """
    start = time.perf_counter()
    text = (sql or "").strip()

    # Why: a model sometimes returns nothing (e.g. it only wrote prose). SQLite would give
    # a confusing message, so we give the model a clear one.
    if not text:
        return ExecResult(None, "empty query", 0.0)

    # Why: the SQL comes from an LLM, so we never trust it. The read-only connection below
    # already blocks writes; this check also rejects DROP/DELETE/etc. before touching the
    # database. lstrip("(") because "(SELECT ...) UNION (SELECT ...)" is valid Spider SQL.
    if not text.lstrip("( \t\n").lower().startswith(("select", "with")):
        return ExecResult(None, "only SELECT queries are allowed", 0.0)

    # Why connect_readonly (shared helper): every part opens Spider databases the same way,
    # read-only and tolerant of non-UTF-8 text. A missing database becomes an error result
    # instead of crashing a multi-hour run.
    try:
        conn = connect_readonly(db_file)
    except FileNotFoundError as exc:
        return ExecResult(None, str(exc), time.perf_counter() - start)

    # Why a progress handler: sqlite3 has no query timeout. SQLite calls this function every
    # 10,000 virtual-machine steps; returning non-zero aborts the query. Without it, one
    # accidental cross join on a big table could hang the whole experiment.
    deadline = start + timeout_s
    conn.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
    try:
        # Why fetchmany(max_rows + 1): a runaway query must not fill memory. Fetching one row
        # more than the limit tells us whether the result was cut (truncated=True).
        rows = conn.execute(text).fetchmany(max_rows + 1)
        return ExecResult(rows[:max_rows], None, time.perf_counter() - start, len(rows) > max_rows)
    except sqlite3.OperationalError as exc:
        # SQLite reports an aborted query as "interrupted"; we call it "timeout" because that
        # is what the model (and the report) should see.
        message = "timeout" if str(exc) == "interrupted" else str(exc)
        return ExecResult(None, message, time.perf_counter() - start)
    except Exception as exc:  # e.g. several statements in one string, ProgrammingError
        return ExecResult(None, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
    finally:
        conn.close()  # always release the file, even after an error or timeout


def same_result(db_file: str | Path, pred_sql: str, gold_sql: str, timeout: float = 30.0) -> bool:
    """True if predicted and gold SQL give the same result (official comparison rules)."""
    return bool(exec_match(db_file, pred_sql, gold_sql, timeout)["correct"])
