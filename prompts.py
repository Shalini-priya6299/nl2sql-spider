"""Prompt templates. Each template has a version string that is saved in the
experiment config, so results can always be traced back to the exact prompt.

Part 2 (few-shot) and Part 3 (self-correction) should add their templates here
and keep ZERO_SHOT unchanged so E1 stays reproducible.
"""

from data_utils import get_schema_prompt

SYSTEM_PROMPT = (
    "You are an expert in SQLite. Given a database schema and a question, "
    "write one SQLite query that answers the question. "
    "Return only the SQL query, with no explanation."
)

ZERO_SHOT_VERSION = "zeroshot-v1"


def zero_shot_messages(example: dict, n_rows: int = 3) -> list[dict]:
    """Chat messages for E1: schema (CREATE TABLE + sample rows) + question."""
    schema = get_schema_prompt(example["db_id"], n_rows)
    user = (
        f"{schema}\n\n"
        "-- Using valid SQLite, answer the following question for the tables provided above.\n"
        f"-- Question: {example['question']}\n"
        "-- SQL:"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]
