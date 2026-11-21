"""Prompt templates. Each template has a version string that is saved in the
experiment config, so results can always be traced back to the exact prompt.

Part 2 (few-shot) and Part 3 (self-correction) should add their templates here
and keep zero_shot_messages() and ZERO_SHOT_VERSION unchanged so E1 stays reproducible.
"""

from data_utils import get_schema_prompt

SYSTEM_PROMPT = (
    "You are an expert in SQLite. Given a database schema and a question, "
    "write one SQLite query that answers the question. "
    "Return only the SQL query, with no explanation."
)

ZERO_SHOT_VERSION = "zeroshot-v1"
FEW_SHOT_VERSION = "fewshot-v1"


def few_shot_messages(
    example: dict,
    retrieved_examples: list,
    n_rows: int = 3,
) -> list[dict]:
    """Chat messages for E2: schema + few-shot context + question.

    Args:
        example:            Dev/test example dict with 'db_id' and 'question'.
        retrieved_examples: List of retrieval.Example objects from RandomRetriever
                            or SimilarRetriever (may be empty → zero-shot fallback).
        n_rows:             Sample rows per table to include in the schema block.

    Returns:
        List of chat message dicts ready for apply_chat_template.
    """
    schema = get_schema_prompt(example["db_id"], n_rows)

    # Build the few-shot block from retrieved examples.
    few_shot_lines = []
    for i, ex in enumerate(retrieved_examples, 1):
        few_shot_lines.append(
            f"-- Example {i}\n"
            f"-- Question: {ex.question}\n"
            f"{ex.query.strip()}"
        )
    few_shot_block = "\n\n".join(few_shot_lines)

    if few_shot_block:
        user = (
            f"{schema}\n\n"
            "-- Here are some example question/SQL pairs to guide your style:\n\n"
            f"{few_shot_block}\n\n"
            "-- Using valid SQLite, answer the following question for the tables provided above.\n"
            f"-- Question: {example['question']}\n"
            "-- SQL:"
        )
    else:
        # Graceful zero-shot fallback when k=0 or retrieval pool is empty.
        user = (
            f"{schema}\n\n"
            "-- Using valid SQLite, answer the following question for the tables provided above.\n"
            f"-- Question: {example['question']}\n"
            "-- SQL:"
        )

    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


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
