"""self_correct.py: execute generated SQL and retry with the error as feedback.

Used by Part 3:
    E3: few-shot prompt + self-correction (base model)
    E5: fine-tuned model (LoRA adapter) + self-correction

Loop (at most ``max_retries`` extra LLM calls per question; the issue asks for 2-3):
    1. the model writes SQL from the original prompt
    2. the SQL is run read-only on the database (executor.run_sql)
    3. if it failed (SQLite error / timeout) or returned no rows, the previous SQL and
       the error are appended to the conversation and the model is asked again
    4. stop when a query runs, or when the retries are used up

Everything the cost analysis needs is recorded: number of LLM calls, generation time
and execution time per attempt.
"""

from __future__ import annotations

import time
from pathlib import Path

from evaluate import clean_sql
from executor import DEFAULT_TIMEOUT_S, ExecResult, run_sql
from generate import generate_sql

# Why 2 retries: the issue asks for 2-3. Each retry is one more LLM call, so cost grows
# linearly; small models rarely fix a query on a third try that they missed on the second.
MAX_RETRIES = 2
# Why a version string: if the feedback wording changes, results are no longer comparable.
SELF_CORRECT_VERSION = "self-correct-v1"
EMPTY_RESULT_MESSAGE = "The query ran without error but returned no rows."
LOGGED_ROWS = 5  # rows kept per attempt in execution_logs.jsonl (full results can be huge)


def build_correction_messages(
    original_messages: list[dict],
    previous_sql: str,
    error_message: str,
) -> list[dict]:
    """Original prompt + the model's previous SQL + execution feedback."""
    # Why list(...): copy, so the caller's original prompt is never changed (it is reused for
    # every retry).
    messages = list(original_messages)
    # Why the previous SQL as an assistant turn: the model sees its own answer and the
    # feedback as a normal conversation, which is the format chat models are trained on.
    messages.append({"role": "assistant", "content": previous_sql})
    messages.append({
        "role": "user",
        "content": (
            "The SQL query above did not work.\n\n"
            f"Feedback from running it on the database: {error_message}\n\n"
            "Review the original question and database schema. "
            "Correct the SQL query and return only the corrected SQLite query. "
            "Do not repeat the previous mistake."
        ),
    })
    return messages


def _feedback(result: ExecResult) -> str:
    # Why the raw SQLite message: "no such column: singer.nme" tells the model exactly which
    # name is wrong, which is much more useful than a generic "try again".
    return f"SQLite error: {result.error}" if result.error else EMPTY_RESULT_MESSAGE


def self_correct(
    messages: list[dict],
    db_file: str | Path,
    model,
    tokenizer,
    max_retries: int = MAX_RETRIES,
    max_new_tokens: int = 256,
    timeout: float = DEFAULT_TIMEOUT_S,
    retry_on_empty: bool = True,
) -> dict:
    """Generate SQL for ``messages`` and correct it up to ``max_retries`` times.

    Returns a dict with
        sql                final SQL (see "best attempt" below)
        success            True if that SQL ran without error
        error              its error message, or None
        attempts           one dict per LLM call: attempt, sql, success, empty, error,
                           result (first rows), execution_time, generation_time
        total_attempts     len(attempts)
        llm_calls          number of LLM calls (== total_attempts)
        extra_llm_calls    llm_calls - 1, the extra cost of self-correction
        generation_time    seconds spent in the LLM
        execution_time     seconds spent running SQL
    Best attempt: the last query that ran without error; if none did, the last query.
    """
    current = list(messages)
    attempts: list[dict] = []
    generation_time = 0.0

    # Why max_retries + 1 attempts: attempt 1 is the normal first answer (the same SQL E2 or
    # E4 would give); every attempt after it is a retry. So E3/E5 can only differ from E2/E4
    # on questions that the first answer got wrong, and the extra cost is easy to count.
    for attempt in range(1, max_retries + 2):
        # Why the chat template here: the prompt must look exactly like it did in E1/E2/E4
        # (same format the model was trained/evaluated with), plus the feedback turns.
        prompt = tokenizer.apply_chat_template(current, tokenize=False, add_generation_prompt=True)

        # Why we time the LLM call separately: the report must show the extra cost of
        # self-correction, and generation is by far the expensive part.
        t0 = time.perf_counter()
        raw = generate_sql([prompt], model, tokenizer, max_new_tokens=max_new_tokens)[0]
        gen_seconds = time.perf_counter() - t0
        generation_time += gen_seconds

        # Why clean_sql: same post-processing as every other experiment (strips ``` fences,
        # prose and trailing text), so scoring is comparable.
        sql = clean_sql(raw)
        result = run_sql(db_file, sql, timeout_s=timeout)
        attempts.append({
            "attempt": attempt,
            "sql": sql,
            "success": result.ok,
            "empty": result.empty,
            "error": result.error,
            "result": [list(r) for r in (result.rows or [])[:LOGGED_ROWS]],
            "execution_time": round(result.seconds, 6),
            "generation_time": round(gen_seconds, 3),
        })

        # Why retry on an empty result too: the execution error is the clearest signal, but a
        # query that returns no rows is very often a wrong value or wrong condition
        # (e.g. 'france' vs 'France'). It is a weaker signal (an empty answer can be correct),
        # so it can be switched off with retry_on_empty=False.
        needs_retry = (not result.ok) or (retry_on_empty and result.empty)
        if not needs_retry or attempt == max_retries + 1:
            break  # query worked, or no retries left: stop spending LLM calls

        # Why we rebuild from the ORIGINAL messages (not from the previous correction):
        # the prompt grows by exactly one SQL + one feedback message, never by a chain of
        # old failures. That keeps the context short for a 1.5B model and avoids it copying
        # its own earlier mistakes.
        current = build_correction_messages(messages, sql, _feedback(result))

    # Why "best attempt": a retry made only because of an empty result may end in a SQL error.
    # An earlier query that at least ran is a better final answer than one that crashes, so
    # we return the last attempt that executed, and only fall back to the last attempt when
    # none ran. This way a retry can never make the execution-error rate worse.
    best = next((a for a in reversed(attempts) if a["success"]), attempts[-1])
    return {
        "sql": best["sql"],
        "success": best["success"],
        "error": best["error"],
        "attempts": attempts,
        "total_attempts": len(attempts),
        "llm_calls": len(attempts),
        "extra_llm_calls": len(attempts) - 1,
        "generation_time": round(generation_time, 3),
        "execution_time": round(sum(a["execution_time"] for a in attempts), 6),
    }
