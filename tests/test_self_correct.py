"""Tests for executor.py and self_correct.py. No GPU or model needed: the LLM is replaced by a
fake that returns scripted SQL, so we can check the retry logic exactly.

    python -m pytest tests/test_self_correct.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils import db_path  # noqa: E402
from executor import run_sql, same_result  # noqa: E402
from self_correct import build_correction_messages, self_correct  # noqa: E402

DB = db_path("concert_singer")
pytestmark = pytest.mark.skipif(not DB.exists(), reason="Spider databases not downloaded")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "\n".join(m["content"] for m in messages)


def run_loop(scripted, **kwargs):
    """Run self_correct with a fake LLM that returns ``scripted`` SQL strings in order."""
    queue = list(scripted)
    prompts = []

    def fake_generate(batch, model, tokenizer, **_):
        prompts.append(batch[0])
        return [queue.pop(0)]

    messages = [{"role": "user", "content": "How many singers are there?"}]
    with patch("self_correct.generate_sql", side_effect=fake_generate):
        result = self_correct(messages, DB, None, FakeTokenizer(), **kwargs)
    return result, prompts


# ---- executor ---------------------------------------------------------------
def test_run_sql_ok():
    r = run_sql(DB, "SELECT count(*) FROM singer")
    assert r.ok and r.rows[0][0] > 0 and r.seconds >= 0


def test_run_sql_captures_error():
    r = run_sql(DB, "SELECT nme FROM singer")
    assert r.rows is None and "no such column" in r.error


def test_run_sql_blocks_writes():
    r = run_sql(DB, "DROP TABLE singer")
    assert r.error == "only SELECT queries are allowed"
    assert run_sql(DB, "SELECT count(*) FROM singer").ok  # table still there


def test_run_sql_timeout():
    # a huge cross join never finishes within 0.2 s
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    r = run_sql(DB, sql, timeout_s=0.2)
    assert r.error == "timeout" and r.seconds < 5


def test_run_sql_missing_database():
    assert "not found" in run_sql("no/such.sqlite", "SELECT 1").error


def test_run_sql_empty_flag():
    assert run_sql(DB, "SELECT * FROM singer WHERE 1 = 0").empty


def test_same_result():
    assert same_result(DB, "SELECT count(*) FROM singer", "SELECT count(*) FROM singer")
    # 0 rows counted vs all rows counted: different results
    assert not same_result(DB, "SELECT count(*) FROM singer WHERE 1 = 0", "SELECT count(*) FROM singer")


# ---- self-correction loop ---------------------------------------------------
def test_no_retry_when_first_sql_works():
    result, prompts = run_loop(["SELECT count(*) FROM singer"])
    assert result["success"] and result["llm_calls"] == 1 and result["extra_llm_calls"] == 0
    assert len(prompts) == 1


def test_error_is_fixed_on_retry_and_fed_back():
    result, prompts = run_loop(["SELECT nme FROM singer", "SELECT count(*) FROM singer"])
    assert result["success"] and result["sql"] == "SELECT count(*) FROM singer"
    assert result["extra_llm_calls"] == 1
    assert "no such column" in prompts[1]  # the SQLite error reached the model
    assert "SELECT nme FROM singer" in prompts[1]  # and so did its previous SQL


def test_retries_are_limited():
    bad = ["SELECT a FROM singer"] * 5
    result, prompts = run_loop(bad, max_retries=2)
    assert not result["success"] and result["total_attempts"] == 3 and len(prompts) == 3
    result, _ = run_loop(bad, max_retries=0)
    assert result["total_attempts"] == 1


def test_empty_result_triggers_retry():
    result, prompts = run_loop(["SELECT * FROM singer WHERE 1 = 0", "SELECT count(*) FROM singer"])
    assert result["extra_llm_calls"] == 1 and "no rows" in prompts[1]


def test_empty_retry_can_be_disabled():
    result, _ = run_loop(["SELECT * FROM singer WHERE 1 = 0"], retry_on_empty=False)
    assert result["extra_llm_calls"] == 0


def test_best_attempt_is_kept_when_retry_crashes():
    # attempt 1 runs but is empty, attempts 2 and 3 crash: keep attempt 1, not a crashing query
    result, _ = run_loop(["SELECT * FROM singer WHERE 1 = 0", "SELECT zzz FROM singer", "SELECT yyy FROM singer"])
    assert result["success"] and result["sql"] == "SELECT * FROM singer WHERE 1 = 0"


def test_cost_is_recorded():
    result, _ = run_loop(["SELECT nme FROM singer", "SELECT count(*) FROM singer"])
    assert result["generation_time"] >= 0 and result["execution_time"] >= 0
    assert all({"attempt", "sql", "success", "error", "execution_time"} <= a.keys() for a in result["attempts"])


def test_correction_messages_do_not_change_original():
    original = [{"role": "user", "content": "q"}]
    out = build_correction_messages(original, "SELECT 1", "boom")
    assert len(original) == 1 and [m["role"] for m in out] == ["user", "assistant", "user"]
