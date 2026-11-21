"""Sanity tests for the evaluator. Run: python -m pytest tests/  (or python tests/test_evaluate.py)
Requires Spider dev data (python data_utils.py download --only-dev)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json  # noqa: E402

import pytest  # noqa: E402

from data_utils import _quote, db_path, load_split  # noqa: E402
from evaluate import SUBSET_FILE, clean_sql, exec_match, execute_sql, results_equal  # noqa: E402

DB = db_path("concert_singer")


def test_results_equal_rules():
    assert results_equal([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], order_matters=False)
    assert not results_equal([(1, "a"), (2, "b")], [(2, "b"), (1, "a")], order_matters=True)
    assert results_equal([(1, "a")], [("a", 1)], order_matters=False)  # column order ignored
    assert not results_equal([(1,), (1,)], [(1,)], order_matters=False)  # multiset, not set
    assert results_equal([], [], order_matters=False)


def test_exec_match():
    gold = "SELECT name, age FROM singer ORDER BY age DESC"
    assert exec_match(DB, "SELECT Name, Age FROM singer ORDER BY Age DESC", gold)["correct"]
    assert exec_match(DB, "select age, name from singer order by age desc", gold)["correct"]
    assert not exec_match(DB, "SELECT name, age FROM singer ORDER BY age ASC", gold)["correct"]
    assert not exec_match(DB, "SELECT name FROM singer", gold)["correct"]
    r = exec_match(DB, "SELECT nme FROM singer", gold)
    assert not r["correct"] and "no such column" in r["pred_error"]


def test_execute_sql_is_read_only():
    rows, err = execute_sql(DB, "DELETE FROM singer")
    assert rows is None and "readonly" in err.replace("-", "").replace(" ", "").lower()


def test_clean_sql():
    assert clean_sql("```sql\nSELECT *\nFROM t;\n```") == "SELECT * FROM t"
    assert clean_sql("Here is the query: SELECT a FROM t; -- done") == "SELECT a FROM t"
    assert clean_sql("```sql\nSELECT 1") == "SELECT 1"


def test_clean_sql_edge_cases():
    # a query that already starts with SELECT must not be cut at a later "with"
    assert clean_sql("SELECT name FROM t WHERE note LIKE 'x with y'") == \
        "SELECT name FROM t WHERE note LIKE 'x with y'"
    assert clean_sql("(SELECT 1) UNION (SELECT 2)") == "(SELECT 1) UNION (SELECT 2)"
    assert clean_sql("WITH a AS (SELECT 1) SELECT * FROM a;") == "WITH a AS (SELECT 1) SELECT * FROM a"
    assert clean_sql("SELECT a FROM t WHERE b = 'x;y'; SELECT 2") == "SELECT a FROM t WHERE b = 'x;y'"
    assert clean_sql("") == "" and clean_sql(None) == ""


def test_quote_identifiers():
    assert _quote("singer_id") == "singer_id"
    assert _quote("Song Name") == '"Song Name"'
    assert _quote("1st") == '"1st"'
    assert _quote("") == '""'  # must not raise IndexError
    assert _quote('a"b') == '"a""b"'


def test_missing_database_raises():
    with pytest.raises(FileNotFoundError):
        execute_sql(db_path("no_such_db"), "SELECT 1")


def test_empty_and_multi_statement_sql_are_errors():
    assert execute_sql(DB, "")[0] is None
    rows, err = execute_sql(DB, "SELECT 1; SELECT 2")
    assert rows is None and err


def test_balanced_subset_file():
    sub = json.loads(SUBSET_FILE.read_text())
    dev = load_split("dev")
    idxs = sub["dev_indices"]
    assert len(idxs) == len(set(idxs)) == 4 * sub["per_level"]
    from collections import Counter
    assert Counter(dev[i]["hardness"] for i in idxs) == {k: sub["per_level"] for k in sub["counts"]}


def test_dev_hardness_matches_official_counts():
    from collections import Counter
    c = Counter(e["hardness"] for e in load_split("dev"))
    assert c == {"easy": 248, "medium": 446, "hard": 174, "extra": 166}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
