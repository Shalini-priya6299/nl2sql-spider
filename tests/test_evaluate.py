"""Sanity tests for the evaluator. Run: python -m pytest tests/  (or python tests/test_evaluate.py)
Requires Spider dev data (python data_utils.py download --only-dev)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_utils import db_path, load_split  # noqa: E402
from evaluate import clean_sql, exec_match, execute_sql, results_equal  # noqa: E402

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


def test_dev_hardness_matches_official_counts():
    from collections import Counter
    c = Counter(e["hardness"] for e in load_split("dev"))
    assert c == {"easy": 248, "medium": 446, "hard": 174, "extra": 166}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
