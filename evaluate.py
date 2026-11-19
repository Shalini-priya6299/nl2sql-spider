"""Execution Accuracy (EX) evaluation for Spider, with per-difficulty breakdown.

- Difficulty ("hardness") uses the official Spider rules (evaluation.py,
  taoyds/spider), implemented in ``hardness.py`` and re-exported here.
- EX runs predicted and gold SQL on the database and compares results with the
  same logic as the official test-suite evaluator (taoyds/test-suite-sql-eval,
  exec_eval.py): rows compared as a multiset (as a list if the gold query has
  ORDER BY), columns may be permuted.
- The fixed balanced dev subset (same N per difficulty) is created once with a
  fixed seed and saved to ``splits/dev_subset_balanced.json`` (committed to git)
  so every part of the project evaluates on the exact same questions.

Library use (other parts):
    from evaluate import execute_sql, exec_match, evaluate_predictions, load_dev_subset

Command line:
    python evaluate.py make-subset [--per-level 50 --seed 42]
    python evaluate.py score results/e1_zeroshot/predictions.jsonl
    python evaluate.py self-test         # gold-vs-gold sanity check, must be ~100%
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from data_utils import db_path, load_split
from hardness import HARDNESS_LEVELS, eval_hardness  # noqa: F401  (re-exported)

ROOT = Path(__file__).parent
SUBSET_FILE = ROOT / "splits" / "dev_subset_balanced.json"

# --------------------------------------------------------------------------- #
# SQL execution
# --------------------------------------------------------------------------- #
def execute_sql(db_file: str | Path, sql: str, timeout: float = 30.0):
    """Run ``sql`` read-only on ``db_file``.

    Returns (rows, None) on success or (None, error_message) on failure.
    The error message is what the self-correction loop (Part 3) feeds back.
    """
    if not sql or not sql.strip():
        return None, "empty query"
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        rows = conn.execute(sql).fetchall()
        return rows, None
    except sqlite3.OperationalError as e:
        msg = str(e)
        return None, ("timeout" if msg == "interrupted" else msg)
    except Exception as e:  # sqlite3.Warning (multiple statements), ProgrammingError, ...
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


# --- result comparison (test-suite-sql-eval exec_eval.py logic) --------------
def _unorder_row(row: tuple) -> tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def _rows_match_fast(r1, r2, order_matters: bool) -> bool:
    s1 = [_unorder_row(r) for r in r1]
    s2 = [_unorder_row(r) for r in r2]
    return s1 == s2 if order_matters else set(s1) == set(s2)


def _col_permutations(tab1_cols: list[set], result2: list[tuple], sample: int = 20):
    num_cols = len(result2[0])
    if num_cols <= 3:
        return itertools.permutations(range(num_cols))
    # restrict candidates: column j of result2 can map to column i of result1
    # only if its (sampled) values are all in column i
    rows = random.Random(0).sample(result2, min(sample, len(result2)))
    candidates = []
    for i in range(num_cols):
        candidates.append([j for j in range(num_cols) if all(r[j] in tab1_cols[i] for r in rows)])
    return itertools.product(*candidates)


def results_equal(r1: list[tuple], r2: list[tuple], order_matters: bool) -> bool:
    if len(r1) == 0 and len(r2) == 0:
        return True
    if len(r1) != len(r2):
        return False
    n = len(r1[0])
    if len(r2[0]) != n:
        return False
    if not _rows_match_fast(r1, r2, order_matters):
        return False
    tab1_cols = [{row[i] for row in r1} for i in range(n)]
    for perm in _col_permutations(tab1_cols, r2):
        if len(perm) != len(set(perm)):
            continue
        r2p = r2 if n == 1 else [tuple(row[j] for j in perm) for row in r2]
        if order_matters:
            if r1 == r2p:
                return True
        elif set(r1) == set(r2p) and Counter(r1) == Counter(r2p):
            return True
    return False


def exec_match(db_file: str | Path, pred_sql: str, gold_sql: str, timeout: float = 30.0) -> dict:
    """Execute both queries and compare. Returns a dict with 'correct' and diagnostics."""
    gold_rows, gold_err = execute_sql(db_file, gold_sql, timeout)
    pred_rows, pred_err = execute_sql(db_file, pred_sql, timeout)
    if gold_err is not None:
        return {"correct": False, "pred_error": pred_err, "gold_error": gold_err}
    if pred_err is not None:
        return {"correct": False, "pred_error": pred_err, "gold_error": None}
    order_matters = "order by" in gold_sql.lower()
    return {"correct": results_equal(gold_rows, pred_rows, order_matters),
            "pred_error": None, "gold_error": None}


# --------------------------------------------------------------------------- #
# Prediction post-processing
# --------------------------------------------------------------------------- #
def clean_sql(text: str) -> str:
    """Extract a single SQL statement from raw model output."""
    if text is None:
        return ""
    m = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        text = m.group(1)
    elif "```" in text:  # unterminated fence
        text = text.split("```", 1)[1]
        text = re.sub(r"^(sql|sqlite)\s", "", text, flags=re.I)
    text = text.strip()
    # if the model added prose first, start at the first SELECT/WITH
    m = re.search(r"\b(SELECT|WITH)\b", text, flags=re.I)
    if m:
        text = text[m.start():]
    text = text.split(";")[0]
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# Balanced dev subset
# --------------------------------------------------------------------------- #
def make_dev_subset(per_level: int = 50, seed: int = 42, out: Path = SUBSET_FILE) -> dict:
    dev = load_split("dev")
    rng = random.Random(seed)
    by_level = defaultdict(list)
    for ex in dev:
        by_level[ex["hardness"]].append(ex["idx"])
    idxs = []
    for lvl in HARDNESS_LEVELS:
        idxs.extend(sorted(rng.sample(by_level[lvl], per_level)))
    subset = {"description": f"Balanced Spider dev subset: {per_level} per difficulty level",
              "source": "dev.json", "seed": seed, "per_level": per_level,
              "counts": {lvl: per_level for lvl in HARDNESS_LEVELS},
              "dev_indices": idxs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=1))
    print(f"wrote {out} ({len(idxs)} examples)")
    return subset


def load_dev_subset(which: str = "balanced") -> list[dict]:
    """'balanced' -> fixed balanced subset; 'full' -> all 1034 dev examples."""
    dev = load_split("dev")
    if which == "full":
        return dev
    subset = json.loads(SUBSET_FILE.read_text())
    return [dev[i] for i in subset["dev_indices"]]


# --------------------------------------------------------------------------- #
# Scoring a predictions file
# --------------------------------------------------------------------------- #
def evaluate_predictions(preds: list[dict], timeout: float = 30.0) -> tuple[dict, list[dict]]:
    """preds: dicts with at least 'idx' (dev index) and 'pred_sql'.

    Returns (metrics, per_example). metrics has EX per difficulty + overall.
    """
    dev = load_split("dev")
    per_example = []
    for p in preds:
        ex = dev[p["idx"]]
        r = exec_match(db_path(ex["db_id"]), p["pred_sql"], ex["query"], timeout)
        per_example.append({"idx": ex["idx"], "db_id": ex["db_id"], "hardness": ex["hardness"],
                            "question": ex["question"], "gold_sql": ex["query"],
                            "pred_sql": p["pred_sql"], **r})
    metrics = summarize(per_example)
    return metrics, per_example


def summarize(per_example: list[dict]) -> dict:
    out = {}
    for lvl in HARDNESS_LEVELS + ["all"]:
        rows = [r for r in per_example if lvl == "all" or r["hardness"] == lvl]
        n = len(rows)
        correct = sum(r["correct"] for r in rows)
        errors = sum(r["pred_error"] is not None for r in rows)
        out[lvl] = {"n": n, "correct": correct, "ex": round(correct / n, 4) if n else None,
                    "exec_error_rate": round(errors / n, 4) if n else None}
    return out


def print_metrics(metrics: dict, title: str = "") -> None:
    if title:
        print(title)
    print(f"{'level':<8}{'n':>6}{'EX':>9}{'exec err':>10}")
    for lvl in HARDNESS_LEVELS + ["all"]:
        m = metrics[lvl]
        if m["n"]:
            print(f"{lvl:<8}{m['n']:>6}{m['ex'] * 100:>8.1f}%{m['exec_error_rate'] * 100:>9.1f}%")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def score_file(pred_file: str | Path, timeout: float = 30.0) -> dict:
    """Score predictions.jsonl; writes metrics.json and eval_details.jsonl next to it."""
    pred_file = Path(pred_file)
    metrics, per_example = evaluate_predictions(read_jsonl(pred_file), timeout)
    (pred_file.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_jsonl(pred_file.parent / "eval_details.jsonl", per_example)
    print_metrics(metrics, f"== {pred_file}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-subset")
    m.add_argument("--per-level", type=int, default=50)
    m.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("score")
    s.add_argument("pred_file")
    s.add_argument("--timeout", type=float, default=30.0)
    t = sub.add_parser("self-test")
    t.add_argument("--subset", choices=["balanced", "full"], default="full")
    args = p.parse_args()

    if args.cmd == "make-subset":
        make_dev_subset(args.per_level, args.seed)
    elif args.cmd == "score":
        score_file(args.pred_file, args.timeout)
    elif args.cmd == "self-test":
        exs = load_dev_subset(args.subset)
        metrics, details = evaluate_predictions([{"idx": e["idx"], "pred_sql": e["query"]} for e in exs])
        print_metrics(metrics, "gold vs gold (should be ~100%)")
        for d in details:
            if not d["correct"]:
                print("  not matched:", d["idx"], d["gold_error"] or d["pred_error"], "|", d["gold_sql"])


if __name__ == "__main__":
    main()
