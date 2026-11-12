"""Error analysis for incorrect predictions (E1: 25 examples; reusable for E2-E5).

1) Sample incorrect predictions into a CSV to annotate by hand:
       python error_analysis.py sample results/<run>/eval_details.jsonl -n 25
   -> results/<run>/error_analysis.csv  (fill the "category" column; "suggested" is a hint only)
2) After annotating, count the categories:
       python error_analysis.py summarize results/<run>/error_analysis.csv
   -> results/<run>/error_categories.md

Categories follow the error taxonomy of DIN-SQL (Pourreza & Rafiei, 2023),
plus two categories for value and output-format errors.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from evaluate import HARDNESS_LEVELS, read_jsonl

CATEGORIES = {
    "schema_linking": "wrong or non-existent table/column (incl. 'no such column')",
    "join": "missing, extra or wrong JOIN / join condition",
    "group_by": "missing or wrong GROUP BY / aggregation",
    "nested_set_ops": "wrong nesting or INTERSECT/EXCEPT/UNION",
    "condition_value": "wrong WHERE/HAVING operator or literal value (case, spelling, quotes)",
    "select_output": "wrong/extra/missing SELECT columns, DISTINCT, ORDER BY or LIMIT",
    "invalid_sql": "syntax error, truncated output, or not SQL at all",
    "gold_issue": "prediction is a valid reading of the question; gold is ambiguous/wrong",
}


def suggest(row: dict) -> str:
    err = (row.get("pred_error") or "").lower()
    if "no such column" in err or "no such table" in err or "ambiguous column" in err:
        return "schema_linking"
    if err:
        return "invalid_sql"
    return ""


def sample(details_file: Path, n: int, seed: int) -> Path:
    rows = [r for r in read_jsonl(details_file) if not r["correct"] and not r.get("gold_error")]
    rng = random.Random(seed)
    # stratify by difficulty so all levels are represented
    by_level = defaultdict(list)
    for r in rows:
        by_level[r["hardness"]].append(r)
    picked, levels = [], [lvl for lvl in HARDNESS_LEVELS if by_level[lvl]]
    for lvl in levels:
        rng.shuffle(by_level[lvl])
    while len(picked) < min(n, len(rows)):
        for lvl in levels:
            if by_level[lvl] and len(picked) < n:
                picked.append(by_level[lvl].pop())
    picked.sort(key=lambda r: (HARDNESS_LEVELS.index(r["hardness"]), r["idx"]))

    out = details_file.parent / "error_analysis.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "db_id", "hardness", "question", "gold_sql", "pred_sql", "pred_error",
                    "suggested", "category", "notes"])
        for r in picked:
            w.writerow([r["idx"], r["db_id"], r["hardness"], r["question"], r["gold_sql"],
                        r["pred_sql"], r.get("pred_error") or "", suggest(r), "", ""])
    print(f"wrote {out} ({len(picked)} of {len(rows)} incorrect). Fill the 'category' column with one of:")
    for k, v in CATEGORIES.items():
        print(f"  {k:<16} {v}")
    return out


def summarize(csv_file: Path) -> Path:
    with open(csv_file) as f:
        rows = list(csv.DictReader(f))
    missing = [r["idx"] for r in rows if not r["category"].strip()]
    if missing:
        raise SystemExit(f"{len(missing)} rows have no category yet (idx {missing})")
    bad = {r["category"] for r in rows} - set(CATEGORIES)
    if bad:
        raise SystemExit(f"unknown categories: {bad}. Allowed: {list(CATEGORIES)}")

    total = Counter(r["category"] for r in rows)
    by_level = {lvl: Counter(r["category"] for r in rows if r["hardness"] == lvl) for lvl in HARDNESS_LEVELS}
    lines = ["| Category | Description | " + " | ".join(HARDNESS_LEVELS) + " | Total | % |",
             "|---|---|" + "---|" * (len(HARDNESS_LEVELS) + 2)]
    for cat, cnt in total.most_common():
        lines.append(f"| {cat} | {CATEGORIES[cat]} | "
                     + " | ".join(str(by_level[lvl][cat]) for lvl in HARDNESS_LEVELS)
                     + f" | {cnt} | {100 * cnt / len(rows):.0f}% |")
    out = csv_file.parent / "error_categories.md"
    out.write_text(f"Error categories ({len(rows)} incorrect predictions, {csv_file.parent.name})\n\n"
                   + "\n".join(lines) + "\n")
    print(out.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("details_file", type=Path)
    s.add_argument("-n", type=int, default=25)
    s.add_argument("--seed", type=int, default=42)
    m = sub.add_parser("summarize")
    m.add_argument("csv_file", type=Path)
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.details_file, args.n, args.seed)
    else:
        summarize(args.csv_file)


if __name__ == "__main__":
    main()
