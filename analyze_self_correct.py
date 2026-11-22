"""Cost and benefit analysis of self-correction (E3 / E5).

    python analyze_self_correct.py results/<run>/            # print
    python analyze_self_correct.py results/<run>/ --save     # also write cost_summary.json / .md

Answers the two questions of the Part 3 report section, from one run's predictions.jsonl:
  1. How many queries did self-correction fix?
       - execution errors fixed (first SQL crashed, final SQL runs)
       - EX before vs after: the FIRST SQL of every question is scored too (initial_sql),
         so the gain is measured on the same questions, with no second model run needed
       - queries made worse (first SQL was correct, final SQL is not)
  2. What did it cost?  extra LLM calls, generation seconds, execution seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import HARDNESS_LEVELS, evaluate_predictions, read_jsonl


def analyze(run_dir: Path) -> dict:
    preds = read_jsonl(run_dir / "predictions.jsonl")
    if not preds:
        raise SystemExit(f"{run_dir}/predictions.jsonl is empty")

    # Why we re-score initial_sql: the run only scored the FINAL sql. Scoring the first answer
    # with the same evaluator gives "no self-correction" on exactly the same questions, which
    # is a fairer before/after than comparing against a different run.
    initial_metrics, initial_rows = evaluate_predictions(
        [{"idx": p["idx"], "question": p["question"], "pred_sql": p["initial_sql"]} for p in preds]
    )
    final_metrics, final_rows = evaluate_predictions(
        [{"idx": p["idx"], "question": p["question"], "pred_sql": p["pred_sql"]} for p in preds]
    )
    init_ok = {r["idx"]: r["correct"] for r in initial_rows}
    final_ok = {r["idx"]: r["correct"] for r in final_rows}

    n = len(preds)
    retried = [p for p in preds if p["extra_llm_calls"] > 0]
    fixed_errors = sum(1 for p in preds if p.get("execution_error_fixed"))
    gained = sum(1 for i in init_ok if not init_ok[i] and final_ok[i])   # wrong -> right
    broken = sum(1 for i in init_ok if init_ok[i] and not final_ok[i])   # right -> wrong

    # Retry-on-empty diagnostic: an empty first result can be a correct answer. Count the
    # questions whose first SQL returned no rows, how many of those were already correct, and
    # how many of those correct ones the retry turned into wrong answers. This bounds the harm
    # of retry_on_empty without needing a second run with --no-retry-on-empty.
    empty_ids = {p["idx"] for p in preds if p.get("initial_empty")}
    empty_initial_correct = sum(1 for i in empty_ids if init_ok[i])
    empty_broken = sum(1 for i in empty_ids if init_ok[i] and not final_ok[i])
    empty_fixed = sum(1 for i in empty_ids if not init_ok[i] and final_ok[i])

    gen_s = sum(p["generation_time"] for p in preds)
    exec_s = sum(p["execution_time"] for p in preds)
    extra_calls = sum(p["extra_llm_calls"] for p in preds)

    # Why we estimate the cost of the first answers: the extra generation time is the time
    # spent on retries only. Calls have similar length, so first-answer time ~ n/(n + extra)
    # of the total. It is an estimate; the report should say so.
    first_share = n / (n + extra_calls)
    summary = {
        "run": run_dir.name,
        "n_questions": n,
        "ex_before": initial_metrics["all"]["ex"],
        "ex_after": final_metrics["all"]["ex"],
        "ex_by_level_before": {l: initial_metrics[l]["ex"] for l in HARDNESS_LEVELS},
        "ex_by_level_after": {l: final_metrics[l]["ex"] for l in HARDNESS_LEVELS},
        "exec_error_rate_before": initial_metrics["all"]["exec_error_rate"],
        "exec_error_rate_after": final_metrics["all"]["exec_error_rate"],
        "questions_retried": len(retried),
        "execution_errors_fixed": fixed_errors,  # crash -> runs only; not semantic fixes
        "empty_first_result": len(empty_ids),
        "empty_first_result_correct": empty_initial_correct,
        "empty_retry_broken": empty_broken,
        "empty_retry_fixed": empty_fixed,
        "wrong_to_right": gained,
        "right_to_wrong": broken,
        "extra_llm_calls": extra_calls,
        "extra_calls_per_question": round(extra_calls / n, 3),
        "total_generation_seconds": round(gen_s, 1),
        "est_extra_generation_seconds": round(gen_s * (1 - first_share), 1),
        "total_execution_seconds": round(exec_s, 2),
        "relative_llm_cost": round((n + extra_calls) / n, 3),  # 1.0 = no extra calls
    }
    return summary


def to_markdown(s: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    lines = [
        f"Self-correction analysis: {s['run']} ({s['n_questions']} questions)",
        "",
        "| | Before (first SQL) | After (final SQL) |",
        "|---|---|---|",
        f"| EX overall | {pct(s['ex_before'])} | {pct(s['ex_after'])} |",
        f"| Execution-error rate | {pct(s['exec_error_rate_before'])} | {pct(s['exec_error_rate_after'])} |",
    ]
    for lvl in HARDNESS_LEVELS:
        lines.append(f"| EX {lvl} | {pct(s['ex_by_level_before'][lvl])} | {pct(s['ex_by_level_after'][lvl])} |")
    lines += [
        "",
        f"- Questions retried: {s['questions_retried']}",
        f"- Execution errors fixed (first SQL crashed, final SQL runs; crash-to-working only, "
        f"not semantic fixes): {s['execution_errors_fixed']}",
        f"- Wrong -> right: {s['wrong_to_right']}; right -> wrong: {s['right_to_wrong']}",
        f"- First SQL returned no rows: {s['empty_first_result']} "
        f"(already correct: {s['empty_first_result_correct']}; "
        f"correct -> wrong after retry: {s['empty_retry_broken']}; "
        f"wrong -> right: {s['empty_retry_fixed']})",
        f"- Extra LLM calls: {s['extra_llm_calls']} ({s['extra_calls_per_question']} per question, "
        f"total LLM cost x{s['relative_llm_cost']})",
        f"- Generation time: {s['total_generation_seconds']} s total "
        f"(about {s['est_extra_generation_seconds']} s of it for retries, estimated)",
        f"- SQL execution time: {s['total_execution_seconds']} s",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--save", action="store_true", help="write cost_summary.json and cost_summary.md")
    args = ap.parse_args()

    summary = analyze(args.run_dir)
    text = to_markdown(summary)
    print(text)
    if args.save:
        (args.run_dir / "cost_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.run_dir / "cost_summary.md").write_text(text)
        print(f"saved to {args.run_dir}")


if __name__ == "__main__":
    main()
