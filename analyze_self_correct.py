"""Cost and benefit analysis of self-correction (E3 / E5).
# This is the module-level documentation string describing what this script does.

    python analyze_self_correct.py results/<run>/            # print
    # This command runs the script for a particular result directory and prints the analysis.

    python analyze_self_correct.py results/<run>/ --save     # also write cost_summary.json / .md
    # This command runs the analysis and also saves the results into JSON and Markdown files.

Answers the two questions of the Part 3 report section, from one run's predictions.jsonl:
# The script answers two main questions for the Part 3 report using predictions.jsonl.

  1. How many queries did self-correction fix?
  # First question: determine how many queries became correct because of self-correction.

       - execution errors fixed (first SQL crashed, final SQL runs)
       # Count cases where the initial SQL failed during execution but the final SQL executed successfully.

       - EX before vs after: the FIRST SQL of every question is scored too (initial_sql),
       # Compare execution accuracy of the initial SQL and final SQL for the same questions.

         so the gain is measured on the same questions, with no second model run needed
         # This makes the comparison fair because both before and after results use the same questions.

       - queries made worse (first SQL was correct, final SQL is not)
       # Also count cases where self-correction changed a correct answer into an incorrect one.

  2. What did it cost?  extra LLM calls, generation seconds, execution seconds.
  # Second question: measure the additional computational cost caused by self-correction.
"""

from __future__ import annotations
# Enables postponed evaluation of type annotations.

import argparse
# Imports argparse so command-line arguments can be read.

import json
# Imports json so the summary can be saved as a JSON file.

from pathlib import Path
# Imports Path for safely working with file and directory paths.

from evaluate import HARDNESS_LEVELS, evaluate_predictions, read_jsonl
# Imports the hardness levels, evaluation function, and JSONL reader from evaluate.py.


def analyze(run_dir: Path) -> dict:
    # Defines the main analysis function and accepts the result directory as a Path object.

    preds = read_jsonl(run_dir / "predictions.jsonl")
    # Reads predictions.jsonl from the given run directory.

    if not preds:
        # Checks whether the predictions file contains any predictions.

        raise SystemExit(f"{run_dir}/predictions.jsonl is empty")
        # Stops the program with an error message if there are no predictions.

    # Why we re-score initial_sql: the run only scored the FINAL sql. Scoring the first answer
    # with the same evaluator gives "no self-correction" on exactly the same questions, which is
    # a fairer before/after than comparing against a different run.
    # We evaluate the initial SQL ourselves because the original run only evaluated the final SQL.
    # This allows a fair before-vs-after comparison using exactly the same questions.


    initial_metrics, initial_rows = evaluate_predictions(
        # Evaluates the initial SQL answers.

        [{"idx": p["idx"], "question": p["question"], "pred_sql": p["initial_sql"]} for p in preds]
        # Creates a list containing each question ID, question text, and its initial SQL.
    )
    # Stores both overall metrics and individual question-level evaluation results.

    final_metrics, final_rows = evaluate_predictions(
        # Evaluates the final SQL answers after self-correction.

        [{"idx": p["idx"], "question": p["question"], "pred_sql": p["pred_sql"]} for p in preds]
        # Creates a list containing each question ID, question text, and final SQL.
    )
    # Stores the metrics and individual results for the final SQL answers.

    init_ok = {r["idx"]: r["correct"] for r in initial_rows}
    # Creates a dictionary mapping each question ID to whether its initial SQL was correct.

    final_ok = {r["idx"]: r["correct"] for r in final_rows}
    # Creates a dictionary mapping each question ID to whether its final SQL was correct.

    n = len(preds)
    # Stores the total number of questions.

    retried = [p for p in preds if p["extra_llm_calls"] > 0]
    # Selects all questions for which at least one additional LLM call was made.

    fixed_errors = sum(1 for p in preds if p.get("execution_error_fixed"))
    # Counts questions where the initial SQL caused an execution error and self-correction fixed it.

    gained = sum(1 for i in init_ok if not init_ok[i] and final_ok[i])
    # Counts questions that changed from incorrect initially to correct finally.

    broken = sum(1 for i in init_ok if init_ok[i] and not final_ok[i])
    # Counts questions that changed from correct initially to incorrect finally.

    # Retry-on-empty diagnostic: an empty first result can be a correct answer. Count the
    # questions whose first SQL returned no rows, how many of those were already correct, and
    # how many of those correct ones the retry turned into wrong answers. This bounds the harm
    # of retry_on_empty without needing a second run with --no-retry-on-empty.
    # These comments explain that an empty SQL result is not automatically incorrect.
    # Some questions legitimately have no matching rows.


    empty_ids = {p["idx"] for p in preds if p.get("initial_empty")}
    # Creates a set containing IDs where the initial SQL returned no rows.

    empty_initial_correct = sum(1 for i in empty_ids if init_ok[i])
    # Counts how many empty initial results were actually correct answers.

    empty_broken = sum(1 for i in empty_ids if init_ok[i] and not final_ok[i])
    # Counts cases where an initially correct empty result became incorrect after retrying.

    empty_fixed = sum(1 for i in empty_ids if not init_ok[i] and final_ok[i])
    # Counts cases where an initially incorrect empty result became correct after retrying.

    gen_s = sum(p["generation_time"] for p in preds)
    # Adds together the LLM generation time for all questions.

    exec_s = sum(p["execution_time"] for p in preds)
    # Adds together SQL execution time for all questions.

    extra_calls = sum(p["extra_llm_calls"] for p in preds)
    # Counts the total number of additional LLM calls made because of self-correction.

    # Why we estimate the cost of the first answers: the extra generation time is the time
    # spent on retries only. Calls have similar length, so first-answer time ~ n/(n + extra)
    # of the total. It is an estimate; the report should say so.
    # The script estimates how much generation time belongs specifically to retry calls.
    # This is an approximation because it assumes the calls have similar generation cost.


    first_share = n / (n + extra_calls)
    # Estimates the fraction of all LLM generation calls that correspond to the first answers.

    summary = {
        # Creates a dictionary containing all analysis results.

        "run": run_dir.name,
        # Stores the name of the current run.

        "n_questions": n,
        # Stores the total number of questions analyzed.

        "ex_before": initial_metrics["all"]["ex"],
        # Stores execution accuracy before self-correction.

        "ex_after": final_metrics["all"]["ex"],
        # Stores execution accuracy after self-correction.

        "ex_by_level_before": {l: initial_metrics[l]["ex"] for l in HARDNESS_LEVELS},
        # Stores execution accuracy before correction for each difficulty level.

        "ex_by_level_after": {l: final_metrics[l]["ex"] for l in HARDNESS_LEVELS},
        # Stores execution accuracy after correction for each difficulty level.

        "exec_error_rate_before": initial_metrics["all"]["exec_error_rate"],
        # Stores the SQL execution-error rate before self-correction.

        "exec_error_rate_after": final_metrics["all"]["exec_error_rate"],
        # Stores the SQL execution-error rate after self-correction.

        "questions_retried": len(retried),
        # Stores how many questions triggered at least one retry.

        "execution_errors_fixed": fixed_errors,
        # Stores how many execution errors were fixed by self-correction.

        "empty_first_result": len(empty_ids),
        # Stores how many questions had an empty result from the initial SQL.

        "empty_first_result_correct": empty_initial_correct,
        # Stores how many of those empty initial results were actually correct.

        "empty_retry_broken": empty_broken,
        # Stores how many initially correct empty results became wrong after retry.

        "empty_retry_fixed": empty_fixed,
        # Stores how many initially wrong empty results became correct after retry.

        "wrong_to_right": gained,
        # Stores the number of questions that changed from wrong to right.

        "right_to_wrong": broken,
        # Stores the number of questions that changed from right to wrong.

        "extra_llm_calls": extra_calls,
        # Stores the total number of additional LLM calls.

        "extra_calls_per_question": round(extra_calls / n, 3),
        # Calculates the average number of extra LLM calls per question.

        "total_generation_seconds": round(gen_s, 1),
        # Stores the total LLM generation time rounded to one decimal place.

        "est_extra_generation_seconds": round(gen_s * (1 - first_share), 1),
        # Estimates how many generation seconds were spent on retry calls.

        "total_execution_seconds": round(exec_s, 2),
        # Stores total SQL execution time rounded to two decimal places.

        "relative_llm_cost": round((n + extra_calls) / n, 3),
        # Calculates relative LLM cost, where 1.0 means no additional calls.
    }
    # Finishes the summary dictionary.

    return summary
    # Returns the completed analysis results.


def to_markdown(s: dict) -> str:
    # Defines a function that converts the summary dictionary into Markdown text.

    def pct(x):
        # Defines a helper function for converting a decimal value into a percentage.

        return "n/a" if x is None else f"{100 * x:.1f}%"
        # Returns "n/a" for missing values; otherwise converts the value to a percentage.

    lines = [
        # Starts a list containing the Markdown lines.

        f"Self-correction analysis: {s['run']} ({s['n_questions']} questions)",
        # Adds the title containing the run name and number of questions.

        "",
        # Adds an empty line for Markdown formatting.

        "| | Before (first SQL) | After (final SQL) |",
        # Creates the Markdown table header.

        "|---|---|---|",
        # Creates the Markdown table separator.

        f"| EX overall | {pct(s['ex_before'])} | {pct(s['ex_after'])} |",
        # Adds overall execution accuracy before and after correction.

        f"| Execution-error rate | {pct(s['exec_error_rate_before'])} | {pct(s['exec_error_rate_after'])} |",
        # Adds execution-error rates before and after correction.
    ]
    # Finishes the initial Markdown table lines.

    for lvl in HARDNESS_LEVELS:
        # Loops through every difficulty level.

        lines.append(
            f"| EX {lvl} | {pct(s['ex_by_level_before'][lvl])} | {pct(s['ex_by_level_after'][lvl])} |"
        )
        # Adds execution accuracy for the current difficulty level.

    lines += [
        # Adds additional analysis information after the table.

        "",
        # Adds an empty line.

        f"- Questions retried: {s['questions_retried']}",
        # Shows how many questions received at least one retry.

        f"- Execution errors fixed (first SQL crashed, final SQL runs; crash-to-working only, "
        f"not semantic fixes): {s['execution_errors_fixed']}",
        # Shows how many execution errors were fixed.

        f"- Wrong -> right: {s['wrong_to_right']}; right -> wrong: {s['right_to_wrong']}",
        # Shows how many answers improved and how many became worse.

        f"- First SQL returned no rows: {s['empty_first_result']} "
        f"(already correct: {s['empty_first_result_correct']}; "
        f"correct -> wrong after retry: {s['empty_retry_broken']}; "
        f"wrong -> right: {s['empty_retry_fixed']})",
        # Shows the diagnostic information for queries whose first SQL returned no rows.

        f"- Extra LLM calls: {s['extra_llm_calls']} ({s['extra_calls_per_question']} per question, "
        f"total LLM cost x{s['relative_llm_cost']})",
        # Shows total extra calls, average extra calls per question, and relative cost.

        f"- Generation time: {s['total_generation_seconds']} s total "
        f"(about {s['est_extra_generation_seconds']} s of it for retries, estimated)",
        # Shows total generation time and estimated generation time spent on retries.

        f"- SQL execution time: {s['total_execution_seconds']} s",
        # Shows total time spent executing SQL queries.
    ]
    # Finishes the Markdown content.

    return "\n".join(lines) + "\n"
    # Joins all lines together with newline characters and adds a final newline.


def main() -> None:
    # Defines the main function of the program.

    ap = argparse.ArgumentParser(
        # Creates an argument parser for command-line arguments.

        description=__doc__,
        # Uses the module documentation string as the command-line description.

        formatter_class=argparse.RawDescriptionHelpFormatter
        # Keeps the formatting of the multi-line description.
    )

    ap.add_argument("run_dir", type=Path)
    # Adds a required command-line argument specifying the result directory.

    ap.add_argument(
        "--save",
        action="store_true",
        help="write cost_summary.json and cost_summary.md"
    )
    # Adds an optional --save flag that saves the analysis results to files.

    args = ap.parse_args()
    # Reads and stores the command-line arguments.

    summary = analyze(args.run_dir)
    # Runs the analysis on the specified result directory.

    text = to_markdown(summary)
    # Converts the analysis results into Markdown format.

    print(text)
    # Prints the Markdown report to the terminal.

    if args.save:
        # Checks whether the user provided the --save option.

        (args.run_dir / "cost_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        # Converts the summary dictionary into formatted JSON and saves it.

        (args.run_dir / "cost_summary.md").write_text(text)
        # Saves the Markdown report to cost_summary.md.

        print(f"saved to {args.run_dir}")
        # Prints a confirmation message showing where the files were saved.


if __name__ == "__main__":
    # Checks whether this Python file is being executed directly.

    main()
    # Calls the main function to start the program.
