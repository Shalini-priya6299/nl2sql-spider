# Part 3: Self-correction (E3, E5) — report sections

> DRAFT. Every `[FILL: ...]` must be replaced with a number from the final runs
> (`results/<run>/cost_summary.md`, produced by `python analyze_self_correct.py <run> --save`).
> Do not report numbers from the interrupted 55-question test run.

## Self-correction methods

**Idea.** A text-to-SQL model writes SQL without ever running it. Spider databases are small
and local, so we can run the query and give the model what the database said. We call this
*execution-based self-correction* (in the spirit of the self-correction step of DIN-SQL [2]).

**Loop** (`self_correct.py`). For each question:

1. The model writes SQL from the normal prompt. This first answer is identical to what E2 (or E4)
   would produce, so any change in accuracy comes only from the retries.
2. The SQL is executed read-only with a 10 s timeout (`executor.py`).
3. If it failed (SQLite error or timeout) or returned no rows, the conversation is extended with
   the model's previous SQL and the feedback (for example `SQLite error: no such column: nme`),
   and the model is asked for a corrected query.
4. This repeats up to 2 times (so at most 3 LLM calls per question). Then we stop.
5. The final answer is the last query that ran without error; if none did, the last query.

**Design choices and why.**

| Choice | Reason |
|---|---|
| Read-only connection, `SELECT`/`WITH` only, timeout, row cap | The SQL comes from a model, so it is never trusted; one runaway query must not stop a multi-hour run. |
| Feedback is the raw SQLite message | It names the wrong table or column, which is far more useful than "try again". |
| Retry on an empty result | Often a wrong literal or condition. It is a weaker signal (an empty answer can be right), so it is a switch (`retry_on_empty`). |
| Prompt rebuilt from the original messages + one failed attempt | Keeps the context short for a 1.5B model and avoids a chain of old mistakes. |
| At most 2 retries | The issue asks for 2-3; every retry costs a full LLM call. |
| Keep the best executing attempt | A retry can never turn a working query into a crashing one. |

**What it can and cannot fix.** The model only sees errors, never the gold answer. It can repair
wrong names, bad syntax and some wrong conditions. It cannot notice a query that runs and returns
a plausible but wrong answer (wrong join, wrong aggregation), which is where the model's mistakes
concentrate (see the E1/E4 error analyses).

**Experiments.**

| ID | Model | Prompt | Retries |
|---|---|---|---|
| E3 | Qwen2.5-Coder-1.5B-Instruct (base) | few-shot, 3 similar examples (as E2) | up to 2 |
| E5 | same + Part 4 LoRA adapter | zero-shot (the prompt the adapter was trained on, as E4) | up to 2 |

**Results.** (balanced dev subset, 200 questions, greedy decoding)

| | EX | Exec-error rate | Extra LLM calls | Errors fixed |
|---|---|---|---|---|
| E2 few-shot (no correction) | [FILL] | [FILL] | 0 | - |
| E3 few-shot + correction | [FILL] | [FILL] | [FILL] | [FILL] |
| E4 fine-tuned (no correction) | 69.0% | 12.0% | 0 | - |
| E5 fine-tuned + correction | [FILL] | [FILL] | [FILL] | [FILL] |

**Cost.** E3: [FILL] extra calls (x[FILL] LLM cost), about [FILL] s extra generation, [FILL] s of
SQL execution. E5: [FILL]. (Generation time of retries is estimated from the call counts.)

**Error analysis of 25 wrong E3 predictions.** [FILL: run
`python error_analysis.py sample results/<e3 run>/eval_details.jsonl -n 25`, label the CSV, then
`python error_analysis.py summarize ...`; discuss how many are still execution errors versus
queries that run but are wrong.]

## Self-correction vs fine-tuning

| | Self-correction (E3) | Fine-tuning (E4) |
|---|---|---|
| What changes | Nothing in the model; extra calls at inference | The model's weights (LoRA, 4.3 GPU-hours on a T4) |
| One-time cost | None | Training time, GPU, data preparation |
| Cost per question | More LLM calls (x[FILL]) | Same as the base model |
| Fixes | Errors the database can report | Systematic habits (schema layout, SQL style, when to stop) |
| Needs | A database to run queries on | Labeled training pairs |

**Discussion.** [FILL after the runs. Points to answer with the numbers:
(1) which gave the bigger EX gain, correction (E3 vs E2) or fine-tuning (E4 vs E1)?
(2) did fine-tuning already remove most execution errors (E4 exec-error rate 12% vs E1 18.5%), so
that E5 has less to fix than E3 had?
(3) are the two gains additive (E5 vs E4)?
(4) is the extra inference cost of correction worth the gain compared with the one-time training
cost? Mention that both use one seed and a 200-question subset, so differences of a few points
are 3-6 questions.]

## Weekly sync notes / progress

| Week | Done | Next |
|---|---|---|
| Nov 4-8 | Executor, self-correction loop, first E3 test run (55 questions, interrupted) | Fix interface mismatch, tests |
| Nov 9-14 | `executor.py`, `self_correct.py`, E5 runner option, cost analysis script, tests | Run E3 and E5 on Colab with the Part 4 adapter |
