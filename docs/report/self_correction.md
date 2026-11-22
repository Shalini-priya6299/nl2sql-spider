# Part 3: Self-correction (E3, E5) — report sections

> STATUS: E4/E5 numbers below come from the committed run summaries in `results/`
> (`config.json`, `metrics.json`, `cost_summary.md`, produced by
> `python analyze_self_correct.py <run> --save`). The E2 and E3 200-question runs are **not in
> this repository yet**; their cells are marked `[PENDING RUN FILES]` and must be filled from
> those runs' `metrics.json` / `cost_summary.md` (not from the interrupted 55-question test run).

## Self-correction methods

**Idea.** A text-to-SQL model writes SQL without ever running it. Spider databases are small
and local, so we can run the query and give the model what the database said. We call this
*execution-based self-correction* (in the spirit of the self-correction step of DIN-SQL [2]).

**Loop** (`self_correct.py`). For each question:

1. The model writes SQL from the normal prompt. This first answer is identical to what E2 (or E4)
   would produce, so any change in accuracy comes only from the retries.
2. The SQL is executed read-only with a 10 s timeout (`executor.py`). Final scoring uses the same
   30 s timeout as E1/E2 (`--eval-timeout`, recorded in `config.json`).
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
| Retry on an empty result | Often a wrong literal or condition. It is a weaker signal (an empty answer can be right), so it is a switch: `--retry-on-empty` (default) / `--no-retry-on-empty`. See the limitation below. |
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

**Results.** (balanced dev subset, 200 questions, greedy decoding, one seed)

| | EX | Exec-error rate | Extra LLM calls | Exec errors fixed (crash → runs) |
|---|---|---|---|---|
| E2 few-shot (no correction) | [PENDING RUN FILES] | [PENDING RUN FILES] | 0 | - |
| E3 few-shot + correction | [PENDING RUN FILES] | [PENDING RUN FILES] | [PENDING RUN FILES] | [PENDING RUN FILES] |
| E4 fine-tuned (no correction) | 69.0% | 12.0% | 0 | - |
| E5 fine-tuned + correction | 71.0% | 9.0% | 67 | 5 |

"Exec errors fixed" counts only questions whose first SQL crashed and whose final SQL runs. It
does not count queries that were wrong but ran (those show up in EX, below).

E5 measured on its own first answers (same 200 questions, `cost_summary.md`): EX 68.5% before →
71.0% after correction; exec-error rate 11.5% → 9.0%. 5 questions went wrong → right, 0 went
right → wrong. 36 of 200 questions were retried. (E4's committed 69.0% is one question above E5's
first-answer 68.5%; both are greedy runs on different hardware/library versions, so treat that as
noise.) E5 vs E4: +2.0 points, i.e. 4 questions.

Gain by difficulty, E5 first SQL → final SQL: easy 94 → 94, medium 82 → 84, hard 54 → 58,
extra 44 → 48.

**Cost (E5).** 67 extra LLM calls on 200 questions (0.335 per question, total LLM cost ×1.335;
267 calls in all). Total generation time was 8617 s on CPU, of which about 2162 s is the retries
(estimated from call counts, because per-call times are similar). SQL execution took 5.2 s in
total, so execution is negligible next to generation. E3: [PENDING RUN FILES].

**Limitation: retry on empty results.** `retry_on_empty` is on in the reported runs, so a correct
empty answer could be replaced by wrong rows. We did not run a full `--no-retry-on-empty`
ablation (it needs another multi-hour CPU run). We instead measured the risk directly from the E5
log: 13 questions had an empty first result, 5 of those were already correct, and **0** were
turned wrong by the retry (and 0 wrong ones were fixed by it). So on E5 the empty-result retry was neutral: no gain and no harm. This is 13 questions
on one seed, so it does not rule out harm elsewhere; the same check for E3 is
[PENDING RUN FILES] (`cost_summary.md` prints these counts). The ablation remains future work.

**Error analysis of 25 wrong E3 predictions.** [PENDING: needs the E3 run and manual labelling. run
`python error_analysis.py sample results/<e3 run>/eval_details.jsonl -n 25`, label the CSV, then
`python error_analysis.py summarize ...`; discuss how many are still execution errors versus
queries that run but are wrong.]

## Self-correction vs fine-tuning

| | Self-correction (E3) | Fine-tuning (E4) |
|---|---|---|
| What changes | Nothing in the model; extra calls at inference | The model's weights (LoRA, 4.3 GPU-hours on a T4) |
| One-time cost | None | Training time, GPU, data preparation |
| Cost per question | More LLM calls (E5: ×1.335; E3: [PENDING RUN FILES]) | Same as the base model |
| Fixes | Errors the database can report | Systematic habits (schema layout, SQL style, when to stop) |
| Needs | A database to run queries on | Labeled training pairs |

**Discussion.** What the committed E5 data supports so far:

- *Fine-tuning already removed many execution errors.* E4's exec-error rate is 12.0% against
  E1's 18.5%, so E5 had fewer crashes to fix than a base model would (11.5% on its first SQL).
- *Gains are small and additive but modest.* Correction on top of fine-tuning adds +2.5 points on
  E5's own first answers (68.5% → 71.0%, 5 questions), and +2.0 over E4 (4 questions). With one
  seed and 200 questions, a few questions is within noise.
- *It only fixes what the database can report.* Only 5 of the 23 first-SQL execution failures
  were turned into running queries; the remaining wrong answers are queries that run but are
  wrong, which execution feedback cannot detect.
- *Cost.* ×1.335 LLM calls for +2.5 points on the same questions; fine-tuning cost 4.3 GPU-hours
  once and no extra calls per question.

Still to answer once the E2/E3 files are added: [PENDING RUN FILES] which gave the bigger EX gain,
correction (E3 vs E2) or fine-tuning (E4 vs E1), and whether the extra inference cost of
correction is worth it compared with training. Both use one seed and a 200-question subset, so
differences of a few points are 3-6 questions.

## Weekly sync notes / progress

| Week | Done | Next |
|---|---|---|
| Nov 4-8 | Executor, self-correction loop, first E3 test run (55 questions, interrupted) | Fix interface mismatch, tests |
| Nov 9-14 | `executor.py`, `self_correct.py`, E5 runner option, cost analysis script, tests | Run E3 and E5 on Colab with the Part 4 adapter |
