# Part 3: Self-correction (E3, E5) — report sections

> STATUS: all four runs (E2, E3, E4, E5) are in. Numbers come from the run summaries in
> `results/` (`config.json`, `metrics.json`, `cost_summary.md`, produced by
> `python analyze_self_correct.py <run> --save`). Only the manual category labels of the E3
> error analysis are still open.

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
| E2 few-shot (no correction) | 51.5% | 31.5% | 0 | - |
| E3 few-shot + correction | 56.5% | 23.0% | 137 | 15 |
| E4 fine-tuned (no correction) | 69.0% | 12.0% | 0 | - |
| E5 fine-tuned + correction | 71.0% | 9.0% | 67 | 5 |

"Exec errors fixed" counts only questions whose first SQL crashed and whose final SQL runs. It
does not count queries that were wrong but ran (those show up in EX, below).

E3 measured on its own first answers (same 200 questions, `results/e3_gpu_balanced/cost_summary.md`):
EX 52.0% → 56.5% after correction; exec-error rate 30.5% → 23.0%. 9 questions went wrong → right,
0 went right → wrong; 75 of 200 questions were retried. Gain by difficulty (first → final):
easy 82 → 88, medium 60 → 64, hard 34 → 40, extra 32 → 34. E3's first answers (52.0%, 30.5%) agree with
the separate E2 run (51.5%, 31.5%) to within one question, since both use the same prompt and
greedy decoding. Comparing E3 with E2 directly, correction adds +5.0 points (51.5% → 56.5%, 10
questions) and lowers the exec-error rate by 8.5 points (31.5% → 23.0%).

**Note on the E3 score.** The Colab run scored 57.5% (115/200). Re-scoring the same
predictions locally gives 56.5% (113/200): two `wta_1` questions (idx 453, 459) are judged
correct on the Colab copy of `wta_1.sqlite` and wrong on the local one. The Colab copy had to be
recreated after the file went missing, so the two copies differ. We report the local score
because for idx 453 ("oldest player") the prediction orders `birth_date DESC`, which only matches
the gold answer on a database where the DESC order happens to give the same row. All E3 numbers
here are from the local database; E1, E4 and E5 must be checked against the same `wta_1.sqlite`.

E5 measured on its own first answers (same 200 questions, `cost_summary.md`): EX 68.5% before →
71.0% after correction; exec-error rate 11.5% → 9.0%. 5 questions went wrong → right, 0 went
right → wrong. 36 of 200 questions were retried. (E4's committed 69.0% is one question above E5's
first-answer 68.5%; both are greedy runs on different hardware/library versions, so treat that as
noise.) E5 vs E4: +2.0 points, i.e. 4 questions.

Gain by difficulty, E5 first SQL → final SQL: easy 94 → 94, medium 82 → 84, hard 54 → 58,
extra 44 → 48.

**Cost.** E2 needed one call per question (633.7 s on a T4 for 200 questions).

**Cost (E5).** 67 extra LLM calls on 200 questions (0.335 per question, total LLM cost ×1.335;
267 calls in all). Total generation time was 8617 s on CPU, of which about 2162 s is the retries
(estimated from call counts, because per-call times are similar). SQL execution took 5.2 s in
total, so execution is negligible next to generation.

**Cost (E3).** 137 extra LLM calls on 200 questions (0.685 per question, total LLM cost ×1.685;
337 calls in all). Total generation time was 906.5 s on a Colab T4, of which about 368 s is the
retries (estimated); SQL execution took 0.17 s. E3 needs about twice the retry calls of E5
(0.685 vs 0.335 per question) because its first answers crash more often (30.5% vs 11.5%).
The E3 and E5 times are not comparable: E3 ran on a GPU, E5 on a CPU.

**Limitation: retry on empty results.** `retry_on_empty` is on in the reported runs, so a correct
empty answer could be replaced by wrong rows. We did not run a full `--no-retry-on-empty`
ablation (it needs another multi-hour CPU run). We instead measured the risk directly from the E5
log: 13 questions had an empty first result, 5 of those were already correct, and **0** were
turned wrong by the retry (and 0 wrong ones were fixed by it). So on E5 the empty-result retry was neutral: no gain and no harm. This is 13 questions
on one seed, so it does not rule out harm elsewhere; on E3, 14 first results
were empty, 5 of them already correct, and 0 were turned wrong (0 were fixed either). The ablation remains future work.

**Error analysis of 25 wrong E3 predictions.** A sample of 25 wrong E3 predictions
(`results/e3_gpu_balanced/25_incorrect_predictions.csv`, 17 medium, 6 easy, 2 hard):
12 still end in an execution error after the retries (7 `no such column`, 5 `no such table`) and
13 run but return the wrong answer. So about half of the remaining errors are ones the retries
could in principle fix and did not (the model repeats or swaps the wrong name), and half are
logic errors that the database feedback cannot detect. [PENDING: manual category labels
(schema linking, join, aggregation, ...) with `python error_analysis.py`.]

## Self-correction vs fine-tuning

| | Self-correction (E3) | Fine-tuning (E4) |
|---|---|---|
| What changes | Nothing in the model; extra calls at inference | The model's weights (LoRA, 4.3 GPU-hours on a T4) |
| One-time cost | None | Training time, GPU, data preparation |
| Cost per question | More LLM calls (E3: ×1.685; E5: ×1.335) | Same as the base model |
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

Findings from E3 and E5 together:

- *Correction helps the weaker model more.* E3 gains +4.5 points over its own first answers
  (9 questions, 15 crashes fixed) for ×1.685 calls; E5 gains +2.5 for ×1.335.
- *Fine-tuning gives the larger lift.* E4 vs E1 is +8.5 points (69.0 vs 60.5) with no extra
  calls; E3's correction on top of few-shot ends at 56.5%, still below E4 (69.0%) and E5 (71.0%).
  Compared with zero-shot, the few-shot prompt itself did not help: E2 is 51.5% against E1's 60.5%,
  and its exec-error rate is 31.5% against 18.5%. So E3's correction mostly recovers what the
  few-shot prompt lost, and E3 (56.5%) ends below plain zero-shot (E1, 60.5%) on this model.
- *Gains overlap, they do not simply add.* Correction fixes crashes; fine-tuning already
  removed many of them (first-answer crash rate 30.5% for E3 vs 11.5% for E5).

Whether the extra inference cost is worth it compared with training depends on volume:
correction costs nothing up front but multiplies per-question cost, fine-tuning is a one-time
4.3 GPU-hours. Both use one seed and a 200-question subset, so
differences of a few points are 3-6 questions.

## Weekly sync notes / progress

| Week | Done | Next |
|---|---|---|
| Nov 4-8 | Executor, self-correction loop, first E3 test run (55 questions, interrupted) | Fix interface mismatch, tests |
| Nov 9-14 | `executor.py`, `self_correct.py`, E5 runner option, cost analysis script, tests | Run E3 and E5 on Colab with the Part 4 adapter |
