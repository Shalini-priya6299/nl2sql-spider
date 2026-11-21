# Report sections (Part 1): Dataset and Experimental Setup

## 3. Dataset

We use Spider 1.0 [Yu et al., 2018], a large, human-annotated, cross-domain Text-to-SQL benchmark.
Each example pairs an English question with a SQL query over one of 200 relational databases
covering 138 domains. Spider is **cross-database**: the databases in the development set do not
appear in the training set, so a model must generalise to schemas it has never seen.
We confirmed that no database is shared between our training and evaluation data.

| Split | Questions | Databases | Avg. tables / DB | Avg. columns / DB |
|---|---:|---:|---:|---:|
| Train (train_spider + train_others) | 8,659 | 146 | 5.4 | 27.8 |
| Dev | 1,034 | 20 | 4.0 | 22.1 |

The official test set is hidden, so, following common practice, we report results on the dev set.
(Two dev questions, e.g. "Count the number of documents.", have exactly the same text and SQL as a
training question. They are asked about different databases, so they are generic questions and not leakage.)

**Difficulty levels.** Spider divides queries into four difficulty levels based on how many SQL
components they use (joins, GROUP BY, ORDER BY, nesting, set operations, aggregations, etc.). We compute the levels with
the official Spider evaluation rules and reproduce the official dev counts exactly:

| Split | Easy | Medium | Hard | Extra hard |
|---|---:|---:|---:|---:|
| Train | 1,983 (22.9%) | 2,999 (34.6%) | 1,922 (22.2%) | 1,755 (20.3%) |
| Dev | 248 (24.0%) | 446 (43.1%) | 174 (16.8%) | 166 (16.1%) |

**Balanced evaluation subset.** The dev set is dominated by medium queries, and running every
experiment (including multiple self-correction rounds and several LoRA settings) on all 1,034 questions
is expensive on free GPUs. We therefore fix a balanced subset of 200 dev questions, 50 per
difficulty level, sampled once with seed 42 (`splits/dev_subset_balanced.json`). It covers 19 of the 20 dev
databases. All experiments are compared on this same subset. Its overall EX is a macro-average over
difficulty levels, not an estimate of full-dev accuracy, so we also report E1 on the full dev set (63.9% EX overall vs. 60.5% on the balanced subset;
see `results/summary.md`).

## 4. Experimental Setup

**Task formulation.** Given a natural-language question and the schema of the target database,
the model generates one SQLite query. No gold SQL, database values (other than the sample rows
below) or question-specific hints are given.

**Schema representation.** Following Rajkumar et al. (2022), each table is shown as a
`CREATE TABLE` statement with column types, primary keys and foreign keys, built from Spider's
`tables.json` using the original table and column names. Each statement is followed by three example rows
(`SELECT * FROM <table> LIMIT 3`) so the model can see value formats such as capitalisation and date
styles. Prompts are 940–5,490 characters long (mean ≈2,100) on the evaluation subset.

**Model.** Qwen2.5-Coder-1.5B-Instruct, a small open code LLM that fits
on a free 16 GB T4 GPU in half precision. The same model is the base for few-shot (E2),
self-correction (E3/E5) and LoRA fine-tuning (E4), so differences come from the method, not the model.

**E1: zero-shot.** A system instruction ("write one SQLite query … return only the SQL") followed by
the schema, the question and the prefix `-- SQL:` (prompt version `zeroshot-v1`). Decoding is greedy
(no sampling) with at most 256 new tokens. Raw outputs are post-processed by taking the content of a
Markdown code block if one is present, starting at the first `SELECT`/`WITH`, and cutting at the first `;`.

**Evaluation metric.** Execution accuracy (EX): the predicted and gold queries are executed on
the database (read-only, 30 s timeout) and the prediction is correct if the results match. Results are compared
with the rules of the official test-suite evaluator [Zhong et al., 2020]: rows are compared as a
multiset, or as an ordered list when the gold query contains `ORDER BY`, and column order is ignored.
A prediction that fails to execute is counted as incorrect. We also report the execution-error rate.
As a sanity check, scoring every gold query against itself gives 100% EX on the full dev set. We do not
use the distilled test-suite databases, so our EX can be slightly higher than test-suite accuracy
(a wrong query can return the right result by chance on a single database).

**Error analysis.** We sample 25 incorrect E1 predictions, stratified by difficulty, and label each
with one category adapted from DIN-SQL's error taxonomy [Pourreza & Rafiei, 2023]: schema linking,
JOIN, GROUP BY, nesting/set operations, condition values, SELECT/output format, invalid SQL, and
questionable gold.

**E1 results.** Accuracy drops steadily as queries get harder, and execution errors rise:

| Eval set | n | Easy | Medium | Hard | Extra hard | Overall | Exec errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced subset | 200 | 88.0 | 64.0 | 50.0 | 40.0 | **60.5** | 18.5% |
| Full dev | 1,034 | 82.7 | 65.5 | 51.1 | 45.2 | **63.9** | 17.7% |

**E1 error categories** (25 incorrect predictions on the balanced subset, stratified by difficulty):

| Category | Easy | Medium | Hard | Extra | Total |
|---|---:|---:|---:|---:|---:|
| Schema linking | 5 | 3 | 1 | 3 | 12 (48%) |
| Nesting / set operations | 0 | 0 | 3 | 1 | 4 (16%) |
| JOIN | 0 | 2 | 1 | 0 | 3 (12%) |
| Condition values | 0 | 2 | 0 | 1 | 3 (12%) |
| GROUP BY | 0 | 0 | 1 | 1 | 2 (8%) |
| Questionable gold | 1 | 0 | 0 | 0 | 1 (4%) |

Schema linking is the main failure, and 8 of these 12 are queries that do not execute at all
(e.g. a column taken from the wrong table alias: `SELECT T1.Title ... FROM singer AS T1 JOIN song AS T2`,
where `Title` belongs to `song`). These errors provide explicit SQLite feedback and motivate evaluating whether
execution feedback (E3/E5) can correct schema-linking mistakes. The most common error on hard and extra-hard queries is missing nesting
or set operations (4 of the 8 hard/extra errors that are not schema linking): for "citizenships shared by singers born before 1945 and after 1955" the model writes
`WHERE Birth_Year < 1945 AND Birth_Year > 1955` (always empty) instead of an `INTERSECT`, and for
"students who do not have a cat" it writes `PetType != 'cat'` instead of `NOT IN (...)`. These queries run
without error but return the wrong rows, so execution feedback alone cannot detect them. Condition-value errors
include a wrong literal (`'US'` instead of `'US Territory'`) and a missing `IsOfficial = 'T'` filter.
One error is arguably not a model error: "tell me about students in reverse alphabetical order" does not
say which column to sort by, and the model's `ORDER BY last_name DESC` is a valid reading.
Full labels and notes: `results/e1_zeroshot_Qwen2.5-Coder-1.5B-Instruct_balanced/error_analysis.csv`.

**Reproducibility.** Every run saves its full configuration (model, prompt version, decoding,
subset, git commit, library versions, GPU) together with all raw outputs and per-question
results in `results/<run>/`.

**Compute.** All generation ran on a single NVIDIA Tesla T4 (16 GB, Google Colab) in half precision with batch size 8.
E1 took 596 s for the 200-question subset (≈3.0 s/question) and 2,083 s (≈35 min, ≈2.0 s/question) for the full dev set.
Scoring, error analysis and plots run on CPU.
