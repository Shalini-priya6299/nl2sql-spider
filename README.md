# nl2sql-spider

A Text-to-SQL system that uses small open-source LLMs to convert natural language questions into SQL queries for unseen databases. Explores zero-shot and few-shot prompting, execution-based self-correction, and LoRA fine-tuning using the Spider benchmark.

**Research questions:** How does execution accuracy (EX) change with query difficulty? How much does
execute-and-retry self-correction help compared with LoRA fine-tuning?

## Experiments

| ID | Experiment | Owner (issue) | Script |
|---|---|---|---|
| E1 | Zero-shot, schema prompt (CREATE TABLE + 3 sample rows) | Part 1 (#1) | `run_e1_zeroshot.py` |
| E2 | Few-shot prompting | Part 2 (#2) | _TBD_ |
| E3, E5 | Execution feedback and self-correction | Part 3 (#3) | `run_e3_self_correct.py` (E5: add `--adapter`) |
| E4 | LoRA fine-tuning of the code LLM | Part 4 (#4) | _TBD_ |

## Setup

```bash
git clone https://github.com/Shalini-priya6299/nl2sql-spider && cd nl2sql-spider
pip install -r requirements.txt
python data_utils.py download            # full Spider (~1.8 GB unzipped, needs train DBs for LoRA)
python data_utils.py download --only-dev # low-disk option: JSON + 20 dev databases (~150 MB)
python data_utils.py stats               # sanity check: dev = 248/446/174/166 by difficulty
python -m pytest tests/                  # evaluator tests
```

## Shared code (use these in every part)

| File | What it gives you |
|---|---|
| `data_utils.py` | `load_split("train"/"dev")` (each example has `idx`, `hardness`), `get_schema_prompt(db_id)`, `db_path(db_id)`, `check_no_leakage()` |
| `hardness.py` | `eval_hardness(sql)`: official Spider difficulty, `HARDNESS_LEVELS` (no project imports, so it cannot cause a cycle) |
| `evaluate.py` | `execute_sql(db, sql)` → `(rows, error)`, `exec_match(db, pred, gold)`, `load_dev_subset("balanced"/"full")`, `clean_sql(raw_output)`, `score_file(predictions.jsonl)` |
| `prompts.py` | prompt templates with version strings (add yours here, don't edit `zeroshot-v1`) |
| `error_analysis.py` | sample incorrect predictions to a CSV for annotation, then count categories |
| `make_results.py` | final results table (`results/summary.md`) and plots (`figures/`) |

### Evaluation protocol (everyone must follow this so numbers are comparable)

- **Metric:** execution accuracy (EX). Predicted and gold SQL are executed on the Spider database and
  their results compared with the official test-suite rules (row multiset, list if gold has `ORDER BY`,
  column order ignored). Gold-vs-gold scores 100% on all 1,034 dev questions.
- **Difficulty:** official Spider hardness (easy / medium / hard / extra), computed from `dev.json`.
- **Dev subset:** `splits/dev_subset_balanced.json`, a fixed 200 questions with 50 per difficulty (seed 42). Use it for all
  comparisons. Final headline numbers can also be reported on the full dev set (`--subset full`).
- **No leakage:** few-shot examples and LoRA training data come **only** from the training split.
  Spider's dev databases never appear in train (`python data_utils.py stats` checks this).
- **Output format:** every run writes `results/<run_name>/` with `config.json`, `predictions.jsonl`
  (`idx`, `pred_sql`, `raw_output`, ...), `metrics.json`, `eval_details.jsonl`. Score any
  predictions file with `python evaluate.py score results/<run>/predictions.jsonl`.
  Only `config.json` and `metrics.json` are committed: `predictions.jsonl` and `eval_details.jsonl`
  contain Spider questions and gold SQL (CC BY-SA 4.0) and are git-ignored, so share them via the
  team Drive or a release asset. Run names starting with `_` are scratch runs and are not committed.

## E1: zero-shot baseline

Needs a GPU, so run it on Google Colab (the free T4 GPU is enough for the default 1.5B model):
open `notebooks/run_e1_colab.ipynb` in Colab (Runtime → Change runtime type → T4 GPU) and run the cells in order.
These are the main commands the notebook runs:

```bash
python run_e1_zeroshot.py                     # balanced subset, Qwen2.5-Coder-1.5B-Instruct, greedy
python run_e1_zeroshot.py --subset full       # all 1,034 dev questions
python error_analysis.py sample results/e1_zeroshot_Qwen2.5-Coder-1.5B-Instruct_balanced/eval_details.jsonl -n 25
```

## E3 / E5: execution feedback and self-correction

`executor.py` runs SQL read-only with a timeout; `self_correct.py` retries up to 2 times with the
SQLite error as feedback. Needs a GPU, so run it on Colab:

```bash
python run_e3_self_correct.py --strategy similar --k 3          # E3: few-shot + correction
python run_e3_self_correct.py --adapter <path>/ft_1p5b/adapter   # E5: fine-tuned (Part 4) + correction
python analyze_self_correct.py results/<run>/ --save             # fixed queries + extra cost
python -m pytest tests/test_self_correct.py                      # no GPU needed
```

## Final results

```bash
python make_results.py --labels e1_zeroshot_Qwen2.5-Coder-1.5B-Instruct_balanced="E1 zero-shot"
```

Writes `results/summary.md`, `results/summary.csv`, `figures/ex_by_difficulty.png` and `figures/exec_error_rate.png`.

## Repository layout

```
data_utils.py  evaluate.py  prompts.py        shared modules
hardness.py                                   shared (Spider difficulty)
run_e1_zeroshot.py                            E1
error_analysis.py  make_results.py            analysis
splits/dev_subset_balanced.json               fixed evaluation subset (committed)
results/<run>/                                config.json + metrics.json committed; predictions git-ignored
figures/                                      plots for the report
docs/report/                                  report section drafts
tests/                                        evaluator tests
data/                                         Spider (git-ignored)
```

## References

1. Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL*, EMNLP 2018. https://arxiv.org/abs/1809.08887
2. Pourreza & Rafiei, *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction*, NeurIPS 2023. https://arxiv.org/abs/2304.11015
3. Zhong, Yu & Klein, *Semantic Evaluation for Text-to-SQL with Distilled Test Suites*, EMNLP 2020. https://arxiv.org/abs/2010.02840
4. Rajkumar, Li & Bahdanau, *Evaluating the Text-to-SQL Capabilities of Large Language Models*, 2022. https://arxiv.org/abs/2204.00498
