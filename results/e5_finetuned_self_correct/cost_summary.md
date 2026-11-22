Self-correction analysis: e5_finetuned_self_correct (200 questions)

| | Before (first SQL) | After (final SQL) |
|---|---|---|
| EX overall | 68.5% | 71.0% |
| Execution-error rate | 11.5% | 9.0% |
| EX easy | 94.0% | 94.0% |
| EX medium | 82.0% | 84.0% |
| EX hard | 54.0% | 58.0% |
| EX extra | 44.0% | 48.0% |

- Questions retried: 36
- Execution errors fixed (first SQL crashed, final SQL runs; crash-to-working only, not semantic fixes): 5
- Wrong -> right: 5; right -> wrong: 0
- First SQL returned no rows: 13 (already correct: 5; correct -> wrong after retry: 0; wrong -> right: 0)
- Extra LLM calls: 67 (0.335 per question, total LLM cost x1.335)
- Generation time: 8617.4 s total (about 2162.4 s of it for retries, estimated)
- SQL execution time: 5.22 s
