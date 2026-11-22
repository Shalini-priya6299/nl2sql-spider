Self-correction analysis: e3_gpu_balanced (200 questions)

| | Before (first SQL) | After (final SQL) |
|---|---|---|
| EX overall | 52.0% | 56.5% |
| Execution-error rate | 30.5% | 23.0% |
| EX easy | 82.0% | 88.0% |
| EX medium | 60.0% | 64.0% |
| EX hard | 34.0% | 40.0% |
| EX extra | 32.0% | 34.0% |

- Questions retried: 75
- Execution errors fixed (first SQL crashed, final SQL runs; crash-to-working only, not semantic fixes): 15
- Wrong -> right: 9; right -> wrong: 0
- First SQL returned no rows: 14 (already correct: 5; correct -> wrong after retry: 0; wrong -> right: 0)
- Extra LLM calls: 137 (0.685 per question, total LLM cost x1.685)
- Generation time: 906.5 s total (about 368.5 s of it for retries, estimated)
- SQL execution time: 0.17 s
