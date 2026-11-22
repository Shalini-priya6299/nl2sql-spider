Error categories (25 incorrect predictions, e4_finetuned_Qwen2.5-Coder-1.5B-Instruct_balanced)

| Category | Description | easy | medium | hard | extra | Total | % |
|---|---|---|---|---|---|---|---|
| schema_linking | wrong or non-existent table/column (incl. 'no such column') | 0 | 2 | 4 | 4 | 10 | 40% |
| join | missing, extra or wrong JOIN / join condition | 0 | 3 | 1 | 1 | 5 | 20% |
| gold_issue | prediction is a valid reading of the question; gold is ambiguous/wrong | 1 | 0 | 0 | 1 | 2 | 8% |
| select_output | wrong/extra/missing SELECT columns, DISTINCT, ORDER BY or LIMIT | 1 | 1 | 0 | 0 | 2 | 8% |
| condition_value | wrong WHERE/HAVING operator or literal value (case, spelling, quotes) | 0 | 2 | 0 | 0 | 2 | 8% |
| nested_set_ops | wrong nesting or INTERSECT/EXCEPT/UNION | 0 | 0 | 2 | 0 | 2 | 8% |
| group_by | missing or wrong GROUP BY / aggregation | 0 | 0 | 1 | 1 | 2 | 8% |
