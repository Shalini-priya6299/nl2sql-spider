Error categories (25 incorrect predictions, e1_zeroshot_Qwen2.5-Coder-1.5B-Instruct_balanced)

| Category | Description | easy | medium | hard | extra | Total | % |
|---|---|---|---|---|---|---|---|
| schema_linking | wrong or non-existent table/column (incl. 'no such column') | 5 | 3 | 1 | 3 | 12 | 48% |
| nested_set_ops | wrong nesting or INTERSECT/EXCEPT/UNION | 0 | 0 | 3 | 1 | 4 | 16% |
| join | missing, extra or wrong JOIN / join condition | 0 | 2 | 1 | 0 | 3 | 12% |
| condition_value | wrong WHERE/HAVING operator or literal value (case, spelling, quotes) | 0 | 2 | 0 | 1 | 3 | 12% |
| group_by | missing or wrong GROUP BY / aggregation | 0 | 0 | 1 | 1 | 2 | 8% |
| gold_issue | prediction is a valid reading of the question; gold is ambiguous/wrong | 1 | 0 | 0 | 0 | 1 | 4% |
