"""Official Spider difficulty ("hardness") classification.

Logic copied from taoyds/spider evaluation.py and computed from the parsed
``sql`` field shipped in dev.json / train_spider.json.

This module has no project imports, so both ``data_utils`` and ``evaluate``
can import it at module level without creating an import cycle.
"""

from __future__ import annotations

HARDNESS_LEVELS = ["easy", "medium", "hard", "extra"]

WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


def _has_agg(unit) -> bool:
    return unit[0] != AGG_OPS.index("none")


def _count_agg(units) -> int:
    return len([u for u in units if _has_agg(u)])


def _get_nested_sql(sql: dict) -> list:
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    for key in ("intersect", "except", "union"):
        if sql[key] is not None:
            nested.append(sql[key])
    return nested


def _count_component1(sql: dict) -> int:
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1
    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([t for t in ao if t == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len([c for c in cond_units if c[1] == WHERE_OPS.index("like")])
    return count


def _count_component2(sql: dict) -> int:
    return len(_get_nested_sql(sql))


def _count_others(sql: dict) -> int:
    count = 0
    agg_count = _count_agg(sql["select"][1])
    agg_count += _count_agg(sql["where"][::2])
    agg_count += _count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += _count_agg([u[1] for u in sql["orderBy"][1] if u[1]]
                                + [u[2] for u in sql["orderBy"][1] if u[2]])
    agg_count += _count_agg(sql["having"])
    if agg_count > 1:
        count += 1
    if len(sql["select"][1]) > 1:
        count += 1
    if len(sql["where"]) > 1:
        count += 1
    if len(sql["groupBy"]) > 1:
        count += 1
    return count


def eval_hardness(sql: dict) -> str:
    """Official Spider difficulty: 'easy' | 'medium' | 'hard' | 'extra'."""
    c1, c2, co = _count_component1(sql), _count_component2(sql), _count_others(sql)
    if c1 <= 1 and co == 0 and c2 == 0:
        return "easy"
    if (co <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and co < 2 and c2 == 0):
        return "medium"
    if ((co > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and co <= 2 and c2 == 0)
            or (c1 <= 1 and co == 0 and c2 <= 1)):
        return "hard"
    return "extra"
