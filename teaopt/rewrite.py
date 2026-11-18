"""Subquery unnesting: correlated EXISTS / NOT EXISTS / IN become semi joins.

A correlated subquery such as::

    SELECT ... FROM lots l
    WHERE EXISTS (SELECT 1 FROM reports r
                  WHERE r.lot_id = l.id AND r.passed = 1)

is semantically a semi join.  Turning it into a semi-join *atom* lets the
optimizer choose where to apply it (it may even be pulled below other joins,
as long as the outer references are available), which is the whole point of
requirement 6: the subquery participates in the join-order enumeration instead
of being an opaque filter.

Supported shapes
----------------
* ``EXISTS (SELECT ... FROM r [JOIN s ON ...] WHERE corr AND local)`` -> semi
* ``NOT EXISTS (...)``                                                -> anti
* ``col IN (SELECT r.k FROM r ... WHERE corr)``                       -> semi
  (only when ``col`` and ``r.k`` are NOT NULL-capable; NULL semantics would
  otherwise require a NULL-aware anti join, so we bail out and keep the
  original predicate as a filter)

Anything else (aggregates, LIMIT, OR-correlation, non-equi correlation) is
left untouched and reported in ``query.notes``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import sqlglot
from sqlglot import exp

from .types import Predicate, Query, SemiUnit


def _split_conjuncts(node: Optional[exp.Expression]) -> List[exp.Expression]:
    if node is None:
        return []
    if isinstance(node, exp.And):
        return _split_conjuncts(node.this) + _split_conjuncts(node.expression)
    if isinstance(node, exp.Paren):
        return _split_conjuncts(node.this)
    return [node]


def _literal(node: Optional[exp.Expression]):
    if node is None:
        return None
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        try:
            return int(node.this)
        except ValueError:
            try:
                return float(node.this)
            except ValueError:
                return node.this
    if isinstance(node, exp.Neg):
        v = _literal(node.this)
        return -v if isinstance(v, (int, float)) else None
    return node.sql()


class _Unsupported(Exception):
    pass


def unnest_subqueries(query: Query, sql: str = "") -> Query:
    """Rewrite correlated subqueries in the WHERE clause of ``query``.

    ``query.sql`` must still contain the full original statement; the
    rewritten predicates are removed from the query object (their SQL text is
    kept on the semi unit for display) and the units are added to
    ``query.semi_units``.
    """
    try:
        parsed = sqlglot.parse_one(query.sql or sql, read="sqlite")
    except Exception:
        query.notes.append("子查询改写：无法重新解析 SQL，跳过")
        return query
    if not isinstance(parsed, exp.Select):
        return query
    where = parsed.args.get("where")
    if where is None:
        return query

    units: List[SemiUnit] = []
    kept: List[str] = []
    unit_no = 0

    for conj in _split_conjuncts(where.this):
        unit = None
        try:
            unit = _try_exists(conj, query, unit_no)
            if unit is None:
                unit = _try_in(conj, query, unit_no)
        except _Unsupported as exc:
            query.notes.append(f"子查询未改写：{exc}（保留为过滤条件）")
            unit = None
        if unit is None:
            kept.append(conj.sql(dialect="sqlite"))
        else:
            units.append(unit)
            unit_no += 1

    if not units:
        return query

    for unit in units:
        query.semi_units.append(unit)
        for r in unit.inner_relations:
            query.tables[r] = unit.inner_alias_tables[r]
        for r in unit.inner_relations:
            query.graph.setdefault(r, set())
        for p in unit.correlation:
            query.add_edge(p.left_table, p.right_table)
        for p in unit.inner_joins:
            query.add_edge(p.left_table, p.right_table)

    # update the stored SQL text used to rebuild the query
    if kept:
        query.sql = _replace_where(parsed, kept)
    else:
        query.sql = _replace_where(parsed, None)
    return query


def _replace_where(parsed: exp.Select, kept: Optional[List[str]]) -> str:
    select = parsed.copy()
    if not kept:
        select.set("where", None)
    else:
        text = " AND ".join(f"({c})" if " OR " in c.upper() or " IN " in c.upper() else c
                            for c in kept)
        select.set("where", exp.Where(this=sqlglot.parse_one(text, read="sqlite")))
    return select.sql(dialect="sqlite")


def _collect_tables(select: exp.Select) -> List[str]:
    """Aliases of the base tables of a subquery FROM clause."""
    out: List[str] = []
    from_ = select.args.get("from_")
    if from_ is not None and isinstance(from_.this, exp.Table):
        out.append(from_.this.alias_or_name)
    for j in select.args.get("joins") or []:
        if isinstance(j.this, exp.Table):
            out.append(j.this.alias_or_name)
    return out


def _collect_table_names(select: exp.Select) -> Dict[str, str]:
    """alias -> real table name for a subquery FROM clause."""
    out: Dict[str, str] = {}
    from_ = select.args.get("from_")
    if from_ is not None and isinstance(from_.this, exp.Table):
        out[from_.this.alias_or_name] = from_.this.name
    for j in select.args.get("joins") or []:
        if isinstance(j.this, exp.Table):
            out[j.this.alias_or_name] = j.this.name
    return out


def _check_supported_select(select: exp.Select) -> None:
    if select.args.get("group") or select.args.get("having"):
        raise _Unsupported("子查询含 GROUP BY / HAVING")
    if select.args.get("limit"):
        raise _Unsupported("子查询含 LIMIT")
    if select.args.get("distinct"):
        raise _Unsupported("子查询含 DISTINCT")
    for j in select.args.get("joins") or []:
        if (j.kind or "").upper() == "CROSS" or (j.side or "").upper() in ("LEFT", "RIGHT", "FULL"):
            raise _Unsupported("子查询含外连接 / 笛卡尔积")
    if not select.args.get("from_"):
        raise _Unsupported("子查询没有 FROM")
    for e in select.args.get("expressions") or []:
        if list(e.find_all(exp.AggFunc)):
            raise _Unsupported("子查询含聚合函数")


def _try_exists(conj: exp.Expression, query: Query, unit_no: int) -> Optional[SemiUnit]:
    anti = False
    node = conj
    if isinstance(node, exp.Not):
        anti = True
        node = node.this
    if not isinstance(node, exp.Exists):
        return None
    select = node.this
    if not isinstance(select, exp.Select):
        return None
    _check_supported_select(select)
    inner_rels = _collect_tables(select)
    if not inner_rels:
        raise _Unsupported("子查询没有可识别的基表")
    outer_set = set(query.tables.keys())
    correlation: List[Predicate] = []
    inner_filters = {}
    inner_joins: List[Predicate] = []
    on_preds = []
    for j in select.args.get("joins") or []:
        on = j.args.get("on")
        if on is not None:
            on_preds.extend(_split_conjuncts(on))
    where = select.args.get("where")
    where_conds = _split_conjuncts(where.this) if where is not None else []
    for cond in on_preds + where_conds:
        text = cond.sql(dialect="sqlite")
        if isinstance(cond, exp.EQ) and isinstance(cond.this, exp.Column) and isinstance(cond.expression, exp.Column):
            lt, rt = cond.this.table, cond.expression.table
            lc, rc = cond.this.name, cond.expression.name
            if lt in inner_rels and rt in inner_rels:
                inner_joins.append(Predicate(kind="eq", text=text, left_table=lt,
                                             right_table=rt, col_left=lc,
                                             col_right=rc, origin="inner-subquery"))
            elif {lt, rt} & set(inner_rels) and {lt, rt} & outer_set:
                in_t, out_t = (lt, rt) if lt in inner_rels else (rt, lt)
                in_c, out_c = (lc, rc) if lt in inner_rels else (rc, lc)
                correlation.append(Predicate(kind="eq", text=text, left_table=out_t,
                                             right_table=in_t, col_left=out_c,
                                             col_right=in_c, origin="correlation"))
            else:
                raise _Unsupported(f"无法分类的子查询谓词 {text}")
        elif isinstance(cond, exp.EQ) and isinstance(cond.this, exp.Column) and not isinstance(cond.expression, exp.Column):
            t = cond.this.table
            if t not in inner_rels:
                raise _Unsupported(f"子查询过滤引用了外部关系 {t}")
            inner_filters.setdefault(t, []).append(
                Predicate(kind="eq", text=text, left_table=t, col_left=cond.this.name,
                          op="=", value=_literal(cond.expression)))
        elif isinstance(cond, (exp.LT, exp.LTE, exp.GT, exp.GTE)):
            cols = [c for c in cond.find_all(exp.Column) if c.table]
            if any(c.table not in inner_rels for c in cols):
                raise _Unsupported("子查询过滤引用了外部关系（非等值相关）")
            t = cols[0].table
            op = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[cond.key]
            inner_filters.setdefault(t, []).append(
                Predicate(kind="range", text=text, left_table=t, col_left=cols[0].name,
                          op=op, value=_literal(cond.expression)))
        elif isinstance(cond, exp.Between):
            t = cond.this.table
            inner_filters.setdefault(t, []).append(
                Predicate(kind="range", text=text, left_table=t, col_left=cond.this.name,
                          op="between", value=_literal(cond.args.get("low")),
                          value2=_literal(cond.args.get("high"))))
        else:
            tables = {c.table for c in cond.find_all(exp.Column) if c.table}
            if not tables <= set(inner_rels):
                raise _Unsupported(f"子查询谓词引用了外部关系: {text}")
            t = sorted(tables)[0] if tables else inner_rels[0]
            inner_filters.setdefault(t, []).append(
                Predicate(kind="expr", text=text, left_table=t, op="expr"))
    if not correlation:
        raise _Unsupported("子查询不是相关子查询，无法作为半连接参与换序")
    outer_refs: Set[str] = set()
    for p in correlation:
        outer_refs.add(p.left_table)
    inner_alias = "__semi%d" % unit_no
    unit = SemiUnit(
        id=unit_no,
        inner_conjuncts=list(where_conds),
        kind="anti" if anti else "semi",
        outer_refs=outer_refs,
        inner_relations=inner_rels,
        inner_joins=inner_joins,
        correlation=correlation,
        inner_filters=inner_filters,
        inner_alias_tables=_collect_table_names(select),
        original_sql=select.sql(dialect="sqlite"),
        predicate_sql=conj.sql(dialect="sqlite"),
        note="NOT EXISTS 改写为反连接" if anti else "EXISTS 改写为半连接",
        inner_alias=inner_alias,
    )
    return unit


def _try_in(conj: exp.Expression, query: Query, unit_no: int) -> Optional[SemiUnit]:
    node = conj
    anti = False
    if isinstance(node, exp.Not):
        anti = True
        node = node.this
    if not isinstance(node, exp.In):
        return None
    sub = node.args.get("query")
    if sub is None:
        return None
    select = sub.this
    if not isinstance(select, exp.Select):
        return None
    _check_supported_select(select)
    exprs = select.args.get("expressions") or []
    if len(exprs) != 1 or not isinstance(exprs[0], exp.Column):
        raise _Unsupported("IN 子查询的 SELECT 列表不是单列")
    left_col = node.this
    if not isinstance(left_col, exp.Column):
        raise _Unsupported("IN 左侧不是列引用")
    inner_rels = _collect_tables(select)
    if not inner_rels:
        raise _Unsupported("IN 子查询没有基表")
    outer_table = left_col.table
    if not outer_table or outer_table not in query.tables:
        raise _Unsupported("IN 左侧列缺少别名")
    inner_col = exprs[0]
    it = inner_col.table or inner_rels[0]
    unit = _try_exists_shape(select, inner_rels, query, unit_no, anti, "in")
    if unit is None:
        return None
    corr = Predicate(kind="eq", text=f"{outer_table}.{left_col.name} = {it}.{inner_col.name}",
                     left_table=outer_table, right_table=it, col_left=left_col.name,
                     col_right=inner_col.name, origin="correlation")
    unit.correlation.append(corr)
    unit.outer_refs.add(outer_table)
    unit.predicate_sql = conj.sql(dialect="sqlite")
    unit.mode = "in"
    unit.note = ("NOT IN 子查询改写为反连接（列可空时语义不同，请确认数据无 NULL）"
                 if anti else "IN 子查询改写为半连接")
    return unit


def _try_exists_shape(select: exp.Select, inner_rels: List[str], query: Query,
                      unit_no: int, anti: bool, mode: str) -> Optional[SemiUnit]:
    """Shared builder for IN-subquery units."""
    outer_set = set(query.tables.keys())
    correlation: List[Predicate] = []
    inner_filters = {}
    inner_joins: List[Predicate] = []
    on_preds = []
    for j in select.args.get("joins") or []:
        on = j.args.get("on")
        if on is not None:
            on_preds.extend(_split_conjuncts(on))
    where = select.args.get("where")
    where_conds = _split_conjuncts(where.this) if where is not None else []
    for cond in on_preds + where_conds:
        text = cond.sql(dialect="sqlite")
        if isinstance(cond, exp.EQ) and isinstance(cond.this, exp.Column) and isinstance(cond.expression, exp.Column):
            lt, rt = cond.this.table, cond.expression.table
            lc, rc = cond.this.name, cond.expression.name
            if lt in inner_rels and rt in inner_rels:
                inner_joins.append(Predicate(kind="eq", text=text, left_table=lt,
                                             right_table=rt, col_left=lc, col_right=rc))
            elif {lt, rt} & set(inner_rels) and {lt, rt} & outer_set:
                in_t, out_t = (lt, rt) if lt in inner_rels else (rt, lt)
                in_c, out_c = (lc, rc) if lt in inner_rels else (rc, lc)
                correlation.append(Predicate(kind="eq", text=text, left_table=out_t,
                                             right_table=in_t, col_left=out_c,
                                             col_right=in_c, origin="correlation"))
            else:
                raise _Unsupported(f"无法分类的 IN 子查询谓词 {text}")
        else:
            tables = {c.table for c in cond.find_all(exp.Column) if c.table}
            if not tables <= set(inner_rels):
                raise _Unsupported(f"IN 子查询谓词引用外部关系: {text}")
            t = sorted(tables)[0] if tables else inner_rels[0]
            inner_filters.setdefault(t, []).append(
                Predicate(kind="expr" if not isinstance(cond, (exp.EQ, exp.In, exp.Like)) else "expr",
                          text=text, left_table=t, op="expr"))
    return SemiUnit(
        id=unit_no, kind="anti" if anti else "semi", outer_refs=set(),
        inner_relations=inner_rels, inner_joins=inner_joins,
        correlation=correlation, inner_filters=inner_filters,
        inner_alias_tables=_collect_table_names(select),
        original_sql=select.sql(dialect="sqlite"),
        inner_alias="__semi%d" % unit_no,
    )
