"""Execute a plan on SQLite with the join order *enforced*.

SQLite is allowed to reorder inner joins, so simply writing the tables in the
optimizer's order would not execute the plan we picked.  The reliable way is a
chain of ``MATERIALIZED`` CTEs, one per join step::

    WITH s1 AS MATERIALIZED (SELECT o.* FROM origins o WHERE o.province = '云南'),
         s2 AS MATERIALIZED (SELECT s1.*, l.* FROM s1 JOIN lots l
                             ON l.origin_id = s1.o__origin_id),
         s3 AS MATERIALIZED (SELECT s2.* FROM s2
                             WHERE EXISTS (SELECT 1 FROM reports r
                                           WHERE r.lot_id = s2.l__lot_id))
    SELECT ... FROM s3;

Each CTE materialises exactly one step, so the execution order is the plan
order.  Outer joins are rendered with ``LEFT JOIN ... ON`` and semi joins with
``EXISTS`` / ``NOT EXISTS`` in the CTE that applies them.

The module also measures the *actual* row count of every plan node by running
``SELECT COUNT(*)`` on each CTE prefix, which feeds requirement 7.
"""

from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import sqlglot
from sqlglot import exp

from .types import PlanNode, Query, SemiUnit


class SqlBuildError(ValueError):
    pass


class QueryTimeout(RuntimeError):
    """Raised when an execution exceeds its time budget."""


def _install_timeout(conn: sqlite3.Connection, timeout_ms: int):
    """Abort a running statement after ``timeout_ms`` milliseconds.

    Returns a callable that removes the handler.  SQLite calls the progress
    handler every ~1000 VM instructions; returning non-zero aborts the query.
    """
    import time as _time
    if timeout_ms is None or timeout_ms <= 0:
        return lambda: None
    deadline = _time.monotonic() + timeout_ms / 1000.0

    def handler():
        return 1 if _time.monotonic() > deadline else 0

    conn.set_progress_handler(handler, 1000)

    def clear():
        conn.set_progress_handler(None, 0)
    return clear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]


def cookie(alias: str, col: str) -> str:
    return f"{alias}__{col}"


_COL_RE = re.compile(r'\b([A-Za-z_]\w*)\."?([A-Za-z_]\w*)"?')


def _rewrite(text: str, mapping: Dict[Tuple[str, str], str]) -> str:
    if not text:
        return text

    def repl(m):
        alias, col = m.group(1), m.group(2)
        return mapping.get((alias, col), m.group(0))
    return _COL_RE.sub(repl, text)


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    out, depth, cur, quote = [], 0, [], None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            cur.append(ch)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return out


def _split_conjuncts(node) -> List[Any]:
    if node is None:
        return []
    if isinstance(node, exp.And):
        return _split_conjuncts(node.this) + _split_conjuncts(node.expression)
    if isinstance(node, exp.Paren):
        return _split_conjuncts(node.this)
    return [node]


# ---------------------------------------------------------------------------
# Plan linearisation
# ---------------------------------------------------------------------------


@dataclass
class Step:
    kind: str                       # scan | join | outer | semi | anti
    node: PlanNode
    inner: Optional["Pipeline"] = None
    scan_alias: Optional[str] = None


@dataclass
class Pipeline:
    steps: List[Step]


def linearise(node: Optional[PlanNode]) -> Pipeline:
    if node is None:
        return Pipeline([])
    if node.node_type == "scan":
        return Pipeline([Step(kind="scan", node=node, scan_alias=node.relation)])
    if node.node_type in ("semi", "anti"):
        left = linearise(node.left)
        inner = linearise(node.right)
        return Pipeline(left.steps + [Step(kind=node.node_type, node=node,
                                           inner=inner)])
    left = linearise(node.left)
    right = linearise(node.right)
    kind = "outer" if node.node_type == "outer" else "join"
    return Pipeline(left.steps + [Step(kind=kind, node=node, inner=right)])


def pipeline_relations(p: Pipeline) -> List[str]:
    out: List[str] = []
    for st in p.steps:
        if st.kind == "scan":
            out.extend(st.node.relations())
        elif st.kind in ("semi", "anti"):
            out.extend(pipeline_relations(st.inner) if st.inner else [])
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# SQL builder
# ---------------------------------------------------------------------------


class PipelineBuilder:
    def __init__(self, conn: sqlite3.Connection, query: Query):
        self.conn = conn
        self.query = query
        self.ctes: List[str] = []
        self.outputs: Dict[str, Dict[Tuple[str, str], str]] = {}
        self.cte_relations: Dict[str, List[str]] = {}
        self._cols: Dict[str, List[str]] = {}
        #: sqlite aliases used inside each CTE (must be unique in one SELECT)
        self._cte_counter = 0

    # -- helpers --------------------------------------------------------
    def columns(self, table: str) -> List[str]:
        if table not in self._cols:
            self._cols[table] = table_columns(self.conn, table)
        return self._cols[table]

    def _base_projection(self, alias: str) -> Tuple[List[str], Dict[Tuple[str, str], str]]:
        table = self.query.tables[alias]
        cols = self.columns(table)
        proj = [f'{alias}."{c}" AS "{cookie(alias, c)}"' for c in cols]
        mapping = {(alias, c): f'"{cookie(alias, c)}"' for c in cols}
        return proj, mapping

    def _new_cte(self, sql: str, relations: Sequence[str],
                 mapping: Dict[Tuple[str, str], str]) -> str:
        name = f"s{len(self.ctes) + 1}"
        self.ctes.append(f"{name} AS MATERIALIZED ({sql})")
        self.outputs[name] = dict(mapping)
        self.cte_relations[name] = list(relations)
        return name

    def _nullable_alias(self, alias: str) -> bool:
        """True when ``alias`` is the nullable side of a surviving outer join.

        A WHERE filter on such a relation must be applied *after* the outer
        join: pushing it into the scan would silently turn the outer join into
        an inner join (the classic outer-join predicate pushdown rule).
        """
        for oj in self.query.outer_joins:
            if oj.kind == "left" and alias in oj.nullable:
                return True
        return False

    def _filters_for(self, alias: str, mapping: Dict[Tuple[str, str], str],
                     include_unit_filters: bool = False) -> List[str]:
        if self._nullable_alias(alias):
            return []
        filters = list(self.query.local_filters.get(alias, []))
        return [_rewrite(p.text, mapping) for p in filters]

    def _post_outer_filters(self, alias: str, mapping: Dict[Tuple[str, str], str]
                            ) -> List[str]:
        if not self._nullable_alias(alias):
            return []
        return [_rewrite(p.text, mapping)
                for p in self.query.local_filters.get(alias, [])]

    # -- public ---------------------------------------------------------
    def build(self, plan: PlanNode) -> str:
        pipeline = linearise(plan)
        current = self._build_steps(pipeline.steps)
        return self._final_select(current)

    # -- step rendering -------------------------------------------------
    def _build_steps(self, steps: Sequence[Step],
                     outer_mapping: Optional[Dict[Tuple[str, str], str]] = None
                     ) -> str:
        """Build a CTE chain; returns the name of the last CTE.

        ``outer_mapping`` is set when this pipeline is the inner side of a
        semi join and needs access to the outer query's columns (not used by
        SQLite CTEs; semi-join inners are rendered inline instead).
        """
        current: Optional[str] = None
        mapping: Dict[Tuple[str, str], str] = {}
        relations: List[str] = []
        for step in steps:
            if step.kind == "scan":
                alias = step.scan_alias
                proj, new_map = self._base_projection(alias)
                table = self.query.tables[alias]
                # filters: main-query filters for this alias
                filters = self._filters_for(alias, new_map)
                sql = f'SELECT {", ".join(proj)} FROM "{table}" AS {alias}'
                if filters:
                    sql += " WHERE " + " AND ".join(filters)
                current = self._new_cte(sql, [alias], new_map)
                mapping = new_map
                relations = [alias]
                continue
            if step.kind in ("semi", "anti"):
                assert current is not None
                unit = self.query.semi_units[step.node.extra["unit_id"]]
                pred = self._semi_subquery_sql(unit, mapping)
                op = "EXISTS" if step.kind == "semi" else "NOT EXISTS"
                proj = ", ".join(mapping.values())
                current = self._new_cte(
                    f'SELECT {proj} FROM {current} WHERE {op} ({pred})',
                    relations, mapping)
                continue
            # join / outer
            assert step.inner is not None
            right_rels = pipeline_relations(step.inner)
            if len(step.inner.steps) == 1 and step.inner.steps[0].kind == "scan":
                alias = step.inner.steps[0].scan_alias
                proj, right_map = self._base_projection(alias)
                table = self.query.tables[alias]
                filters = self._filters_for(alias, right_map)
                right_from = f'(SELECT {", ".join(proj)} FROM "{table}" AS {alias}'
                if filters:
                    right_from += " WHERE " + " AND ".join(filters)
                right_from += f') AS {alias}_src'
            elif step.inner.steps and step.inner.steps[0].kind == "scan" and \
                    len(step.inner.steps) > 1:
                # bushy right side: materialise it as its own CTE chain
                right_cte = self._build_steps(step.inner.steps)
                right_from = right_cte
                right_map = self.outputs[right_cte]
            else:
                right_cte = self._build_steps(step.inner.steps)
                right_from = right_cte
                right_map = self.outputs[right_cte]
            combined = dict(mapping)
            combined.update(right_map)
            combined_rels = relations + list(right_map_relations(right_map, right_rels))
            proj = list(mapping.values()) + list(right_map.values())
            join_preds = [p for p in step.node.predicates if p.is_join()]
            if step.kind == "outer":
                unit = self.query.outer_joins[step.node.extra["outer_id"] - 1]
                conds = [_rewrite(p.text, combined) for p in unit.predicates]
                on_sql = " AND ".join(conds) if conds else "1=1"
                sql = (f'SELECT {", ".join(proj)} FROM {current} '
                       f'LEFT JOIN {right_from} ON {on_sql}')
                post = []
                for alias in unit.nullable:
                    post.extend(self._post_outer_filters(alias, combined))
                if post:
                    sql += " WHERE " + " AND ".join(post)
            else:
                conds = [_rewrite(p.text, combined) for p in join_preds]
                if conds:
                    sql = (f'SELECT {", ".join(proj)} FROM {current} '
                           f'JOIN {right_from} ON {" AND ".join(conds)}')
                else:
                    sql = (f'SELECT {", ".join(proj)} FROM {current} '
                           f'CROSS JOIN {right_from}')
            current = self._new_cte(sql, combined_rels, combined)
            mapping = combined
            relations = combined_rels
        assert current is not None
        return current

    # -- semi join rendering --------------------------------------------
    def _semi_subquery_sql(self, unit: SemiUnit,
                           outer_mapping: Dict[Tuple[str, str], str]) -> str:
        """Render the original subquery, remapping only the *outer* columns.

        The subquery's own FROM keeps its original table names and aliases, so
        its local predicates can be used verbatim.  Correlated references to
        outer aliases are rewritten to the outer CTE's column names.
        """
        inner_aliases = set(unit.inner_alias_tables)
        try:
            select = sqlglot.parse_one(unit.original_sql, read="sqlite")
        except Exception as exc:
            raise SqlBuildError(f"无法解析子查询: {exc}") from exc
        from_sql = _subquery_from(select, lambda a: a)
        conditions: List[str] = []
        correlated_norm = {_norm(p.text) for p in unit.correlation}
        where = select.args.get("where")
        for conj in _split_conjuncts(where.this if where is not None else None):
            text = conj.sql(dialect="sqlite")
            if _norm(text) in correlated_norm:
                continue
            conditions.append(_rewrite_outer_only(text, inner_aliases,
                                                  outer_mapping))
        for p in unit.correlation:
            outer_expr = outer_mapping.get((p.left_table, p.col_left))
            if outer_expr is None:
                raise SqlBuildError(
                    f"无法解析关联条件 {p.text}（外表 {p.left_table}.{p.col_left}）")
            inner_expr = f'{p.right_table}."{p.col_right}"'
            conditions.append(f"{inner_expr} = {outer_expr}")
        sql = f'SELECT 1 FROM {from_sql}'
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        return sql

    # -- final select ----------------------------------------------------
    def _final_select(self, cte: str) -> str:
        mapping = self.outputs[cte]
        select_list = self._final_projection(cte, mapping)
        sql = f"WITH {', '.join(self.ctes)} SELECT {select_list} FROM {cte}"
        if self.query.group_by:
            sql += " " + self._rewrite_aliases(self.query.group_by, mapping)
        if self.query.order_by:
            sql += " " + self._rewrite_aliases(self.query.order_by, mapping)
        if self.query.limit:
            sql += " " + self.query.limit
        return sql

    def _rewrite_aliases(self, clause: str,
                         mapping: Dict[Tuple[str, str], str]) -> str:
        """Rewrite ``a.col`` references in GROUP BY / ORDER BY."""
        if not clause.upper().startswith(("GROUP BY", "ORDER BY")):
            head = "GROUP BY" if "GROUP" in clause.upper() else "ORDER BY"
            clause = head + " " + clause
        return _rewrite(clause, mapping)

    def _order_by(self, mapping: Dict[Tuple[str, str], str]) -> str:
        text = self.query.order_by or ""
        if not text.upper().startswith("ORDER BY"):
            text = "ORDER BY " + text
        return _rewrite(text, mapping)

    def _final_projection(self, cte: str,
                          mapping: Dict[Tuple[str, str], str]) -> str:
        text = (self.query.select_list or "*").strip()
        if text == "*":
            parts = []
            for alias in self.cte_relations[cte]:
                for (a, c), v in mapping.items():
                    if a == alias:
                        parts.append(f'{v} AS "{alias}__{c}"')
            return ", ".join(parts) if parts else "*"
        parts = []
        for expr in _split_top_level(text):
            parts.append(self._project_expr(expr, mapping))
        return ", ".join(parts)

    def _project_expr(self, expr: str,
                      mapping: Dict[Tuple[str, str], str]) -> str:
        expr = expr.strip()
        try:
            node = sqlglot.parse_one(expr, read="sqlite")
        except Exception:
            return _rewrite(expr, mapping)
        if isinstance(node, exp.Column):
            key = (node.table or "", node.name)
            if key in mapping:
                return f'{mapping[key]} AS "{node.name}"'
            hits = [v for (a, c), v in mapping.items() if c == node.name]
            if len(hits) == 1:
                return f'{hits[0]} AS "{node.name}"'
            return expr
        # aggregate or expression: rewrite column references in place
        rewritten = expr
        for (a, c), v in mapping.items():
            rewritten = re.sub(rf'\b{re.escape(a)}\."?{re.escape(c)}"?\b',
                               v.replace('"', ''), rewritten)
        return rewritten


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _is_single_count(sql: str) -> bool:
    """True when the outermost SELECT list is exactly one COUNT(...)."""
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return False
    if not isinstance(parsed, exp.Select):
        return False
    exprs = parsed.args.get("expressions") or []
    if len(exprs) != 1:
        return False
    expr = exprs[0]
    if isinstance(expr, exp.Alias):
        expr = expr.this
    return isinstance(expr, exp.Count)


def _rewrite_outer_only(text: str, inner_aliases: Set[str],
                        outer_map: Dict[Tuple[str, str], str]) -> str:
    """Rewrite references to outer aliases; leave subquery-local ones alone."""
    def repl(m):
        alias, col = m.group(1), m.group(2)
        if alias in inner_aliases:
            return m.group(0)
        return outer_map.get((alias, col), m.group(0))
    return _COL_RE.sub(repl, text)


def _subquery_from(select: exp.Select, map_alias) -> str:
    from_ = select.args.get("from_")
    if from_ is None:
        raise SqlBuildError("子查询缺少 FROM")
    base = from_.this
    if not isinstance(base, exp.Table):
        raise SqlBuildError("子查询 FROM 项不是基表")
    parts = [f'"{base.name}" AS {map_alias(base.alias_or_name)}']
    for j in select.args.get("joins") or []:
        if not isinstance(j.this, exp.Table):
            raise SqlBuildError("子查询 JOIN 项不是基表")
        alias = map_alias(j.this.alias_or_name)
        on = j.args.get("on")
        on_sql = ""
        if on is not None:
            on_sql = " ON " + on.sql(dialect="sqlite")
        parts.append(f'JOIN "{j.this.name}" AS {alias}{on_sql}')
    return " ".join(parts)


def right_map_relations(right_map: Dict[Tuple[str, str], str],
                        right_rels: Sequence[str]) -> List[str]:
    aliases = []
    for (a, _c) in right_map:
        if a not in aliases:
            aliases.append(a)
    return aliases


# ---------------------------------------------------------------------------
# Alternate order rendering (for comparisons)
# ---------------------------------------------------------------------------


def sql_for_order(conn: sqlite3.Connection, query: Query,
                  order: Sequence[str]) -> str:
    """Render the query with relations joined in ``order``.

    Only legal for queries without outer joins and without semi units; used to
    demonstrate the difference between a good and a bad order.
    """
    if query.outer_joins or query.semi_units:
        raise SqlBuildError("该查询含外连接 / 子查询，无法自由重排")
    parts = [f'"{query.tables[order[0]]}" AS {order[0]}']
    have = {order[0]}
    for alias in order[1:]:
        conns = [p for p in query.inner_joins
                 if (p.left_table == alias and p.right_table in have) or
                    (p.right_table == alias and p.left_table in have)]
        join_sql = ""
        if conns:
            join_sql = " ON " + " AND ".join(_quote_cols(p.text, query) for p in conns)
        parts.append(f'INNER JOIN "{query.tables[alias]}" AS {alias}{join_sql}')
        have.add(alias)
    where = [pred.text for a in order for pred in query.local_filters.get(a, [])]
    sql = "SELECT " + query.select_list + " FROM " + " ".join(parts)
    if where:
        sql += " WHERE " + " AND ".join(where)
    return sql


def _quote_cols(text: str, query: Query) -> str:
    return text


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    sql: str
    okay: bool
    error: Optional[str] = None
    #: number of result rows (``None`` when more than ``sample`` were fetched)
    rows: Optional[int] = None
    elapsed_ms: float = 0.0
    sample: Optional[List[tuple]] = None
    explain: List[str] = field(default_factory=list)
    #: value of the first column when the query is a single COUNT(...)
    count_value: Optional[int] = None


def execute(conn: sqlite3.Connection, sql: str, sample: int = 5,
            explain: bool = True, timeout_ms: int = 0) -> ExecutionResult:
    result = ExecutionResult(sql=sql, okay=False)
    clear = _install_timeout(conn, timeout_ms)
    try:
        start = time.perf_counter()
        cur = conn.execute(sql)
        fetched = cur.fetchmany(sample)
        result.elapsed_ms = (time.perf_counter() - start) * 1000.0
        result.sample = [tuple(r) for r in fetched]
        # Exact row count of the *result set*.  When the query is a single
        # COUNT(...) we must not re-run it (it may be expensive); the value is
        # in the first column, so remember it separately.
        if _is_single_count(sql) and result.sample:
            try:
                result.count_value = int(result.sample[0][0])
            except (TypeError, ValueError):
                result.count_value = None
        result.rows = len(fetched) if sample else None
        result.okay = True
        if explain:
            result.explain = [" | ".join(str(c) for c in row)
                              for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
    except Exception as exc:
        if conn.in_transaction:
            pass
        result.error = f"{type(exc).__name__}: {exc}"
        if "interrupted" in str(exc).lower():
            result.error = f"执行超时（>{timeout_ms} ms），已中止"
    finally:
        clear()
    return result


def execute_timed(conn: sqlite3.Connection, sql: str, repeats: int = 3,
                  timeout_ms: int = 3000) -> float:
    """Median wall-clock time for running the query to completion (ms).

    Returns ``inf`` if any run fails or exceeds ``timeout_ms``; the caller can
    then report "超时" rather than blocking the page.
    """
    times = []
    for _ in range(max(1, repeats)):
        clear = _install_timeout(conn, timeout_ms)
        start = time.perf_counter()
        try:
            conn.execute(sql).fetchall()
        except Exception:
            clear()
            return float("inf")
        finally:
            clear()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def measure_step_rows(conn: sqlite3.Connection, query: Query,
                      plan: PlanNode, timeout_ms: int = 2000,
                      max_rows: float = 20_000_000.0) -> Dict[int, float]:
    """Actual row count of every plan node, by counting CTE prefixes.

    Stops early when an intermediate result becomes very large (the remaining
    prefixes would only be slower) or when a step exceeds its time budget.
    """
    steps = pipeline_steps_with_nodes(plan)
    result: Dict[int, float] = {}
    for i in range(1, len(steps) + 1):
        prefix = [s for s, _ in steps[:i]]
        b = PipelineBuilder(conn, query)
        clear = _install_timeout(conn, timeout_ms)
        try:
            current = b._build_steps(prefix)
            sql = f"WITH {', '.join(b.ctes)} SELECT COUNT(*) FROM {current}"
            n = conn.execute(sql).fetchone()[0]
        except Exception:
            clear()
            break
        finally:
            clear()
        node = steps[i - 1][1]
        if node is not None and node.node_type != "scan":
            result[id(node)] = float(n)
        if n > max_rows:
            break
    return result


def pipeline_steps_with_nodes(plan: PlanNode) -> List[Tuple[Step, Optional[PlanNode]]]:
    """Linearised steps; the semi-join inner pipeline is skipped."""
    pip = linearise(plan)
    return [(st, st.node if st.kind != "scan" else None) for st in pip.steps]
