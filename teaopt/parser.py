"""SQL parsing and analysis.

Turns a SQL query into a :class:`teaopt.types.Query`:

* flattens comma joins and inner joins into an undirected join graph,
* keeps outer joins as constrained edges (LEFT/FULL; RIGHT is normalised to
  LEFT by swapping the operands),
* derives *stages*: blocks of relations that may be reordered among
  themselves.  Stages are evaluated in order, which is exactly the legality
  restriction outer joins impose (requirement 5),
* keeps every predicate's SQL text so the resulting plan can be re-executed
  against SQLite for validation,
* after parsing, :func:`teaopt.rewrite.unnest_subqueries` turns correlated
  ``EXISTS`` / ``NOT EXISTS`` / ``IN`` subqueries into semi / anti join units
  that participate in the join-order enumeration (requirement 6).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import sqlglot
from sqlglot import exp

from .types import OuterJoin, Predicate, Query, Stage


class ParseError(ValueError):
    pass


def _split_conjuncts(node: Optional[exp.Expression]) -> List[exp.Expression]:
    if node is None:
        return []
    if isinstance(node, exp.And):
        return _split_conjuncts(node.this) + _split_conjuncts(node.expression)
    if isinstance(node, exp.Paren):
        return _split_conjuncts(node.this)
    return [node]


def _references_filter(p) -> bool:
    """A WHERE predicate on the nullable side kills NULL-extended rows.

    ``IS NULL`` is the exception: it *selects* those NULL-extended rows, so
    the outer join must be preserved.
    """
    return p.kind != "null"


def _literal_value(node: Optional[exp.Expression]):
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
        v = _literal_value(node.this)
        return -v if isinstance(v, (int, float)) else None
    if isinstance(node, exp.Null):
        return None
    if isinstance(node, exp.Boolean):
        return bool(node.this)
    try:
        return float(node.sql())
    except (TypeError, ValueError):
        return node.sql()


class QueryAnalyzer:
    """Parse + analyze one SQL statement."""

    def __init__(self, sql: str, dialect: str = "sqlite"):
        self.sql = sql
        self.dialect = dialect
        self.query = Query(sql=sql, original_sql=sql)

    # -- public -----------------------------------------------------------
    def analyze(self) -> Query:
        try:
            parsed = sqlglot.parse_one(self.sql, read=self.dialect)
        except Exception as exc:
            raise ParseError(f"无法解析 SQL: {exc}") from exc
        if not isinstance(parsed, exp.Select):
            raise ParseError("只支持单条 SELECT 查询")
        self._analyze_select(parsed)
        self._build_stages()
        self._validate()
        return self.query

    # -- FROM / JOIN -------------------------------------------------------
    def _analyze_select(self, select: exp.Select) -> None:
        q = self.query
        if select.args.get("group") or select.args.get("having"):
            q.notes.append("检测到 GROUP BY / HAVING，只优化 FROM-JOIN 部分")
        if select.args.get("distinct"):
            q.notes.append("DISTINCT 在连接之后去重，不影响连接顺序")
        exprs = select.args.get("expressions") or []
        q.select_list = ", ".join(e.sql(dialect=self.dialect) for e in exprs) or "*"
        if select.args.get("group"):
            q.group_by = select.args["group"].sql(dialect=self.dialect)
        if select.args.get("order"):
            q.order_by = select.args["order"].sql(dialect=self.dialect)
        if select.args.get("limit"):
            q.limit = select.args["limit"].sql(dialect=self.dialect)

        from_ = select.args.get("from_")
        if from_ is None:
            raise ParseError("查询缺少 FROM 子句")
        self._add_relation(from_.this)

        for join in select.args.get("joins") or []:
            kind = (join.kind or "").upper()
            side = (join.side or "").upper()
            alias = self._add_relation(join.this)
            preds = self._predicates_from(join.args.get("on"), default_table=alias)
            external = side in ("LEFT", "RIGHT", "FULL")
            for p in preds:
                p.origin = "on-outer" if external else "on-inner"
            if external:
                self._register_outer(side.lower(), alias, preds)
            else:
                joins = [p for p in preds if p.is_join()]
                locals_ = [p for p in preds if not p.is_join()]
                for p in joins:
                    self._check_relation(p, "连接谓词")
                    q.inner_joins.append(p)
                # ``JOIN t ON t.k = x.k AND t.flag = 1``: the non-join part is a
                # filter on the joined relation and belongs to its scan
                for p in locals_:
                    table = p.left_table or alias
                    if table in q.tables:
                        q.local_filters.setdefault(table, []).append(p)
                    else:
                        raise ParseError(f"无法确定 ON 条件下谓词所属的表: {p.text}")
                if kind == "CROSS" and not preds:
                    q.notes.append(f"{alias} 是笛卡尔积连接（没有连接谓词），代价很高")

        where = select.args.get("where")
        if where is not None:
            for p in self._predicates_from(where.this, default_table=None):
                if p.is_join():
                    self._check_relation(p, "WHERE 中的连接谓词")
                    q.inner_joins.append(p)
                elif p.left_table in q.tables:
                    q.local_filters.setdefault(p.left_table, []).append(p)
                elif p.kind == "expr" and p.left_table is None:
                    if len(q.tables) == 1:
                        q.local_filters.setdefault(next(iter(q.tables)), []).append(p)
                    else:
                        raise ParseError(
                            f"无法确定谓词所属的表: {p.text}"
                            f"（多表查询请给列加表别名）")
                else:
                    raise ParseError(
                        f"无法确定谓词所属的表: {p.text}（多表查询请给列加表别名）")

        q.from_order = list(q.tables.keys())

    def _check_relation(self, p: Predicate, what: str) -> None:
        for t in (p.left_table, p.right_table):
            if t not in self.query.tables:
                raise ParseError(f"{what}引用了未知表别名 {t}: {p.text}")

    def _add_relation(self, node: exp.Expression) -> str:
        """Register a FROM item; single-table derived tables are folded."""
        q = self.query
        if isinstance(node, exp.Table):
            alias = node.alias_or_name
            if not alias:
                raise ParseError("每张表都需要别名")
            if alias in q.tables:
                raise ParseError(f"表别名 {alias} 重复")
            q.tables[alias] = node.name
            return alias
        if isinstance(node, exp.Subquery):
            inner = node.this
            if (isinstance(inner, exp.Select) and not inner.args.get("joins")
                    and inner.args.get("from_") is not None
                    and isinstance(inner.args["from_"].this, exp.Table)):
                alias = node.alias_or_name or f"subq_{len(q.tables) + 1}"
                base = inner.args["from_"].this
                q.tables[alias] = base.name
                q.folded[alias] = base.name
                q.notes.append(
                    f"派生表 {alias} 折叠为基表 {base.name}，内部过滤作为扫描谓词下推")
                w = inner.args.get("where")
                for p in self._predicates_from(w.this if w is not None else None,
                                               default_table=alias):
                    if p.is_join():
                        raise ParseError(f"派生表 {alias} 内含连接，请展开为平铺查询")
                    q.local_filters.setdefault(alias, []).append(p)
                return alias
            raise ParseError("不支持含连接 / 聚合的派生表，请先展开为平铺查询")
        raise ParseError(f"不支持的 FROM 项: {type(node).__name__}")

    def _register_outer(self, side: str, alias: str, preds: List[Predicate]) -> None:
        q = self.query
        if not preds:
            raise ParseError(f"外连接 {alias} 缺少 ON 条件")
        prefix = [r for r in q.tables.keys() if r != alias]
        if not prefix:
            raise ParseError(f"外连接 {alias} 没有左侧关系")
        for p in preds:
            if p.is_join():
                self._check_relation(p, "外连接谓词")
        if side == "left":
            oj = OuterJoin(id=len(q.outer_joins) + 1, kind="left",
                           preserved=prefix, nullable=[alias], predicates=preds)
            q.outer_joins.append(oj)
        elif side == "right":
            if len(prefix) != 1:
                raise ParseError(
                    "RIGHT JOIN 的左侧包含多张表，暂不支持；请改写为等价的 LEFT JOIN")
            # A RIGHT JOIN B  ==  B LEFT JOIN A.  Reorder the relations so the
            # preserved side is scanned first; the FROM order changes.
            preserved = [alias]
            nullable = prefix
            q.tables = {alias: q.tables[alias], prefix[0]: q.tables[prefix[0]]}
            oj = OuterJoin(id=len(q.outer_joins) + 1, kind="left",
                           preserved=preserved, nullable=nullable, predicates=preds)
            oj.note = f"RIGHT JOIN 归一化：{alias} LEFT JOIN {prefix[0]}"
            q.outer_joins.append(oj)
            q.notes.append(f"RIGHT JOIN 归一化为 LEFT JOIN（保留侧 {alias} 先求值）")
        else:  # full
            oj = OuterJoin(id=len(q.outer_joins) + 1, kind="full",
                           preserved=prefix, nullable=[alias], predicates=preds)
            q.outer_joins.append(oj)

    # -- predicates --------------------------------------------------------
    def _predicates_from(self, node: Optional[exp.Expression],
                         default_table: Optional[str]) -> List[Predicate]:
        out: List[Predicate] = []
        for conj in _split_conjuncts(node):
            pred = self._convert_conjunct(conj, default_table)
            if pred is not None:
                out.append(pred)
        return out

    def _convert_conjunct(self, node: exp.Expression,
                          default_table: Optional[str]) -> Optional[Predicate]:
        text = node.sql(dialect=self.dialect)
        if isinstance(node, exp.EQ):
            return self._make_eq(node.this, node.expression, text)
        if isinstance(node, (exp.LT, exp.LTE, exp.GT, exp.GTE)):
            return self._make_range(node, text)
        if isinstance(node, exp.Between):
            return self._make_between(node, text)
        if isinstance(node, exp.In) and node.args.get("query") is None and node.args.get("expressions"):
            return self._make_in(node, text)
        if isinstance(node, exp.Like):
            return self._make_like(node, text)
        if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null) \
                and isinstance(node.this, exp.Column):
            t = node.this.table or default_table
            return Predicate(kind="null", text=text, left_table=t,
                             col_left=node.this.name, op="IS NULL")
        if isinstance(node, exp.Not) and isinstance(node.this, exp.Is):
            inner = node.this
            if isinstance(inner.expression, exp.Null) and isinstance(inner.this, exp.Column):
                t = inner.this.table or default_table
                return Predicate(kind="notnull", text=text, left_table=t,
                                 col_left=inner.this.name, op="IS NOT NULL")
        if isinstance(node, exp.Exists):
            return None
        if isinstance(node, exp.Not) and isinstance(node.this, exp.Exists):
            return None
        if isinstance(node, exp.In):
            return None
        if isinstance(node, exp.Boolean):
            return None
        tables = {c.table for c in node.find_all(exp.Column) if c.table}
        if len(tables) == 1:
            return Predicate(kind="expr", text=text, left_table=tables.pop(), op="expr")
        if len(tables) == 0 and default_table is not None:
            return Predicate(kind="expr", text=text, left_table=default_table, op="expr")
        return Predicate(kind="expr", text=text, op="expr")

    def _make_eq(self, left: exp.Expression, right: exp.Expression, text: str) -> Predicate:
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            lt, rt = left.table or None, right.table or None
            if not lt or not rt:
                raise ParseError(
                    f"连接谓词缺少表别名: {text}（多表查询中列引用必须带别名）")
            return Predicate(kind="eq", text=text, left_table=lt, right_table=rt,
                             col_left=left.name, col_right=right.name, op="=")
        if isinstance(left, exp.Column):
            return self._make_const(left, right, text)
        if isinstance(right, exp.Column):
            return self._make_const(right, left, text)
        return Predicate(kind="expr", text=text, op="=")

    def _make_const(self, col: exp.Column, val: exp.Expression, text: str) -> Predicate:
        return Predicate(kind="eq", text=text, left_table=col.table or None,
                         col_left=col.name, op="=", value=_literal_value(val))

    def _make_range(self, node: exp.Expression, text: str) -> Predicate:
        op = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[node.key]
        if isinstance(node.this, exp.Column):
            return Predicate(kind="range", text=text, left_table=node.this.table or None,
                             col_left=node.this.name, op=op,
                             value=_literal_value(node.expression))
        if isinstance(node.expression, exp.Column):
            inv = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]
            return Predicate(kind="range", text=text, left_table=node.expression.table or None,
                             col_left=node.expression.name, op=inv,
                             value=_literal_value(node.this))
        return Predicate(kind="expr", text=text, op=op)

    def _make_between(self, node: exp.Between, text: str) -> Predicate:
        if isinstance(node.this, exp.Column):
            return Predicate(kind="range", text=text, left_table=node.this.table or None,
                             col_left=node.this.name, op="between",
                             value=_literal_value(node.args.get("low")),
                             value2=_literal_value(node.args.get("high")))
        return Predicate(kind="expr", text=text, op="between")

    def _make_in(self, node: exp.In, text: str) -> Predicate:
        if isinstance(node.this, exp.Column):
            vals = [_literal_value(e) for e in node.args["expressions"]]
            return Predicate(kind="in", text=text, left_table=node.this.table or None,
                             col_left=node.this.name, op="in", values=vals)
        return Predicate(kind="expr", text=text, op="in")

    def _make_like(self, node: exp.Like, text: str) -> Predicate:
        if isinstance(node.this, exp.Column):
            return Predicate(kind="like", text=text, left_table=node.this.table or None,
                             col_left=node.this.name, op="like",
                             value=_literal_value(node.expression))
        return Predicate(kind="expr", text=text, op="like")

    # -- stages ------------------------------------------------------------
    def _build_stages(self) -> None:
        """Split relations into reorderable stages according to outer joins.

        Legal-reordering rules (requirement 5):

        1. Relations joined by inner joins before the first outer join form the
           first stage, freely reorderable.
        2. A LEFT / FULL join's nullable relation forms its own single-relation
           stage.  It can never move before the preserved side.
        3. A RIGHT join is normalised to a LEFT join with operands swapped at
           parse time.
        4. Relations that appear after an outer join form a new inner stage and
           are evaluated after it.  This is conservative, hence always legal:
           an inner join above an outer join cannot be pulled below it when it
           references the nullable side, and moving it is only safe under
           additional checks we deliberately do not take here.
        5. Semi / anti units are *not* stages: they float inside their stage
           and are attached as soon as their outer references are available
           (requirement 6).

        Predicates that span two stages stay in ``inner_joins`` so they can be
        used as the join condition when the later stage is attached, but they
        are never used to reorder relations inside a single stage.
        """
        q = self.query
        aliases = list(q.tables.keys())
        if not q.outer_joins:
            q.stages = [Stage(relations=aliases)]
            return
        # A LEFT JOIN whose nullable side is filtered in WHERE degenerates to
        # an inner join: the NULL-extended rows can never satisfy the filter.
        for oj in q.outer_joins:
            if oj.kind != "left":
                continue
            filtered = [r for r in oj.nullable
                        if any(_references_filter(p) for p in q.local_filters.get(r, []))]
            if filtered:
                oj.degenerate = True
                for p in oj.predicates:
                    if p.is_join():
                        p.origin = "on-inner"
                        q.inner_joins.append(p)
                    elif p.left_table in q.tables:
                        # non-join ON predicates become ordinary filters
                        q.local_filters.setdefault(p.left_table, []).append(p)
                oj.predicates = []
                q.notes.append(
                    f"LEFT JOIN {oj.nullable} 的 NULL 补充行被 WHERE 过滤，"
                    f"已退化为内连接（允许参与换序）")
        q.outer_joins = [oj for oj in q.outer_joins if not oj.degenerate]
        if not q.outer_joins:
            q.stages = [Stage(relations=aliases)]
            q.notes.append("全部外连接均已退化为内连接，连接顺序可自由枚举")
            return
        boundary: Dict[int, OuterJoin] = {}
        for oj in q.outer_joins:
            pos = min(aliases.index(r) for r in oj.nullable)
            boundary[pos] = oj
        stages: List[Stage] = []
        current: List[str] = []
        for i, alias in enumerate(aliases):
            if i in boundary:
                if current:
                    stages.append(Stage(relations=current))
                current = [alias]
                stages.append(Stage(relations=current, outer=boundary[i]))
                current = []
                continue
            current.append(alias)
        if current:
            stages.append(Stage(relations=current))
        # sanity: every relation appears in exactly one stage
        seen: List[str] = [r for st in stages for r in st.relations]
        assert sorted(seen) == sorted(aliases), (seen, aliases)
        q.stages = stages
        # report cross-boundary predicates (they constrain the order)
        stage_of = {r: i for i, st in enumerate(stages) for r in st.relations}
        for p in q.inner_joins:
            if stage_of.get(p.left_table) != stage_of.get(p.right_table):
                q.notes.append(
                    f"谓词 {p.text} 跨外连接边界，在附加后一阶段时使用")

    def _validate(self) -> None:
        q = self.query
        aliases = set(q.tables)
        in_stage: Set[str] = set()
        for st in q.stages:
            in_stage.update(st.relations)
        assert in_stage == aliases, (in_stage, aliases)
        for p in q.inner_joins:
            assert p.left_table in aliases and p.right_table in aliases


def analyze(sql: str, dialect: str = "sqlite") -> Query:
    """Parse SQL into a :class:`Query` and apply subquery unnesting."""
    from .rewrite import unnest_subqueries
    query = QueryAnalyzer(sql, dialect=dialect).analyze()
    unnest_subqueries(query)
    return query
