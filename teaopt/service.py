"""End-to-end service used by the web UI and the tests.

Ties together parsing, statistics, optimisation, SQLite execution and the
reporting data structures.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence

from . import explain as explain_mod
from .demo import create_database
from .estimate import Estimator
from .parser import ParseError, analyze
from .search import Optimizer, brute_force
from .stats import collect
from .types import OptimizerConfig, OrderResult, Query, TableStats
from .validate import timed_plan, validate


class ServiceError(ValueError):
    pass


class QueryService:
    """A session around one SQLite database file."""

    def __init__(self, db_path: str, multicolumn: Optional[Sequence] = None,
                 tables: Optional[Sequence[str]] = None):
        if not os.path.exists(db_path):
            raise ServiceError(f"数据库不存在: {db_path}")
        self.db_path = db_path
        self.multicolumn = list(multicolumn or [])
        self.tables = list(tables) if tables else self._list_tables()
        self._stats: Optional[Dict[str, TableStats]] = None
        self._indexes: Optional[List[str]] = None

    # -- connection ------------------------------------------------------
    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA cache_size = -64000")
        return conn

    def _list_tables(self) -> List[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        return [r[0] for r in rows]

    # -- stats -----------------------------------------------------------
    def stats(self) -> Dict[str, TableStats]:
        if self._stats is None:
            with self.connect() as conn:
                self._stats = collect(conn, self.tables, self.multicolumn)
        return self._stats

    def refresh_stats(self) -> Dict[str, TableStats]:
        self._stats = None
        return self.stats()

    def schema(self) -> List[Dict[str, Any]]:
        st = self.stats()
        out = []
        for t in self.tables:
            ts = st.get(t)
            if ts is None:
                continue
            out.append({
                "name": t,
                "rows": ts.row_count,
                "pages": ts.page_count,
                "columns": [
                    {"name": c.name, "type": self._column_type(c),
                     "ndv": c.ndv, "nulls": c.null_count,
                     "histogram": c.histogram is not None,
                     "mcv": [{"value": v, "count": n} for v, n in c.mcv[:5]]}
                    for c in ts.columns.values()
                ],
                "indexes": ["(" + ", ".join(cols) + ")" for cols in ts.indexes],
                "multicol": [
                    {"columns": list(mc.columns), "ndv": mc.ndv,
                     "mcv": [{"values": list(v), "count": n} for v, n in mc.mcv[:5]]}
                    for mc in ts.multicol.values()
                ],
            })
        return out

    @staticmethod
    def _column_type(cs) -> str:
        if cs.is_numeric:
            return "numeric"
        return "text"

    # -- optimize --------------------------------------------------------
    def optimize(self, sql: str, config: Optional[OptimizerConfig] = None,
                 run_on_sqlite: bool = True) -> Dict[str, Any]:
        config = config or OptimizerConfig()
        started = time.perf_counter()
        try:
            query = analyze(sql)
        except ParseError as exc:
            raise ServiceError(str(exc)) from exc
        stats = self.stats()
        optimizer = Optimizer(query, stats, config)
        result = optimizer.optimize()
        parse_ms = (time.perf_counter() - started) * 1000.0

        payload: Dict[str, Any] = {
            "sql": sql,
            "query": self._query_summary(query, optimizer),
            "plan": explain_mod.plan_to_dict(result.plan),
            "cost": result.total_cost,
            "est_rows": result.est_rows,
            "algorithm": result.algorithm,
            "order": result.order,
            "subproblems": result.subproblems,
            "transitions": result.transitions,
            "alternatives": [
                {"order": a.order, "cost": a.cost, "rows": a.rows,
                 "algorithm": a.algorithm_summary}
                for a in result.alternatives
            ],
            "notes": list(dict.fromkeys(query.notes + result.notes)),
            "stages": [s.describe() for s in query.stages],
            "predicates": self._predicate_table(query, optimizer.est),
            "parse_ms": parse_ms,
            "join_graph": self._join_graph(query, optimizer),
        }
        if result.greedy_cost is not None:
            payload["greedy"] = {
                "cost": result.greedy_cost,
                "order": result.greedy_order,
                "plan": explain_mod.plan_to_dict(result.greedy_plan),
            }
        if result.stats.get("exact_cost") is not None:
            payload["exact_optimum"] = {
                "cost": result.stats["exact_cost"],
                "order": result.stats.get("exact_order"),
                "gap": result.stats.get("greedy_gap"),
            }
        payload["exhaustive_available"] = (
            result.algorithm.startswith("exhaustive"))
        payload["threshold"] = config.exhaustive_threshold
        payload["atom_count"] = len(optimizer.atoms)

        if run_on_sqlite:
            with self.connect() as conn:
                validation = validate(conn, query, result)
                payload["validation"] = validation
                payload["plan_sql"] = validation.get("sql", "")
                payload["timing"] = self._timing(conn, query, result, config)
            payload["brute_force"] = self._brute_force_summary(
                sql, query, config, payload)
        return payload

    def _brute_force_summary(self, sql: str, query: Query,
                             config: OptimizerConfig,
                             payload: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the DP result with an exhaustive permutation search when
        the query is small enough; the UI shows this as a correctness badge."""
        from .search import brute_force
        atoms = payload.get("atom_count", 0)
        if query.outer_joins or query.semi_units:
            return {"skipped": True, "reason": "含外连接 / 子查询，仅对单阶段内连接做穷举对比"}
        if atoms > 8:
            return {"skipped": True,
                    "reason": f"原子数 {atoms} 超过 8，穷举 {_factorial(atoms)} 种排列代价过高"}
        try:
            cost, order, n = brute_force(query, self.stats(), config)
        except Exception as exc:  # pragma: no cover
            return {"skipped": True, "reason": f"{type(exc).__name__}: {exc}"}
        return {
            "skipped": False,
            "perms": n,
            "cost": cost,
            "order": order,
            "dp_cost": payload["cost"],
            "dp_order": payload["order"],
            "match": abs(cost - payload["cost"]) < 1e-6,
        }

    # -- helpers ---------------------------------------------------------
    def _query_summary(self, query: Query, optimizer: Optimizer) -> Dict[str, Any]:
        edges = query.edge_predicates()
        shape = explain_mod.shape_of(edges, [a for a in query.tables
                                            if a not in optimizer.inner_aliases])
        return {
            "tables": [{"alias": a, "table": t} for a, t in query.tables.items()],
            "inner_joins": [p.text for p in query.inner_joins],
            "outer_joins": [
                {"kind": oj.kind, "preserved": oj.preserved,
                 "nullable": oj.nullable,
                 "predicates": [p.text for p in oj.predicates]}
                for oj in query.outer_joins
            ],
            "semi_units": [
                {"kind": u.kind, "inner": u.inner_relations,
                 "outer_refs": sorted(u.outer_refs),
                 "correlation": [p.text for p in u.correlation],
                 "predicate": u.predicate_sql,
                 "note": u.note}
                for u in query.semi_units
            ],
            "stages": [s.describe() for s in query.stages],
            "graph": {
                "nodes": [{"id": a, "table": t} for a, t in query.tables.items()],
                "edges": [
                    {"source": a, "target": b,
                     "kind": self._edge_kind(query, a, b)}
                    for a, b in edges
                ],
            },
            "shape": shape,
            "atom_count": len(optimizer.atoms),
            "rewritten_sql": query.sql,
        }

    @staticmethod
    def _edge_kind(query: Query, a: str, b: str) -> str:
        for oj in query.outer_joins:
            for p in oj.predicates:
                if p.is_join() and {p.left_table, p.right_table} == {a, b}:
                    return "outer"
        for unit in query.semi_units:
            if a in unit.inner_relations or b in unit.inner_relations:
                inner = set(unit.inner_relations)
                if (a in inner) != (b in inner):
                    return unit.kind
        return "inner"

    def _join_graph(self, query: Query, optimizer: Optimizer) -> Dict[str, Any]:
        """Nodes / edges with cardinalities and selectivities for the UI."""
        est = optimizer.est
        nodes = []
        for alias, table in query.tables.items():
            if alias in optimizer.inner_aliases:
                continue
            atom = est.atoms.get(alias)
            nodes.append({
                "id": alias, "table": table,
                "rows": atom.rows if atom else None,
                "base_rows": atom.base_rows if atom else None,
                "filters": [{"text": p.text, "selectivity": p.selectivity,
                             "source": p.source, "note": p.note}
                            for p in (atom.filters if atom else [])],
            })
        edges = []
        for key, preds in est.edge_preds.items():
            if len(key) != 2:
                continue
            a, b = sorted(key)
            if a in optimizer.inner_aliases or b in optimizer.inner_aliases:
                # semi join edge: show it with a dashed style
                unit = next((u for u in query.semi_units
                             if a in u.inner_relations or b in u.inner_relations), None)
                edges.append({
                    "source": a, "target": b, "kind": unit.kind if unit else "inner",
                    "selectivity": est.edge_sel.get(key),
                    "source_of_estimate": preds[0].source if preds else None,
                    "predicates": [p.text for p in preds],
                })
                continue
            edges.append({
                "source": a, "target": b, "kind": "inner",
                "selectivity": est.edge_sel.get(key),
                "source_of_estimate": preds[0].source if preds else None,
                "predicates": [p.text for p in preds],
            })
        for oj in query.outer_joins:
            for a in oj.preserved:
                for b in oj.nullable:
                    key = frozenset((a, b))
                    preds = [p for p in oj.predicates if p.is_join()]
                    edges.append({
                        "source": a, "target": b, "kind": oj.kind,
                        "selectivity": (preds[0].selectivity if preds else None),
                        "source_of_estimate": (preds[0].source if preds else None),
                        "predicates": [p.text for p in oj.predicates],
                    })
        return {"nodes": nodes, "edges": edges}

    def _predicate_table(self, query: Query, est: Estimator) -> List[Dict[str, Any]]:
        out = []

        def add(p, scope):
            out.append({
                "text": p.text, "kind": p.kind, "scope": scope,
                "selectivity": p.selectivity, "source": p.source,
                "note": p.note,
            })
        for unit in query.semi_units:
            for p in unit.correlation:
                add(p, "correlation")
            for p in unit.inner_joins:
                add(p, "inner-join")
            for filt in unit.inner_filters.values():
                for p in filt:
                    add(p, "subquery-filter")
        for p in query.inner_joins:
            add(p, "join")
        for filters in query.local_filters.values():
            for p in filters:
                add(p, "filter")
        for oj in query.outer_joins:
            for p in oj.predicates:
                add(p, "outer-join")
        return out

    def _timing(self, conn, query: Query, result: OrderResult,
                config: OptimizerConfig) -> Dict[str, Any]:
        """Actual timings: chosen plan vs. the reversed order vs. SQLite's own."""
        chosen_ms, chosen_err = timed_plan(conn, query, result.plan,
                                           repeats=2, timeout_ms=3000)
        info: Dict[str, Any] = {
            "chosen_ms": chosen_ms,
            "chosen_error": chosen_err,
        }
        # An expensive alternative order, executed through the same
        # materialised-CTE pipeline so it is the *order* being measured, not
        # SQLite's own planner.  We pick the slowest permutation among a few
        # candidates (by estimated cost) instead of blindly reversing.
        try:
            if not query.outer_joins and not query.semi_units and result.order:
                from .execute import PipelineBuilder, execute_timed
                from .search import plan_for_order
                import itertools
                worst = None
                if len(result.order) <= 6:
                    candidates = list(itertools.permutations(result.order))
                else:
                    candidates = [tuple(reversed(result.order)),
                                  tuple(sorted(result.order)),
                                  tuple(result.order[1:] + result.order[:1])]
                for perm in candidates:
                    plan_try = plan_for_order(
                        query, self.stats(), list(perm), config)
                    if worst is None or plan_try.cost > worst[0]:
                        worst = (plan_try.cost, list(perm), plan_try)
                if worst is not None:
                    cost, order_try, plan_try = worst
                    bad_sql = PipelineBuilder(conn, query).build(plan_try)
                    info["slow_order"] = order_try
                    info["slow_sql"] = bad_sql
                    info["slow_est_rows"] = plan_try.est_rows
                    info["slow_cost"] = cost
                    info["slow_ms"] = execute_timed(conn, bad_sql, repeats=2,
                                                    timeout_ms=3000)
        except Exception as exc:
            info["slow_error"] = f"{type(exc).__name__}: {exc}"
        # SQLite's own choice
        try:
            info["sqlite_ms"] = _time_sql(conn, query.original_sql)
        except Exception as exc:
            info["sqlite_error"] = str(exc)
        for key in ("chosen_ms", "slow_ms", "sqlite_ms"):
            if info.get(key) == float("inf"):
                info[key] = None
                info[key + "_timeout"] = True
        return info

    # -- analysis utilities ----------------------------------------------
    def enumeration_stats(self, sizes: Sequence[int] = (3, 4, 5, 6, 7, 8)
                          ) -> Dict[str, Any]:
        return explain_mod.shape_summary(sizes)

    def compare_with_brute_force(self, sql: str,
                                 config: Optional[OptimizerConfig] = None
                                 ) -> Dict[str, Any]:
        config = config or OptimizerConfig()
        query = analyze(sql)
        stats = self.stats()
        opt = Optimizer(query, stats, config)
        result = opt.optimize()
        if len(opt.atoms) > 8:
            return {"skipped": True, "reason": "原子数超过 8，穷举代价过高",
                    "dp_cost": result.total_cost, "dp_order": result.order}
        if query.outer_joins or query.semi_units:
            return {"skipped": True,
                    "reason": "含外连接 / 子查询，穷举对比仅覆盖单阶段内连接",
                    "dp_cost": result.total_cost, "dp_order": result.order}
        cost, order, n = brute_force(query, stats, config)
        return {
            "skipped": False,
            "brute_cost": cost,
            "brute_order": order,
            "perms": n,
            "dp_cost": result.total_cost,
            "dp_order": result.order,
            "match": abs(cost - result.total_cost) < 1e-6,
            "subproblems": result.subproblems,
            "transitions": result.transitions,
        }


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def _time_sql(conn: sqlite3.Connection, sql: str, repeats: int = 3) -> float:
    from .execute import execute_timed
    return execute_timed(conn, sql, repeats=repeats, timeout_ms=3000)


# ---------------------------------------------------------------------------
# Demo database bootstrap
# ---------------------------------------------------------------------------


def default_db_path(root: str) -> str:
    return os.path.join(root, "data", "tea.db")


def ensure_demo_database(root: str, force: bool = False) -> str:
    path = default_db_path(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if force or not os.path.exists(path):
        create_database(path)
    return path
