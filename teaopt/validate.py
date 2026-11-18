"""Estimate vs. actual validation on SQLite.

Runs the chosen plan with the join order enforced (MATERIALIZED CTE chain),
then measures every intermediate result so the UI can show the deviation
between the optimizer's row estimate and the truth (requirement 7).
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from .execute import PipelineBuilder, execute, execute_timed, measure_step_rows
from .explain import compare_estimates
from .search import plan_for_order
from .types import OrderResult, PlanNode, Query, TableStats


def validate(conn: sqlite3.Connection, query: Query, result: OrderResult,
             sample: int = 5, timeout_ms: int = 4000) -> Dict[str, Any]:
    """Execute the chosen plan and return estimates vs. actuals.

    A time budget keeps the page responsive when the query is genuinely large;
    the report then says "超时" instead of hanging.
    """
    builder = PipelineBuilder(conn, query)
    try:
        sql = builder.build(result.plan)
    except Exception as exc:
        return {"okay": False,
                "error": f"无法生成执行 SQL: {type(exc).__name__}: {exc}",
                "steps": [], "sql": ""}
    execution = execute(conn, sql, sample=sample, timeout_ms=timeout_ms)
    todo = {"okay": execution.okay, "error": execution.error,
            "sql": sql, "elapsed_ms": execution.elapsed_ms,
            "rows": execution.count_value if execution.count_value is not None
                    else execution.rows,
            "sample": execution.sample,
            "explain": execution.explain}
    if not execution.okay:
        return {**todo, "steps": []}
    # measure each plan node by counting the CTE prefix
    actual = measure_step_rows(conn, query, result.plan,
                               timeout_ms=min(1500, timeout_ms))
    steps = compare_estimates(result.plan, actual)
    return {**todo, "steps": steps,
            "truncated": len(actual) < sum(
                1 for n in result.plan.walk() if n.node_type != "scan")}


def timed_plan(conn: sqlite3.Connection, query: Query, plan: PlanNode,
               repeats: int = 3, timeout_ms: int = 3000) -> Tuple[float, Optional[str]]:
    try:
        sql = PipelineBuilder(conn, query).build(plan)
    except Exception as exc:
        return float("inf"), str(exc)
    elapsed = execute_timed(conn, sql, repeats=repeats, timeout_ms=timeout_ms)
    if elapsed == float("inf"):
        return elapsed, f"超时（>{timeout_ms} ms）"
    return elapsed, None


def plan_for(conn: sqlite3.Connection, query: Query, order: List[str],
             stats: Dict[str, TableStats], config=None) -> PlanNode:
    """Build the plan for an explicit order (re-exported for convenience)."""
    return plan_for_order(query, stats, order, config)
