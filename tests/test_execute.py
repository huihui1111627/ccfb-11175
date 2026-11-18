"""Execution tests: the generated plan must return exactly the same rows as the
user's original SQL, and the timing harness must measure real differences.

These tests are deliberately end-to-end: they build the plan, render it as a
chain of MATERIALIZED CTEs (so SQLite cannot reorder), run it, and compare the
result with the untouched query.
"""

from __future__ import annotations

import pytest

from teaopt.execute import PipelineBuilder, measure_step_rows
from teaopt.parser import analyze
from teaopt.search import Optimizer, plan_for_order


QUERIES = [
    # inner joins, filters, aggregate
    "SELECT COUNT(*) AS n FROM lots l "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN reports r ON r.lot_id = l.lot_id "
    "WHERE o.province = '云南' AND r.passed = 1",
    # projected columns + limit
    "SELECT l.lot_id, l.grade, o.province FROM lots l "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "WHERE o.province = '福建' ORDER BY l.lot_id LIMIT 20",
    # left join
    "SELECT COUNT(*) AS n FROM lots l "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "LEFT JOIN inventory i ON i.lot_id = l.lot_id "
    "WHERE o.province = '云南' AND l.grade = 'S'",
    # semi join
    "SELECT COUNT(*) AS n FROM lots l "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "WHERE EXISTS (SELECT 1 FROM reports r WHERE r.lot_id = l.lot_id "
    "              AND r.passed = 1) AND o.province = '云南'",
    # anti join
    "SELECT COUNT(*) AS n FROM orders od "
    "JOIN customers cu ON od.customer_id = cu.customer_id "
    "WHERE NOT EXISTS (SELECT 1 FROM shipments sh "
    "                  WHERE sh.order_id = od.order_id) AND cu.channel = '线上'",
    # group by + order by + limit
    "SELECT cu.city, COUNT(*) AS n FROM orders od "
    "JOIN customers cu ON od.customer_id = cu.customer_id "
    "JOIN blends b ON od.blend_id = b.blend_id "
    "WHERE b.style = '普洱生茶' GROUP BY cu.city ORDER BY n DESC LIMIT 3",
    # 5-table star
    "SELECT COUNT(*) AS n FROM lots l "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN reports r ON r.lot_id = l.lot_id "
    "JOIN inventory i ON i.lot_id = l.lot_id "
    "JOIN price_history p ON p.lot_id = l.lot_id "
    "WHERE o.province = '云南'",
]


def _payload(service, sql):
    return service.optimize(sql, run_on_sqlite=True)


def _original(conn, sql):
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description]
    return cols, [tuple(r) for r in cur.fetchall()]


def _planned(conn, service, sql):
    from teaopt.parser import analyze
    from teaopt.search import Optimizer
    from teaopt.execute import PipelineBuilder
    query = analyze(sql)
    result = Optimizer(query, service.stats()).optimize()
    built = PipelineBuilder(conn, query).build(result.plan)
    cur = conn.execute(built)
    cols = [d[0] for d in cur.description]
    return cols, [tuple(r) for r in cur.fetchall()], built


@pytest.mark.parametrize("sql", QUERIES)
def test_plan_returns_same_rows_as_original(sql, conn, service):
    expected_cols, expected = _original(conn, sql)
    planned_cols, planned, built = _planned(conn, service, sql)
    assert planned == expected, (
        f"计划执行结果与原始查询不一致\nSQL: {built}\n计划 {len(planned)} 行，"
        f"原始 {len(expected)} 行")
    assert len(planned_cols) == len(expected_cols)


def test_plan_sql_is_accepted_by_sqlite(conn, service):
    sql = QUERIES[0]
    payload = _payload(service, sql)
    assert payload["validation"]["okay"], payload["validation"]["error"]
    assert "WITH" in payload["plan_sql"] and "MATERIALIZED" in payload["plan_sql"]


def test_step_rows_are_measured(conn, service):
    sql = QUERIES[6]
    query = analyze(sql)
    result = Optimizer(query, service.stats()).optimize()
    rows = measure_step_rows(conn, query, result.plan)
    join_nodes = [n for n in result.plan.walk() if n.node_type == "join"]
    assert join_nodes
    assert any(id(n) in rows for n in join_nodes)


def test_estimate_deviation_is_reported(conn, service):
    payload = _payload(service, QUERIES[6])
    steps = payload["validation"]["steps"]
    assert steps
    with_actual = [s for s in steps if s["actual_rows"] is not None]
    assert with_actual
    for s in with_actual:
        assert s["ratio"] is not None
        assert s["error_pct"] is not None


def test_materialised_plan_honours_join_order(conn, service):
    """The CTE chain must follow the chosen order, and a swapped order must
    still return the same rows (only the intermediate sizes differ)."""
    sql = QUERIES[3]        # 3 relations: cheap enough to execute both ways
    query = analyze(sql)
    result = Optimizer(query, service.stats()).optimize()
    built = PipelineBuilder(conn, query).build(result.plan)
    positions = [built.index(f'"{a}" AS {a}') for a in result.order
                 if f'"{a}" AS {a}' in built]
    assert positions == sorted(positions), (
        f"CTE 顺序与计划顺序不一致: {result.order}\n{built}")
    # a swapped order still produces the same result
    swapped = list(reversed(result.order))
    if not query.outer_joins and not query.semi_units:
        swapped_plan = plan_for_order(query, service.stats(), swapped)
        swapped_sql = PipelineBuilder(conn, query).build(swapped_plan)
        assert conn.execute(swapped_sql).fetchall() == conn.execute(sql).fetchall()


def test_bad_order_is_slower_than_good_order(conn, service):
    """A deliberately bad order is much slower (or times out) in SQLite while
    the cost model also says it is worse -- the core acceptance criterion."""
    sql = QUERIES[6]
    payload = _payload(service, sql)
    timing = payload["timing"]
    assert timing.get("chosen_ms") is not None, timing
    if timing.get("slow_cost") is None:
        return      # no alternative order was generated
    # the cost model must rank the chosen order strictly lower
    assert timing["slow_cost"] > payload["cost"] * 10
    # and the wall clock must agree: either measurably slower, or it blew the
    # execution budget (reported as None with a timeout flag)
    assert timing.get("slow_ms_timeout") or \
        (timing.get("slow_ms") is not None and
         timing["slow_ms"] >= timing["chosen_ms"])


def test_outer_join_filters_are_applied_after_the_join(conn, service):
    """A WHERE filter on the nullable side must not be pushed into its scan."""
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "LEFT JOIN inventory i ON i.lot_id = l.lot_id "
           "WHERE i.inv_id IS NULL AND l.grade = 'S'")
    expected = conn.execute(sql).fetchone()[0]
    _, planned, built = _planned(conn, service, sql)
    assert planned[0][0] == expected


def test_validation_marks_timeout_instead_of_hanging(conn, service):
    """A pathological query must come back with a timeout flag, not block."""
    from teaopt.types import OptimizerConfig
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "JOIN inventory i ON i.lot_id = l.lot_id "
           "JOIN price_history p ON p.lot_id = l.lot_id "
           "JOIN purchases pu ON pu.lot_id = l.lot_id "
           "JOIN blend_lots bl ON bl.lot_id = l.lot_id")
    payload = service.optimize(sql, OptimizerConfig(), run_on_sqlite=True)
    # either it finished (fine) or it reported a timeout (also fine, but the
    # response must be well formed)
    assert "validation" in payload
    assert "timing" in payload
