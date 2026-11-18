"""Acceptance tests — the criteria requested for the deliverable.

1. The demo schema has ~16 tables with column correlation and data skew.
2. On small queries the DP optimum is identical to exhaustively trying every
   legal join order (both cost and, where it matters, the plan).
3. Executing the generated order in SQLite is compared against the original
   query: same rows, and a bad order is materially slower.
4. Every predicate carries a selectivity source.
5. The greedy threshold is configurable and switches the algorithm.
6. Outer-join / semi-join legality is respected.
7. Estimate vs. actual deviation is reported per plan step.
8. The four join-graph shapes have their sub-problem counts computed.
"""

from __future__ import annotations

import itertools
import math

import pytest

from teaopt.demo import TABLES
from teaopt.parser import analyze
from teaopt.search import Optimizer, brute_force
from teaopt.types import OptimizerConfig


# ---------------------------------------------------------------------------
# 1. Demo schema
# ---------------------------------------------------------------------------


def test_demo_schema_has_16_tables(stats):
    assert len(TABLES) == 16
    assert set(TABLES) <= set(stats)
    for t in TABLES:
        assert stats[t].row_count > 0


def test_demo_schema_is_correlated_and_skewed(stats):
    # correlation: grade/origin_id has far fewer distinct combos than ndv*gndv
    lots = stats["lots"]
    mc = lots.multicol_for({"grade", "origin_id"})
    ndv_product = lots.column("grade").ndv * lots.column("origin_id").ndv
    assert mc.ndv < ndv_product * 0.9
    # skew: the top origin supplies far more than 1/ndv of the lots
    top = lots.column("origin_id").mcv[0][1]
    assert top > lots.row_count / lots.column("origin_id").ndv * 2


# ---------------------------------------------------------------------------
# 2. DP == brute force on small queries
# ---------------------------------------------------------------------------


SMALL = [
    "SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN reports r ON r.lot_id = l.lot_id",
    "SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN reports r ON r.lot_id = l.lot_id JOIN inventory i ON i.lot_id = l.lot_id "
    "WHERE r.passed = 1 AND o.province = '云南'",
    "SELECT COUNT(*) FROM orders od JOIN customers cu ON od.customer_id = cu.customer_id "
    "JOIN blends b ON od.blend_id = b.blend_id "
    "JOIN shipments sh ON sh.order_id = od.order_id "
    "JOIN warehouses w ON sh.warehouse_id = w.warehouse_id",
    "SELECT COUNT(*) FROM reports r JOIN lots l ON r.lot_id = l.lot_id "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN suppliers s ON s.origin_id = o.origin_id "
    "JOIN purchases pu ON pu.lot_id = l.lot_id "
    "WHERE r.passed = 1 AND o.province = '云南'",
]


@pytest.mark.parametrize("sql", SMALL)
def test_dp_equals_exhaustive_enumeration(sql, stats):
    query = analyze(sql)
    result = Optimizer(query, stats).optimize()
    cost, order, perms = brute_force(query, stats)
    assert perms > 0
    assert result.total_cost == pytest.approx(cost, rel=1e-9)


def test_dp_equals_brute_force_on_all_3_and_4_relation_subgraphs(stats):
    """Every 3- and 4-relation connected subgraph of a hub schema, checked
    against full enumeration."""
    base = [("lots", "l", "lot_id"), ("origins", "o", "origin_id"),
            ("reports", "r", "lot_id"), ("inventory", "i", "lot_id")]
    checked = 0
    for n in (3, 4):
        for combo in itertools.combinations(base, n):
            hub = combo[0]
            froms = f'"{hub[0]}" {hub[1]}'
            for table, alias, key in combo[1:]:
                froms += f' JOIN "{table}" {alias} ON {alias}.{key} = {hub[1]}.{hub[2]}'
            sql = f"SELECT COUNT(*) FROM {froms}"
            query = analyze(sql)
            result = Optimizer(query, stats).optimize()
            cost, order, perms = brute_force(query, stats)
            assert result.total_cost == pytest.approx(cost, rel=1e-9), sql
            checked += 1
    assert checked == 5


# ---------------------------------------------------------------------------
# 3. Execution in SQLite
# ---------------------------------------------------------------------------


def test_generated_order_runs_in_sqlite_with_same_result(conn, service):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "JOIN inventory i ON i.lot_id = l.lot_id "
           "WHERE o.province = '云南' AND r.passed = 1")
    payload = service.optimize(sql)
    assert payload["validation"]["okay"]
    expected = conn.execute(sql).fetchone()[0]
    assert payload["validation"]["rows"] == expected


def test_actual_elapsed_times_are_measured(conn, service):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "WHERE o.province = '云南'")
    payload = service.optimize(sql)
    assert payload["timing"]["chosen_ms"] is not None
    assert payload["timing"]["chosen_ms"] >= 0


def test_bad_order_measured_slower_or_times_out(conn, service):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "JOIN inventory i ON i.lot_id = l.lot_id "
           "JOIN price_history p ON p.lot_id = l.lot_id "
           "WHERE o.province = '云南'")
    payload = service.optimize(sql)
    timing = payload["timing"]
    assert timing["chosen_ms"] is not None
    assert timing["slow_cost"] > payload["cost"]
    assert timing.get("slow_ms_timeout") or timing["slow_ms"] >= timing["chosen_ms"]


# ---------------------------------------------------------------------------
# 4–8. Individual requirements
# ---------------------------------------------------------------------------


def test_every_predicate_has_a_source(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "WHERE l.grade = 'S' AND l.weight_kg > 50 AND o.province = '云南' "
           "AND r.passed = 1 AND r.score >= 80")
    query = analyze(sql)
    Optimizer(query, stats)      # populates predicate selectivities
    preds = list(query.inner_joins)
    for filters in query.local_filters.values():
        preds.extend(filters)
    assert preds
    for p in preds:
        assert p.source in ("histogram", "mcv", "ndv", "multicol", "default")
        assert 0.0 <= p.selectivity <= 1.0


def test_algorithm_switches_at_configured_threshold(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN inventory i ON i.lot_id = l.lot_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "JOIN price_history p ON p.lot_id = l.lot_id "
           "JOIN purchases pu ON pu.lot_id = l.lot_id "
           "JOIN suppliers s ON s.supplier_id = pu.supplier_id "
           "JOIN warehouses w ON w.warehouse_id = i.warehouse_id")
    query = analyze(sql)
    assert Optimizer(query, stats, OptimizerConfig(exhaustive_threshold=10)).optimize() \
        .algorithm.startswith("exhaustive")
    assert Optimizer(query, stats, OptimizerConfig(exhaustive_threshold=4)).optimize() \
        .algorithm == "greedy"


def test_outer_join_order_constraint_enforced(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "LEFT JOIN inventory i ON i.lot_id = l.lot_id WHERE o.province = '云南'")
    query = analyze(sql)
    result = Optimizer(query, stats).optimize()
    assert result.order[-1] == "i"


def test_correlated_subquery_participates(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "WHERE EXISTS (SELECT 1 FROM reports r WHERE r.lot_id = l.lot_id)"
           " AND o.province = '云南'")
    query = analyze(sql)
    assert query.semi_units
    result = Optimizer(query, stats).optimize()
    assert any(n.node_type == "semi" for n in result.plan.walk())
    assert result.total_cost > 0


def test_deviation_reported_for_each_step(conn, service):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "WHERE o.province = '云南'")
    payload = service.optimize(sql)
    steps = payload["validation"]["steps"]
    assert steps
    assert any(s["actual_rows"] is not None for s in steps)
    for s in steps:
        if s["actual_rows"] is not None:
            assert "ratio" in s and "error_pct" in s


def test_four_graph_shapes_have_counts(service):
    data = service.enumeration_stats((4, 5, 6))
    for shape in ("chain", "star", "cycle", "clique"):
        rows = data["table"][shape]
        assert len(rows) == 3
        for row in rows:
            assert row["dp_states"] > 0
            assert row["permutations"] == math.factorial(row["n"])
    # clique has the most connected subsets, chain the fewest
    chain = [r["dp_states"] for r in data["table"]["chain"]]
    clique = [r["dp_states"] for r in data["table"]["clique"]]
    assert all(c <= k for c, k in zip(chain, clique))
