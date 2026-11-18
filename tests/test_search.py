"""Search tests: exhaustive DP vs. brute force, greedy threshold, legality."""

from __future__ import annotations

import pytest

from teaopt.parser import analyze
from teaopt.search import Optimizer, brute_force, plan_for_order, _legal_order
from teaopt.types import OptimizerConfig

# Queries whose join graph is small enough for brute force (<= 6 relations).
SMALL_QUERIES = [
    # chain
    "SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN reports r ON r.lot_id = l.lot_id",
    # star (lots is the hub, three children)
    "SELECT COUNT(*) FROM lots l "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN reports r ON r.lot_id = l.lot_id "
    "JOIN inventory i ON i.lot_id = l.lot_id "
    "JOIN price_history p ON p.lot_id = l.lot_id",
    # two chains crossing
    "SELECT COUNT(*) FROM orders od "
    "JOIN customers cu ON od.customer_id = cu.customer_id "
    "JOIN blends b ON od.blend_id = b.blend_id "
    "JOIN shipments sh ON sh.order_id = od.order_id "
    "JOIN warehouses w ON sh.warehouse_id = w.warehouse_id",
    # with selective filters
    "SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN reports r ON r.lot_id = l.lot_id "
    "WHERE o.province = '云南' AND r.passed = 1 AND l.grade = 'S'",
    # 6 atoms
    "SELECT COUNT(*) FROM blends b JOIN blend_lots bl ON bl.blend_id = b.blend_id "
    "JOIN lots l ON bl.lot_id = l.lot_id "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN suppliers s ON s.origin_id = o.origin_id "
    "JOIN purchases pu ON pu.lot_id = l.lot_id",
]


# ---------------------------------------------------------------------------
# Requirement: DP == exhaustive enumeration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sql", SMALL_QUERIES)
def test_dp_matches_brute_force(sql, stats):
    query = analyze(sql)
    opt = Optimizer(query, stats)
    result = opt.optimize()
    cost, order, perms = brute_force(query, stats)
    assert perms > 0
    assert result.total_cost == pytest.approx(cost, rel=1e-9), (
        f"DP {result.total_cost} != brute force {cost} (DP order {result.order}, "
        f"brute order {order})")


@pytest.mark.parametrize("seed", range(6))
def test_dp_matches_brute_force_on_random_graphs(seed, stats):
    """Random subgraphs of the demo schema, 3..6 relations."""
    import random
    rng = random.Random(seed + 100)
    candidates = [
        ("lots", "l", "lot_id"),
        ("origins", "o", "origin_id"),
        ("reports", "r", "lot_id"),
        ("inventory", "i", "lot_id"),
        ("price_history", "p", "lot_id"),
        ("purchases", "pu", "lot_id"),
        ("suppliers", "s", "origin_id"),
    ]
    n = rng.randint(3, 6)
    hub = candidates[0]
    chosen = [hub] + rng.sample(candidates[1:], n - 1)
    froms = f'"{hub[0]}" {hub[1]}'
    for table, alias, key in chosen[1:]:
        froms += f' JOIN "{table}" {alias} ON {alias}.{key} = {hub[1]}.{hub[2]}'
    sql = f"SELECT COUNT(*) FROM {froms}"
    query = analyze(sql)
    result = Optimizer(query, stats).optimize()
    cost, order, perms = brute_force(query, stats)
    assert result.total_cost == pytest.approx(cost, rel=1e-9)


def test_dp_states_are_connected_subsets(stats):
    sql = SMALL_QUERIES[1]
    query = analyze(sql)
    result = Optimizer(query, stats).optimize()
    n = len(query.tables)
    # star with n atoms: every connected subset of size >= 2 must contain the
    # hub, so there are 2^(n-1) - 1 such states (the singleton states are the
    # seeds and are not counted as DP sub-problems)
    expected = 2 ** (n - 1) - 1
    assert result.subproblems == expected
    assert result.transitions > 0


# ---------------------------------------------------------------------------
# Greedy threshold (requirement 2)
# ---------------------------------------------------------------------------


BIG_QUERY = (
    "SELECT COUNT(*) FROM lots l "
    "JOIN origins o ON l.origin_id = o.origin_id "
    "JOIN inventory i ON i.lot_id = l.lot_id "
    "JOIN reports r ON r.lot_id = l.lot_id "
    "JOIN price_history p ON p.lot_id = l.lot_id "
    "JOIN purchases pu ON pu.lot_id = l.lot_id "
    "JOIN suppliers s ON s.supplier_id = pu.supplier_id "
    "JOIN warehouses w ON w.warehouse_id = i.warehouse_id "
    "WHERE o.province = '云南' AND r.passed = 1")


def test_greedy_used_above_threshold(stats):
    query = analyze(BIG_QUERY)
    result = Optimizer(query, stats,
                       OptimizerConfig(exhaustive_threshold=5)).optimize()
    assert result.algorithm == "greedy"
    assert any("阈值" in n and "贪心" in n for n in result.notes)
    assert result.greedy_cost is not None


def test_exhaustive_used_at_or_below_threshold(stats):
    query = analyze(BIG_QUERY)
    result = Optimizer(query, stats,
                       OptimizerConfig(exhaustive_threshold=10)).optimize()
    assert result.algorithm.startswith("exhaustive")


def test_threshold_is_configurable(stats):
    query = analyze(BIG_QUERY)
    greedy = Optimizer(query, stats,
                       OptimizerConfig(exhaustive_threshold=3)).optimize()
    exact = Optimizer(query, stats,
                      OptimizerConfig(exhaustive_threshold=20)).optimize()
    assert greedy.algorithm == "greedy"
    assert exact.algorithm.startswith("exhaustive")
    # greedy never beats the exact optimum, and may be worse
    assert greedy.total_cost >= exact.total_cost - 1e-9


def test_forced_greedy_is_reported(stats):
    query = analyze(SMALL_QUERIES[0])
    result = Optimizer(query, stats,
                       OptimizerConfig(algorithm="greedy")).optimize()
    assert result.algorithm == "greedy"
    assert len(result.order) == 3


def test_greedy_gap_is_quantified_when_exact_is_available(stats):
    """Above the threshold the answer stays greedy, but the exact optimum is
    still computed (when cheap) so the gap can be shown."""
    query = analyze(BIG_QUERY)
    result = Optimizer(query, stats,
                       OptimizerConfig(exhaustive_threshold=5)).optimize()
    assert result.stats.get("exact_cost") is not None
    assert result.stats["exact_cost"] <= result.total_cost + 1e-9
    assert any("差距" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Legality (requirement 5)
# ---------------------------------------------------------------------------


def test_outer_join_nullable_side_stays_right(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "LEFT JOIN inventory i ON i.lot_id = l.lot_id "
           "WHERE o.province = '云南'")
    query = analyze(sql)
    result = Optimizer(query, stats).optimize()
    assert result.order.index("i") == len(result.order) - 1
    # the plan's outer node must produce i on the right-hand side
    outer_nodes = [n for n in result.plan.walk() if n.node_type == "outer"]
    assert len(outer_nodes) == 1
    assert outer_nodes[0].right.relations() == {"i"}


def test_brute_force_only_returns_legal_orders(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "LEFT JOIN inventory i ON i.lot_id = l.lot_id "
           "WHERE o.province = '云南'")
    query = analyze(sql)
    cost, order, perms = brute_force(query, stats)
    assert order.index("i") > order.index("l")
    # illegal orders are never considered
    assert perms > 0


def test_cross_stage_predicate_used_when_attaching(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "LEFT JOIN inventory i ON i.lot_id = l.lot_id "
           "JOIN warehouses w ON w.warehouse_id = i.warehouse_id "
           "WHERE o.province = '云南'")
    query = analyze(sql)
    result = Optimizer(query, stats).optimize()
    # w is attached after the outer join, using w.warehouse_id = i.warehouse_id
    outer_node = next(n for n in result.plan.walk() if n.node_type == "outer")
    assert "i" in outer_node.relations()
    assert result.total_cost > 0


def test_legal_order_helper():
    q = analyze("SELECT * FROM a JOIN b ON a.x=b.x LEFT JOIN c ON c.x=b.x")
    cfg = OptimizerConfig()
    assert _legal_order(q, ["a", "b", "c"], cfg)
    assert _legal_order(q, ["b", "a", "c"], cfg)
    assert not _legal_order(q, ["c", "a", "b"], cfg)
    assert not _legal_order(q, ["a", "c", "b"], cfg)


# ---------------------------------------------------------------------------
# Semi / anti units participate in the enumeration (requirement 6)
# ---------------------------------------------------------------------------


def test_semi_unit_is_placed_in_enumeration(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "WHERE EXISTS (SELECT 1 FROM reports r WHERE r.lot_id = l.lot_id "
           "              AND r.passed = 1) AND o.province = '云南'")
    query = analyze(sql)
    result = Optimizer(query, stats).optimize()
    semi_nodes = [n for n in result.plan.walk() if n.node_type == "semi"]
    assert len(semi_nodes) == 1
    # the semi join is above the l/o join (its outer ref is l)
    assert "l" in semi_nodes[0].left.relations()
    assert any("半连接" in n for n in result.notes)


def test_anti_unit_records_selectivity_source(stats):
    sql = ("SELECT COUNT(*) FROM orders od JOIN customers cu "
           "ON od.customer_id = cu.customer_id "
           "WHERE NOT EXISTS (SELECT 1 FROM shipments sh "
           "                  WHERE sh.order_id = od.order_id)")
    query = analyze(sql)
    result = Optimizer(query, stats).optimize()
    anti = [n for n in result.plan.walk() if n.node_type == "anti"]
    assert len(anti) == 1
    assert anti[0].predicates[0].source in ("histogram", "ndv", "default")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_plan_for_order_reproduces_cost(stats):
    query = analyze(SMALL_QUERIES[0])
    result = Optimizer(query, stats).optimize()
    replan = plan_for_order(query, stats, result.order)
    assert replan.cost == pytest.approx(result.total_cost, rel=1e-9)
    assert replan.est_rows == pytest.approx(result.est_rows)


def test_empty_join_uses_no_cartesian_by_default(stats):
    query = analyze("SELECT COUNT(*) FROM lots l, origins o")  # disconnected
    result = Optimizer(query, stats).optimize()
    # the fallback still joins, but the plan marks it as a cartesian product
    join_nodes = [n for n in result.plan.walk() if n.node_type == "join"]
    assert join_nodes and join_nodes[0].extra.get("cartesian")
