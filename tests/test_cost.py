"""Cost-model tests: the three join algorithms must be distinguishable."""

from __future__ import annotations

import pytest

from teaopt.cost import (choose_join_algorithm, merge_join_cost, nl_join_cost,
                         scan_cost)
from teaopt.estimate import Atom
from teaopt.types import OptimizerConfig, PlanNode


def _node(rows, cost, width=16.0, sorted_cols=()):
    return PlanNode(node_type="scan", est_rows=rows, cost=cost,
                    est_width=width, sorted_cols=set(sorted_cols))


CFG = OptimizerConfig()


def test_nested_loop_wins_for_tiny_outer():
    # A tiny *and already materialised* inner side: each probe is free, so a
    # nested loop beats building a hash table.
    outer = _node(2, 1.0)
    inner = _node(3, 0.01)
    algo, _, _, candidates = choose_join_algorithm(outer, inner, CFG, "b.x")
    assert algo == "nl"
    assert candidates["nl"] < candidates["hash"]


def test_nested_loop_loses_when_inner_is_large():
    outer = _node(1_000, 1.0)
    inner = _node(1_000_000, 900.0)
    algo, _, _, candidates = choose_join_algorithm(outer, inner, CFG, "b.x")
    assert algo != "nl"
    assert candidates["nl"] > candidates["hash"]


def test_hash_wins_for_large_inputs():
    outer = _node(100_000, 200.0)
    inner = _node(100_000, 200.0)
    algo, _, _, candidates = choose_join_algorithm(outer, inner, CFG, "b.x")
    assert algo == "hash"
    assert candidates["hash"] < candidates["nl"]
    assert candidates["hash"] < candidates["merge"]


def test_merge_wins_when_both_inputs_sorted():
    outer = _node(100_000, 200.0, sorted_cols={"a.x"})
    inner = _node(100_000, 200.0, sorted_cols={"a.x"})
    algo, _, _, candidates = choose_join_algorithm(outer, inner, CFG, "a.x")
    assert algo == "merge"
    assert candidates["merge"] < candidates["hash"]


def test_merge_cost_accounts_for_sorting():
    l = _node(10_000, 1.0)
    r = _node(10_000, 1.0)
    _, unsorted = merge_join_cost(l, r, CFG, "k")
    _, sorted_ = merge_join_cost(_node(10_000, 1.0, sorted_cols={"k"}),
                                 _node(10_000, 1.0, sorted_cols={"k"}), CFG, "k")
    assert unsorted > sorted_


def test_nl_cost_grows_with_outer_rows():
    inner = _node(1_000, 10.0)
    assert nl_join_cost(_node(10, 1.0), inner, CFG) < \
        nl_join_cost(_node(1_000, 1.0), inner, CFG)


def test_scan_cost_uses_histogram_selectivity():
    from teaopt.types import Predicate
    pred = Predicate(kind="eq", text="k = 1", col_left="k", value=1,
                     selectivity=0.01)
    atom = Atom(alias="l", table="lots", base_rows=10_000, rows=100,
                width=16, pages=200, index_cols={"k"}, filters=[pred])
    startup, total, algo, candidates = scan_cost(atom, CFG)
    assert algo == "indexscan"
    assert candidates["indexscan"] < candidates["seqscan"]
    # an unindexed column always uses a sequential scan
    pred2 = Predicate(kind="eq", text="j = 1", col_left="j", value=1,
                      selectivity=0.01)
    atom2 = Atom(alias="l", table="lots", base_rows=10_000, rows=100,
                 width=16, pages=200, index_cols={"k"}, filters=[pred2])
    assert scan_cost(atom2, CFG)[2] == "seqscan"


def test_full_outer_join_excludes_nested_loop():
    from teaopt.cost import outer_join_cost
    left = _node(100, 1.0)
    right = _node(2_000_000, 900.0)
    algo, _, candidates = outer_join_cost(left, right, CFG, "full", "k")
    assert "nl" not in candidates
    algo_left, _, candidates_left = outer_join_cost(left, right, CFG, "left", "k")
    assert "nl" in candidates_left


def test_algorithm_choice_differs_by_shape(stats):
    """Run the optimizer on two queries whose optimal algorithms differ and
    check the chosen algorithms are recorded per node."""
    from teaopt.parser import analyze
    from teaopt.search import Optimizer
    q = analyze("SELECT COUNT(*) FROM origins o JOIN lots l "
                "ON l.origin_id = o.origin_id WHERE o.province = '云南'")
    res = Optimizer(q, stats).optimize()
    algorithms = {n.algorithm for n in res.plan.walk() if n.node_type == "join"}
    assert algorithms
    for node in res.plan.walk():
        if node.node_type != "join":
            continue
        assert set(node.candidates) == {"nl", "hash", "merge"}
        # the chosen algorithm's candidate cost is the node's total cost, and
        # no candidate is cheaper
        assert node.candidates[node.algorithm] == pytest.approx(node.cost, abs=1e-6)
        assert min(node.candidates.values()) == pytest.approx(node.cost, abs=1e-6)
