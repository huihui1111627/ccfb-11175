"""Enumeration statistics for the four canonical join-graph shapes
(requirement 8)."""

from __future__ import annotations

import math

from teaopt.explain import (canonical_graph, count_connected_subsets,
                            count_transitions, enumeration_table, shape_of,
                            shape_summary)


def test_canonical_graphs():
    assert canonical_graph("chain", 4) == [(0, 1), (1, 2), (2, 3)]
    assert canonical_graph("star", 4) == [(0, 1), (0, 2), (0, 3)]
    assert canonical_graph("cycle", 4) == [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert len(canonical_graph("clique", 4)) == 6


def test_chain_subproblem_counts():
    # chain of n: connected subsets are the intervals, n(n+1)/2 of them
    for n in (3, 4, 5, 6):
        edges = canonical_graph("chain", n)
        assert count_connected_subsets(n, edges) == n * (n + 1) // 2


def test_star_subproblem_counts():
    # star of n: the n singletons, plus every subset of size >= 2 that
    # contains the hub -> n + (2^(n-1) - 1)
    for n in (3, 4, 5, 6):
        edges = canonical_graph("star", n)
        assert count_connected_subsets(n, edges) == n + (2 ** (n - 1) - 1)


def test_clique_subproblem_counts():
    # clique: every non-empty subset is connected
    for n in (3, 4, 5):
        edges = canonical_graph("clique", n)
        assert count_connected_subsets(n, edges) == 2 ** n - 1


def test_cycle_subproblem_counts_between_chain_and_clique():
    for n in (4, 5, 6):
        chain = count_connected_subsets(n, canonical_graph("chain", n))
        cycle = count_connected_subsets(n, canonical_graph("cycle", n))
        clique = count_connected_subsets(n, canonical_graph("clique", n))
        assert chain < cycle < clique


def test_transitions_are_bounded_by_subset_pairs():
    # a transition is (subset, relation) with a connecting edge, so its count is
    # bounded by |states| * n and stays far below the n! join orders once n
    # grows -- that is the entire point of subset DP.
    for shape in ("chain", "star", "cycle", "clique"):
        for n in (3, 4, 5, 6, 7):
            edges = canonical_graph(shape, n)
            transitions = count_transitions(n, edges)
            # every (subset, r) pair with an edge inside S is one transition;
            # connected subsets already include singletons that add none, so
            # the safe analytic bound is (number of all subsets) * n
            assert 0 < transitions <= (2 ** n) * n
        edges = canonical_graph(shape, 8)
        assert count_transitions(8, edges) < math.factorial(8)


def test_enumeration_table_rows():
    rows = enumeration_table(sizes=(3, 4))
    assert len(rows) == 4 * 2
    for row in rows:
        assert row["permutations"] == math.factorial(row["n"])
        assert row["dp_states"] >= row["n"]
        assert row["dp_transitions"] > 0


def test_shape_summary_has_all_shapes():
    summary = shape_summary((3, 4, 5))
    assert summary["shapes"] == ["chain", "star", "cycle", "clique"]
    for shape in summary["shapes"]:
        assert len(summary["table"][shape]) == 3


def test_shape_of_classification():
    assert shape_of([("a", "b"), ("b", "c"), ("c", "d")], "abcd") == "chain"
    assert shape_of([("h", "a"), ("h", "b"), ("h", "c")], "habc") == "star"
    assert shape_of([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")], "abcd") == "cycle"
    assert shape_of([("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"),
                     ("b", "d"), ("c", "d")], "abcd") == "clique"


def test_dp_states_grow_much_slower_than_permutations():
    for shape in ("chain", "star", "cycle", "clique"):
        for n in (7, 8, 9):
            edges = canonical_graph(shape, n)
            assert count_connected_subsets(n, edges) < math.factorial(n)
