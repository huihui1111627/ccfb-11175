"""Cost model.

The cost model follows the classic PostgreSQL textbook model:

* scans are charged ``pages * seq_page_cost`` plus a per-row CPU cost,
* index scans are charged ``height + pages * random_page_cost`` when the
  predicate is an equality on an indexed column, else a sequential scan wins,
* joins are charged differently per algorithm (requirement 4):

  - **Nested loop**     ``outer_cost + outer_rows * inner_cost_of_one_probe``,
    the inner side is re-scanned for each outer row, which makes NL the right
    choice only for small outer inputs.
  - **Hash join**       build the hash table on the *smaller* input, one pass
    over both inputs plus a CPU cost per row.
  - **Sort-merge join** both inputs are sorted on the join key (or reused
    sorted), then merged in one pass.  Attractive when an input is already
    sorted (index order) or when the output order helps a downstream step.

The chosen algorithm is recorded with the costs of all candidates so the UI
can show *why* the optimizer picked it.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from .estimate import Atom
from .types import OptimizerConfig, PlanNode


# Cost of sorting ``n`` rows in memory; the classic 1.0 + log2(n) term.
def sort_cost(n: float, width: float, config: OptimizerConfig) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * n * math.log2(max(n, 2.0)) * config.cpu_operator_cost


def hash_build_cost(n: float, config: OptimizerConfig) -> float:
    return n * config.cpu_operator_cost * 1.5


def hash_probe_cost(n: float, config: OptimizerConfig) -> float:
    return n * config.cpu_operator_cost


def merge_cost(left_rows: float, right_rows: float, width: float,
               config: OptimizerConfig, left_sorted: bool, right_sorted: bool) -> float:
    cost = (left_rows + right_rows) * config.cpu_operator_cost
    if not left_sorted:
        cost += sort_cost(left_rows, width, config)
    if not right_sorted:
        cost += sort_cost(right_rows, width, config)
    return cost


def scan_cost(atom: Atom, config: OptimizerConfig) -> Tuple[float, float, str, Dict[str, float]]:
    """Return ``(startup, total, algorithm, candidates)`` for a base scan."""
    pages = max(atom.pages, 1.0)
    seq = pages * config.seq_page_cost + atom.base_rows * config.cpu_tuple_cost
    candidates = {"seqscan": seq}
    best = seq
    best_algo = "seqscan"
    best_startup = 0.0
    # index scan: only when the filters include an equality on an indexed
    # column.  An index scan touches only the matching pages, and an equality
    # lookup is a single descent, so it is modelled as
    #     height + matching_pages * random_page_cost + matching_rows * cpu
    eq_cols = [p.col_left for p in atom.filters
               if p.kind == "eq" and p.value is not None and p.col_left]
    if any(c in atom.index_cols for c in eq_cols if c):
        ts_pages = max(1.0, pages)
        height = max(1, int(math.log2(max(ts_pages, 2))) + 1)
        sel = atom.filter_selectivity
        matching_rows = atom.base_rows * sel
        # ceil, but at least one page is read to answer the lookup
        matching_pages = max(1.0, math.ceil(pages * sel))
        idx = (height + matching_pages * config.random_page_cost
               + matching_rows * config.cpu_index_tuple_cost)
        candidates["indexscan"] = idx
        if idx < best:
            best = idx
            best_algo = "indexscan"
            best_startup = height
    return best_startup, best, best_algo, candidates


def nl_join_cost(outer: PlanNode, inner: PlanNode, config: OptimizerConfig,
                 inner_is_atom: bool = True) -> float:
    """Classic nested loop with one rescan of the inner side per outer row.

    The inner side's *startup* is paid for every outer row (for a base scan the
    pages still have to be read again), so the rescans cost
    ``outer_rows * inner.cost``.  For an inner relation that is itself a join,
    materialising it once and probing it is what a hash join does, so a plain
    nested loop is charged the full recomputation.
    """
    return outer.cost + outer.est_rows * inner.cost


def hash_join_cost(left: PlanNode, right: PlanNode, config: OptimizerConfig) -> Tuple[float, float]:
    build = min(left.est_rows, right.est_rows)
    probe = max(left.est_rows, right.est_rows)
    cost = left.cost + right.cost + hash_build_cost(build, config) + hash_probe_cost(probe, config)
    startup = left.cost + right.cost + hash_build_cost(build, config)
    return startup, cost


def merge_join_cost(left: PlanNode, right: PlanNode, config: OptimizerConfig,
                    key: Optional[str]) -> Tuple[float, float]:
    left_sorted = key is not None and key in left.sorted_cols
    right_sorted = key is not None and key in right.sorted_cols
    width = max(left.est_width, right.est_width)
    cost = left.cost + right.cost + merge_cost(
        left.est_rows, right.est_rows, width, config, left_sorted, right_sorted)
    startup = left.cost + right.cost + (
        0 if left_sorted else sort_cost(left.est_rows, width, config)) + (
        0 if right_sorted else sort_cost(right.est_rows, width, config))
    return startup, cost


def semi_join_cost(left: PlanNode, inner: PlanNode, config: OptimizerConfig,
                   anti: bool, algorithm: str = "hash") -> Tuple[float, float, Dict[str, float]]:
    """Cost of a semi / anti join.  Hash is used unless the inner side is tiny."""
    candidates: Dict[str, float] = {}
    # hash semi join: build the inner hash table, probe with the outer rows
    hash_cost = left.cost + inner.cost + hash_build_cost(inner.est_rows, config) + \
        hash_probe_cost(left.est_rows, config)
    candidates["hash"] = hash_cost
    # nested loop semi join: inner is rescanned per outer row (can be cheap when
    # the inner is an index lookup, approximated by a smaller constant here)
    nl_cost = left.cost + inner.cost * max(1.0, left.est_rows) * 0.02
    candidates["nl"] = nl_cost
    algo = "hash"
    cost = hash_cost
    if nl_cost < hash_cost:
        algo = "nl"
        cost = nl_cost
    # merge semi join requires sorted inputs
    merge_cost_val = left.cost + inner.cost + merge_cost(
        left.est_rows, inner.est_rows, left.est_width, config, False, False)
    candidates["merge"] = merge_cost_val
    if merge_cost_val < cost:
        cost = merge_cost_val
        algo = "merge"
    return left.cost, cost, {"algorithm": algo, "candidates": candidates}


def choose_join_algorithm(left: PlanNode, right: PlanNode, config: OptimizerConfig,
                          join_key: Optional[str]) -> Tuple[str, float, float, Dict[str, float]]:
    """Return ``(algorithm, startup, total, candidate_costs)``."""
    candidates: Dict[str, float] = {}
    candidates["nl"] = nl_join_cost(left, right, config)
    h_start, h_total = hash_join_cost(left, right, config)
    candidates["hash"] = h_total
    m_start, m_total = merge_join_cost(left, right, config, join_key)
    candidates["merge"] = m_total
    algo = min(candidates, key=candidates.get)
    if algo == "nl":
        return "nl", left.cost, candidates["nl"], candidates
    if algo == "hash":
        return "hash", h_start, h_total, candidates
    return "merge", m_start, m_total, candidates


def outer_join_cost(left: PlanNode, right: PlanNode, config: OptimizerConfig,
                    kind: str, join_key: Optional[str]) -> Tuple[str, float, Dict[str, float]]:
    """Cost for LEFT / FULL joins.

    A hash join on the nullable side is the usual choice; a nested loop is
    cheaper when the preserved side is small.  FULL joins cannot use a plain
    nested loop (rows missing on either side must be emitted), so NL is
    excluded there.
    """
    candidates: Dict[str, float] = {}
    candidates["hash"] = hash_join_cost(left, right, config)[1]
    candidates["merge"] = merge_join_cost(left, right, config, join_key)[1]
    if kind != "full":
        candidates["nl"] = nl_join_cost(left, right, config)
    algo = min(candidates, key=candidates.get)
    return algo, candidates[algo], candidates
