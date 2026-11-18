"""Physical plan construction and per-node costing.

All search strategies (exhaustive DP, brute force, greedy) build plans with the
same factory, so their costs are directly comparable.

Design contract that makes the subset DP exact
----------------------------------------------
* Plan shape is left-deep.
* The row estimate of a set of relations ``S`` depends only on ``S`` (see
  :mod:`teaopt.estimate`), never on the join order.
* A join node's cost depends only on the child plans' costs, their row counts
  and the size of the result — and on whether a *singleton* child is an index
  scan (merge join can reuse its order).  Since a singleton relation has
  exactly one plan, this is again a function of the two subsets.

Under these conditions the cheapest plan for every subset is a sufficient
summary, which is exactly what the DP exploits.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from .cost import (choose_join_algorithm, outer_join_cost, scan_cost,
                   semi_join_cost)
from .estimate import Estimator
from .types import OptimizerConfig, PlanNode, Predicate, Query, SemiUnit, TableStats


class PlanFactory:
    def __init__(self, query: Query, stats: Dict[str, TableStats],
                 estimator: Estimator, config: Optional[OptimizerConfig] = None):
        self.query = query
        self.stats = stats
        self.est = estimator
        self.config = config or OptimizerConfig()
        self._scans: Dict[str, PlanNode] = {}

    # ------------------------------------------------------------------
    def scan(self, alias: str) -> PlanNode:
        if alias in self._scans:
            return self._scans[alias]
        atom = self.est.atoms[alias]
        startup, total, algo, candidates = scan_cost(atom, self.config)
        node = PlanNode(
            node_type="scan", relation=alias, algorithm=algo,
            est_rows=atom.rows, est_width=atom.width, cost=total, startup=startup,
            output_relations=[alias], candidates=candidates,
            sorted_cols=set(atom.sorted_cols) if algo == "indexscan" else set(),
        )
        node.extra["filters"] = list(atom.filters)
        node.extra["base_rows"] = atom.base_rows
        node.extra["filter_selectivity"] = atom.filter_selectivity
        self._scans[alias] = node
        return node

    # ------------------------------------------------------------------
    def join(self, left: PlanNode, right: PlanNode, connectors: Sequence[Predicate],
             rows: float) -> PlanNode:
        """Inner join.  ``rows`` is the cardinality of the union subset."""
        key = self._join_key(connectors)
        algo, startup, total, candidates = choose_join_algorithm(
            left, right, self.config, key)
        node = PlanNode(
            node_type="join", left=left, right=right, algorithm=algo,
            predicates=list(connectors), est_rows=rows, cost=total, startup=startup,
            candidates=candidates,
            est_width=max(left.est_width, right.est_width),
            output_relations=list(left.output_relations) + list(right.output_relations),
        )
        node.extra["cartesian"] = not connectors
        if algo == "merge" and key:
            node.sorted_cols = {key}
        return node

    def outer(self, left: PlanNode, right: PlanNode, oj, on_preds: Sequence[Predicate],
              rows: float) -> PlanNode:
        key = self._join_key(on_preds)
        algo, total, candidates = outer_join_cost(left, right, self.config,
                                                   oj.kind, key)
        node = PlanNode(
            node_type="outer", left=left, right=right, algorithm=algo,
            predicates=list(oj.predicates), est_rows=rows, cost=total,
            startup=left.cost + right.cost, candidates=candidates,
            est_width=max(left.est_width, right.est_width),
            output_relations=list(left.output_relations) + list(right.output_relations),
        )
        node.extra["kind"] = oj.kind
        node.extra["outer_id"] = oj.id
        return node

    def semi(self, left: PlanNode, inner: PlanNode, unit: SemiUnit,
             rows: float) -> PlanNode:
        startup, total, info = semi_join_cost(left, inner, self.config,
                                              anti=(unit.kind == "anti"))
        node = PlanNode(
            node_type=unit.kind, left=left, right=inner,
            algorithm=info["algorithm"], est_rows=rows, cost=total,
            startup=startup, candidates=info["candidates"],
            predicates=list(unit.correlation),
            est_width=left.est_width,
            output_relations=list(left.output_relations),
        )
        node.extra["unit_id"] = unit.id
        node.extra["outer_refs"] = sorted(unit.outer_refs)
        node.extra["semi_selectivity"] = self.est.unit_sel.get(unit.id, 0.5)
        node.extra["semi_source"] = (unit.correlation[0].source
                                     if unit.correlation else None)
        return node

    # ------------------------------------------------------------------
    def inner_subplan(self, unit: SemiUnit) -> PlanNode:
        """Cheapest-of-a-few plan for the relations inside a semi unit."""
        rels = list(unit.inner_relations)
        if len(rels) == 1:
            return self.scan(rels[0])
        # greedy by size, using only inner predicates
        ordered = sorted(rels, key=lambda r: (self.est.atoms[r].rows, r))
        node = self.scan(ordered[0])
        have = {ordered[0]}
        for r in ordered[1:]:
            conn = [p for p in unit.inner_joins
                    if (p.left_table in have and p.right_table == r) or
                       (p.right_table in have and p.left_table == r)]
            have.add(r)
            rows = 1.0
            for x in have:
                rows *= self.est.atoms[x].rows
            for p in unit.inner_joins:
                if p.left_table in have and p.right_table in have:
                    rows *= p.selectivity or 1.0
            node = self.join(node, self.scan(r), conn, rows)
        return node

    @staticmethod
    def _join_key(preds: Sequence[Predicate]) -> Optional[str]:
        for p in preds:
            if p.kind == "eq" and p.col_left and p.col_right:
                return f"{p.left_table}.{p.col_left}"
        return None
