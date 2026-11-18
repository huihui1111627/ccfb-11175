"""Join-order search: exhaustive DP, brute-force reference, greedy fallback.

All strategies build plans with :class:`teaopt.plan.PlanFactory`, so their
costs are directly comparable — which is what makes the acceptance test
("DP must equal brute force on small queries") meaningful.

Exhaustive (System-R style subset DP)
-------------------------------------
``DP[S]`` stores the cheapest left-deep plan for subset ``S``, filled over
increasing subset size.  Transitions consider every relation ``r ∈ S`` with
``S1 = S \\ {r}`` already solved and ``r`` connected to ``S1``.  In addition
*all* inner-join predicates between ``S1`` and ``r`` are applied at that step,
so the estimate of ``S`` is a function of the subset only.  That is the
standard condition under which subset DP is exact with respect to trying every
join order.

Stages (outer-join legality, requirement 5)
-------------------------------------------
The parser splits relations into stages that must be evaluated in order.  The
DP enumerates each stage independently and concatenates them; an outer join's
nullable stage therefore can never move below its preserved side, and a
relation that appears after an outer join in the query can never move before
it.  Inner-join predicates that cross a stage boundary become the join
condition used when the later stage is attached.

Semi / anti units (requirement 6)
---------------------------------
Units float: a unit may be attached as soon as all its outer references are in
the current subset.  Inside the DP this is applied after every transition, so
the optimizer chooses *where* to place the subquery.

Greedy fallback (requirement 2)
-------------------------------
Used when the atom count exceeds the configurable threshold.  Repeatedly joins
the candidate with the cheapest incremental cost, then reports how far it is
from the exhaustive optimum when the latter is still computable.

Brute force
-----------
Enumerates every legal order of a single-stage query and costs each one; used
by the test-suite.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .estimate import Estimator
from .plan import PlanFactory
from .types import (OptimizerConfig, OrderResult, PlanAlternative, PlanNode,
                    Query, SemiUnit, Stage, TableStats)


class _StageResult:
    def __init__(self, plan: PlanNode, subproblems: int, transitions: int,
                 alternatives: List[PlanAlternative]):
        self.plan = plan
        self.subproblems = subproblems
        self.transitions = transitions
        self.alternatives = alternatives


class Optimizer:
    def __init__(self, query: Query, stats: Dict[str, TableStats],
                 config: Optional[OptimizerConfig] = None):
        self.query = query
        self.stats = stats
        self.config = config or OptimizerConfig()
        self.est = Estimator(query, stats, self.config)
        self.factory = PlanFactory(query, stats, self.est, self.config)
        self.units = list(query.semi_units)
        self.inner_aliases: Set[str] = set()
        for u in self.units:
            self.inner_aliases |= set(u.inner_relations)
        self.atoms = [a for a in query.tables if a not in self.inner_aliases]

    # ------------------------------------------------------------------
    def optimize(self) -> OrderResult:
        count = len(self.atoms)
        forced = self.config.algorithm
        if forced == "greedy":
            return self._greedy_result(count)
        if forced == "exhaustive":
            return self._exhaustive_result(count, forced=True)
        if count <= self.config.exhaustive_threshold:
            return self._exhaustive_result(count, forced=False)
        greedy = self._greedy_result(count)
        greedy.greedy_plan = greedy.plan
        greedy.greedy_cost = greedy.total_cost
        greedy.greedy_order = greedy.order
        # Keep the greedy plan as the answer (that is what "fall back to
        # greedy" means), but still compute the exact optimum when it is
        # affordable so the UI can show the *price* of the fallback.
        limit = self.config.compare_exact_limit
        if limit and count <= limit:
            try:
                exact = self._exhaustive_result(count, forced=False)
            except Exception:      # pragma: no cover - defensive
                return greedy
            gap = (greedy.total_cost / exact.total_cost - 1.0) * 100 if exact.total_cost > 0 else 0.0
            greedy.notes.append(
                f"贪心代价 {greedy.total_cost:.1f}，精确最优代价 "
                f"{exact.total_cost:.1f}（差距 {gap:.2f}%）；"
                f"精确最优顺序为 {' → '.join(exact.order)}")
            greedy.stats["exact_cost"] = exact.total_cost
            greedy.stats["exact_order"] = list(exact.order)
            greedy.stats["greedy_gap"] = gap / 100.0
        return greedy

    # ------------------------------------------------------------------
    # Exhaustive DP
    # ------------------------------------------------------------------
    def _exhaustive_result(self, count: int, forced: bool) -> OrderResult:
        plan: Optional[PlanNode] = None
        order: List[str] = []
        prefix: List[str] = []
        subproblems = transitions = 0
        alternatives: List[PlanAlternative] = []
        applied: Dict[int, int] = {}
        for stage in self.query.stages:
            result = self._dp_stage(stage, prefix, applied)
            if plan is None:
                plan = result.plan
            else:
                plan = self._attach_stage(plan, result.plan, stage, prefix)
            order.extend(result.plan.atom_order())
            subproblems += result.subproblems
            transitions += result.transitions
            alternatives.extend(result.alternatives)
            prefix = prefix + list(stage.relations)
        assert plan is not None
        # safety net: any unit that never became applicable
        for unit in self.units:
            if unit.id not in applied:
                plan = self._attach_unit(plan, unit)
                applied[unit.id] = len(prefix)
        algo = "exhaustive(forced)" if forced else "exhaustive"
        result = OrderResult(
            plan=plan, total_cost=plan.cost, est_rows=plan.est_rows,
            algorithm=algo, order=order or plan.atom_order(),
            subproblems=subproblems, transitions=transitions,
            alternatives=alternatives[: self.config.max_alternatives],
            stats={"stages": [s.describe() for s in self.query.stages],
                   "atoms": len(self.atoms)},
        )
        if count > self.config.exhaustive_threshold and not forced:
            result.notes.append(
                f"原子数 {count} 超过阈值 {self.config.exhaustive_threshold}，已退回贪心")
        result.notes.extend(self._rewrite_notes())
        return result

    def _attach_stage(self, plan: PlanNode, stage_plan: PlanNode,
                      stage: Stage, prefix: Sequence[str]) -> PlanNode:
        if stage.outer is not None:
            oj = stage.outer
            on_preds = [p for p in oj.predicates if p.is_join()]
            rows = self._outer_rows(prefix, stage.relations, oj, stage_plan.est_rows,
                                    plan.est_rows)
            return self.factory.outer(plan, stage_plan, oj, on_preds, rows)
        # inner stage attached after an outer boundary: use the crossing
        # predicates as the join condition
        crossing = self.est.crossing_connectors(set(prefix), set(stage.relations))
        rows = self.est.subset_rows(list(prefix) + list(stage.relations))
        return self.factory.join(plan, stage_plan, crossing, rows)

    def _outer_rows(self, prefix: Sequence[str], stage_rels: Sequence[str],
                    oj, stage_rows: float, prefix_rows: float) -> float:
        if oj.kind == "left":
            cond = 1.0
            for p in oj.predicates:
                if p.is_join():
                    cond *= p.selectivity if p.selectivity is not None else 1.0
            return max(prefix_rows, prefix_rows * stage_rows * cond)
        # FULL: both sides preserved, at least |L| + |R| - matches
        cond = 1.0
        for p in oj.predicates:
            if p.is_join():
                cond *= p.selectivity if p.selectivity is not None else 1.0
        return max(prefix_rows + stage_rows - prefix_rows * stage_rows * cond,
                   prefix_rows * 0.1 + stage_rows * 0.1)

    def _dp_stage(self, stage: Stage, prefix: Sequence[str],
                  applied: Dict[int, int]) -> _StageResult:
        rels = list(stage.relations)
        stage_units = [u for u in self.units if u.id not in applied]
        best: Dict[FrozenSet[str], PlanNode] = {}
        best_units: Dict[FrozenSet[str], Dict[int, int]] = {}
        subproblems = 0
        transitions = 0

        for r in rels:
            fs = frozenset([r])
            node = self.factory.scan(r)
            units = dict(applied)
            node, units = self._attach_units(node, units, stage_units,
                                             set(prefix) | set(fs))
            best[fs] = node
            best_units[fs] = units

        for size in range(2, len(rels) + 1):
            for combo in itertools.combinations(rels, size):
                S = frozenset(combo)
                if not self.config.allow_cartesian and not self._connected(S):
                    continue
                local_best: Optional[PlanNode] = None
                local_units: Dict[int, int] = {}
                for r in combo:
                    S1 = S - {r}
                    if S1 not in best:
                        continue
                    if not self._connected_to(S1, r):
                        continue
                    transitions += 1
                    left = best[S1]
                    right = self.factory.scan(r)
                    conn = self.est.connectors(set(S1), r)
                    rows = self.est.subset_rows(S)
                    node = self.factory.join(left, right, conn, rows)
                    units = dict(best_units[S1])
                    node, units = self._attach_units(node, units, stage_units,
                                                     set(prefix) | set(S))
                    if local_best is None or node.cost < local_best.cost:
                        local_best = node
                        local_units = units
                if local_best is not None:
                    best[S] = local_best
                    best_units[S] = local_units
                    subproblems += 1

        full = frozenset(rels)
        if full in best:
            node = best[full]
            used = best_units[full]
        else:
            # disconnected join graph: join in query order (cartesian allowed)
            node = self.factory.scan(rels[0])
            have = {rels[0]}
            for r in rels[1:]:
                conn = self.est.connectors(have, r)
                have.add(r)
                node = self.factory.join(node, self.factory.scan(r), conn,
                                         self.est.subset_rows(have))
            used = dict(applied)
        node, used = self._attach_units(node, used, stage_units,
                                        set(prefix) | set(rels), final=True)
        for uid in used:
            applied.setdefault(uid, len(prefix) + len(rels))
        alternatives = []
        for fs, nd in best.items():
            if len(fs) == len(full):
                continue
            alternatives.append(PlanAlternative(
                order=nd.atom_order(), cost=nd.cost, rows=nd.est_rows,
                algorithm_summary=nd.algorithm or ""))
        alternatives.sort(key=lambda a: a.cost)
        return _StageResult(node, subproblems, transitions,
                            alternatives[: self.config.max_alternatives])

    def _attach_units(self, node: PlanNode, used: Dict[int, int],
                      units: Sequence[SemiUnit], available: Set[str],
                      final: bool = False) -> Tuple[PlanNode, Dict[int, int]]:
        for unit in units:
            if unit.id in used:
                continue
            if not set(unit.outer_refs) <= available:
                continue
            if final and not set(unit.outer_refs) <= available:
                continue
            node = self._attach_unit(node, unit)
            used[unit.id] = len(available)
        return node, used

    def _attach_unit(self, node: PlanNode, unit: SemiUnit) -> PlanNode:
        inner = self.factory.inner_subplan(unit)
        sel = self.est.unit_sel.get(unit.id, 0.5)
        rows = node.est_rows * sel if unit.kind == "semi" else \
            max(0.0, node.est_rows * (1.0 - sel))
        return self.factory.semi(node, inner, unit, rows)

    def _connected(self, subset: FrozenSet[str]) -> bool:
        if len(subset) <= 1:
            return True
        adj: Dict[str, Set[str]] = {r: set() for r in subset}
        for key in self.est.edge_preds:
            if len(key) == 2 and key <= subset:
                a, b = tuple(key)
                adj[a].add(b)
                adj[b].add(a)
        start = next(iter(subset))
        seen = {start}
        stack = [start]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen == set(subset)

    def _connected_to(self, subset: FrozenSet[str], r: str) -> bool:
        for key in self.est.edge_preds:
            if len(key) == 2 and r in key and (key - {r}) <= subset:
                return True
        return False

    # ------------------------------------------------------------------
    # Greedy
    # ------------------------------------------------------------------
    def _greedy_result(self, count: int) -> OrderResult:
        plan, order, applied = self._greedy_plan()
        result = OrderResult(
            plan=plan, total_cost=plan.cost, est_rows=plan.est_rows,
            algorithm="greedy", order=order,
            subproblems=len(order), transitions=len(order),
            notes=[f"原子数 {count} > 阈值 {self.config.exhaustive_threshold}，"
                   f"自动退回贪心算法（每步选增量代价最小的连接）"],
            stats={"atoms": count, "threshold": self.config.exhaustive_threshold},
        )
        result.notes.extend(self._rewrite_notes())
        return result

    def _greedy_plan(self) -> Tuple[PlanNode, List[str], Dict[int, int]]:
        rels = sorted(self.atoms, key=lambda r: (self.est.atoms[r].rows, r))
        plan = self.factory.scan(rels[0])
        have = {rels[0]}
        order = [rels[0]]
        remaining = set(rels[1:])
        applied: Dict[int, int] = {}
        plan, applied = self._attach_units(plan, applied, self.units, set(have))
        while remaining:
            best = None
            for cand in sorted(remaining):
                conn = self.est.connectors(have, cand)
                if not conn and not self.config.allow_cartesian:
                    continue
                rows = self.est.subset_rows(have | {cand})
                trial = self.factory.join(plan, self.factory.scan(cand), conn, rows)
                trial2, _ = self._attach_units(trial, dict(applied), self.units,
                                               have | {cand})
                key = (trial2.cost, cand)
                if best is None or key < best[0]:
                    best = (key, cand, trial2)
            if best is None:
                cand = sorted(remaining)[0]
                rows = self.est.subset_rows(have | {cand})
                trial = self.factory.join(plan, self.factory.scan(cand), [], rows)
                best = ((trial.cost, cand), cand, trial)
            _, cand, plan = best
            have.add(cand)
            remaining.remove(cand)
            order.append(cand)
            plan, applied = self._attach_units(plan, applied, self.units, set(have))
        for unit in self.units:
            if unit.id not in applied:
                plan = self._attach_unit(plan, unit)
                applied[unit.id] = len(have)
        return plan, order, applied

    def _rewrite_notes(self) -> List[str]:
        notes = []
        for unit in self.units:
            word = "半连接" if unit.kind == "semi" else "反连接"
            notes.append(
                f"关联子查询改写为{word} {unit.label()}，外引用 "
                f"{sorted(unit.outer_refs)}，已参与连接顺序枚举")
        return notes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def plan_for_order(query: Query, stats: Dict[str, TableStats],
                   order: Sequence[str],
                   config: Optional[OptimizerConfig] = None) -> PlanNode:
    """Build the left-deep plan for an explicit relation order.

    Used by the UI to time an alternative order through the same pipeline, so
    the comparison measures the order and not SQLite's planner.
    """
    cfg = config or OptimizerConfig()
    opt = Optimizer(query, stats, cfg)
    factory = opt.factory
    est = opt.est
    node = factory.scan(order[0])
    have = {order[0]}
    for r in order[1:]:
        conn = est.connectors(have, r)
        have.add(r)
        node = factory.join(node, factory.scan(r), conn, est.subset_rows(have))
    return node


def brute_force(query: Query, stats: Dict[str, TableStats],
                config: Optional[OptimizerConfig] = None
                ) -> Tuple[float, List[str], int]:
    """Cost of the cheapest of ALL *legal* permutations.

    "Legal" means the same restriction the DP works under: relations may only
    be appended when they are connected to the current prefix (cartesian
    products excluded unless ``allow_cartesian``), and relations after an
    outer join may not move before it.  The DP submits to exactly the same
    restriction, so the comparison is apples to apples.
    """
    cfg = config or OptimizerConfig()
    opt = Optimizer(query, stats, cfg)
    factory = opt.factory
    est = opt.est
    rels = list(opt.atoms)
    best = math.inf
    best_order: List[str] = []
    considered = 0
    for perm in itertools.permutations(rels):
        if not _legal_order(query, perm, cfg):
            continue
        node = factory.scan(perm[0])
        have = {perm[0]}
        for r in perm[1:]:
            conn = est.connectors(have, r)
            have.add(r)
            node = factory.join(node, factory.scan(r), conn, est.subset_rows(have))
        considered += 1
        if node.cost < best - 1e-9:
            best = node.cost
            best_order = list(perm)
    return best, best_order, considered


def _legal_order(query: Query, order: Sequence[str],
                 config: OptimizerConfig) -> bool:
    """Is ``order`` a legal evaluation order under the outer-join rules?"""
    have: Set[str] = set()
    stage_of = {r: i for i, st in enumerate(query.stages) for r in st.relations}
    last_stage = -1
    for idx, r in enumerate(order):
        stage = stage_of.get(r, 0)
        if stage < last_stage:
            return False
        last_stage = max(last_stage, stage)
        if idx > 0 and not config.allow_cartesian:
            if not any(
                (p.left_table == r and p.right_table in have) or
                (p.right_table == r and p.left_table in have)
                for p in query.inner_joins
            ):
                # a new stage may be attached through an outer join condition
                if not any(
                    (p.left_table == r and p.right_table in have) or
                    (p.right_table == r and p.left_table in have)
                    for oj in query.outer_joins for p in oj.predicates if p.is_join()
                ):
                    return False
        have.add(r)
    return True
