"""Plan visualisation helpers, enumeration statistics and workload shapes.

This module contains the pieces the web UI needs beyond the optimizer itself:

* :func:`plan_to_dict` – serialise a plan tree for the front-end,
* :func:`enumeration_table` – the number of sub-problems / transitions for the
  four canonical join-graph shapes (chain, star, cycle, clique), used to show
  why exhaustive DP explodes and why greedy is needed (requirement 8),
* :func:`shape_of` – classify a query's join graph,
* :func:`compare_estimates` – per-step estimated vs. actual rows (requirement 7).
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .types import PlanNode


# ---------------------------------------------------------------------------
# Plan serialisation
# ---------------------------------------------------------------------------


def plan_to_dict(node: Optional[PlanNode]) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    predicates = []
    for p in node.predicates:
        predicates.append({
            "text": p.text,
            "selectivity": p.selectivity,
            "source": p.source,
            "note": p.note,
            "kind": p.kind,
            "origin": p.origin,
        })
    filters = node.extra.get("filters") or []
    return {
        "id": id(node),
        "type": node.node_type,
        "label": node.label(),
        "relation": node.relation,
        "algorithm": node.algorithm,
        "est_rows": node.est_rows,
        "est_width": node.est_width,
        "cost": node.cost,
        "startup": node.startup,
        "output_relations": node.output_relations,
        "predicates": predicates,
        "candidates": node.candidates,
        "cartesian": bool(node.extra.get("cartesian")),
        "kind": node.extra.get("kind"),
        "base_rows": node.extra.get("base_rows"),
        "filter_selectivity": node.extra.get("filter_selectivity"),
        "filters": [{"text": p.text, "selectivity": p.selectivity,
                     "source": p.source, "note": p.note} for p in filters],
        "unit_id": node.extra.get("unit_id"),
        "outer_refs": node.extra.get("outer_refs"),
        "semi_selectivity": node.extra.get("semi_selectivity"),
        "semi_source": node.extra.get("semi_source"),
        "children": [plan_to_dict(c) for c in (node.left, node.right) if c is not None],
    }


# ---------------------------------------------------------------------------
# Join-graph shapes (requirement 8)
# ---------------------------------------------------------------------------

SHAPES = ("chain", "star", "cycle", "clique")


def canonical_graph(shape: str, n: int) -> List[Tuple[int, int]]:
    """Edges of the canonical instance of ``shape`` with ``n`` vertices."""
    if shape == "chain":
        return [(i, i + 1) for i in range(n - 1)]
    if shape == "star":
        return [(0, i) for i in range(1, n)]
    if shape == "cycle":
        return [(i, (i + 1) % n) for i in range(n)]
    if shape == "clique":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"unknown shape {shape}")


def shape_of(edges: Iterable[Tuple[str, str]], vertices: Iterable[str]) -> str:
    """Classify a join graph; returns one of the four shapes or ``mixed``."""
    verts = list(dict.fromkeys(vertices))
    n = len(verts)
    if n <= 2:
        return "chain"
    deg: Dict[str, int] = {v: 0 for v in verts}
    adj: Dict[str, Set[str]] = {v: set() for v in verts}
    e = 0
    for a, b in edges:
        if a not in adj or b not in adj:
            continue
        if b not in adj[a]:
            adj[a].add(b)
            adj[b].add(a)
            deg[a] += 1
            deg[b] += 1
            e += 1
    degrees = sorted(deg.values())
    if e == n * (n - 1) // 2:
        return "clique"
    # connected?
    seen = {verts[0]}
    stack = [verts[0]]
    while stack:
        cur = stack.pop()
        for nxt in adj[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    if len(seen) != n:
        return "mixed"
    if degrees[-1] == n - 1 and degrees.count(1) == n - 1:
        return "star"
    if e == n and all(d == 2 for d in degrees):
        return "cycle"
    if e == n - 1 and degrees[-1] <= 2:
        return "chain"
    return "mixed"


def count_connected_subsets(n: int, edges: Sequence[Tuple[int, int]]) -> int:
    """Number of non-empty connected subgraphs (DP states)."""
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    count = 0
    for size in range(1, n + 1):
        for subset in itertools.combinations(range(n), size):
            S = set(subset)
            if size == 1:
                count += 1
                continue
            start = subset[0]
            seen = {start}
            stack = [start]
            while stack:
                cur = stack.pop()
                for nxt in adj[cur]:
                    if nxt in S and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            if seen == S:
                count += 1
    return count


def count_transitions(n: int, edges: Sequence[Tuple[int, int]]) -> int:
    """Number of ``S \\ {r} -> S`` transitions the DP evaluates.

    Every connected subset of size >= 2 contributes one transition per member
    ``r`` that has an edge into the rest of the subset.  Note this can exceed
    ``states * n`` only by the singleton bookkeeping; it is always far below
    ``n!`` for n >= 7.
    """
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    total = 0
    for size in range(2, n + 1):
        for subset in itertools.combinations(range(n), size):
            S = set(subset)
            for r in subset:
                if adj[r] & (S - {r}):
                    total += 1
    return total


def count_left_deep_trees(n: int) -> int:
    """Number of left-deep join trees = n! (every permutation)."""
    return math.factorial(n)


_CANONICAL_CACHE: Dict[Tuple[str, int], Dict[str, int]] = {}


def enumeration_table(sizes: Sequence[int] = (3, 4, 5, 6, 7, 8),
                      shapes: Sequence[str] = SHAPES) -> List[Dict[str, Any]]:
    """Sub-problem / transition counts for each graph shape (requirement 8).

    * ``permutations`` = n! = number of left-deep join orders (what brute force
      enumerates),
    * ``dp_states`` = number of connected subsets (what the DP stores),
    * ``dp_transitions`` = number of subset extensions the DP evaluates.
    """
    out: List[Dict[str, Any]] = []
    for shape in shapes:
        for n in sizes:
            key = (shape, n)
            if key not in _CANONICAL_CACHE:
                edges = canonical_graph(shape, n)
                _CANONICAL_CACHE[key] = {
                    "edges": len(edges),
                    "dp_states": count_connected_subsets(n, edges),
                    "dp_transitions": count_transitions(n, edges),
                }
            c = _CANONICAL_CACHE[key]
            out.append({
                "shape": shape,
                "n": n,
                "edges": c["edges"],
                "permutations": count_left_deep_trees(n),
                "dp_states": c["dp_states"],
                "dp_transitions": c["dp_transitions"],
            })
    return out


def shape_summary(sizes: Sequence[int] = (3, 4, 5, 6, 7, 8)) -> Dict[str, Any]:
    """Grouped view used by the UI table."""
    rows = enumeration_table(sizes)
    table: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SHAPES}
    for r in rows:
        table[r["shape"]].append(r)
    return {"shapes": list(SHAPES), "sizes": list(sizes), "table": table}


# ---------------------------------------------------------------------------
# Estimate vs. actual (requirement 7)
# ---------------------------------------------------------------------------


def compare_estimates(plan: PlanNode,
                      actual_rows: Dict[int, float]) -> List[Dict[str, Any]]:
    """Pair each node's estimate with the measured row count.

    ``actual_rows`` maps ``id(plan_node)`` to the number of rows the node
    produced when the plan was executed.
    """
    out: List[Dict[str, Any]] = []
    for node in plan.walk():
        actual = actual_rows.get(id(node))
        est = node.est_rows
        ratio = (actual / est) if (actual is not None and est > 0) else None
        out.append({
            "node": node.label(),
            "type": node.node_type,
            "relations": sorted(node.relations()),
            "est_rows": est,
            "actual_rows": actual,
            "ratio": ratio,
            "error_pct": (abs(ratio - 1.0) * 100.0) if ratio is not None else None,
            "cost": node.cost,
        })
    return out
