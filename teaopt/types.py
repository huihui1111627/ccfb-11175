"""Core data structures used across the optimizer.

The optimizer works on an abstract *join query*:

* base relations (table aliases) with statistics,
* inner-join predicates (free edges of the join graph),
* outer joins (LEFT/FULL) that impose a legal evaluation order,
* semi / anti join units produced by unnesting correlated subqueries.

Plans are left-deep trees (System-R style).  Every join step has one relation
or one semi-join unit on the right-hand side.  That is the plan space for which
the dynamic program is provably equivalent to brute-forcing every legal join
order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class Histogram:
    """Equi-width histogram.

    ``buckets`` holds ``(low, high, count, ndv)`` tuples; buckets are
    half-open except the last one.  ``ndv`` is the number of distinct values
    inside the bucket.
    """

    low: float
    high: float
    buckets: List[Tuple[float, float, int, int]]
    null_count: int = 0
    total_count: int = 0

    def __post_init__(self) -> None:
        if self.total_count == 0:
            self.total_count = sum(b[2] for b in self.buckets) + self.null_count

    def estimate_eq(self, value: float) -> float:
        """Estimated number of rows whose value equals ``value``."""
        buckets = self.buckets
        for idx, (low, high, count, ndv) in enumerate(buckets):
            last = idx == len(buckets) - 1
            if not ((low <= value < high) or (last and low <= value <= high)):
                continue
            if ndv <= 0 or count <= 0:
                return 0.0
            width = high - low
            if width <= 0:
                return float(count)
            distinct = min(float(ndv), max(width, 1.0))
            return count / distinct
        return 0.0

    def estimate_range(self, low_v: Optional[float], high_v: Optional[float]) -> float:
        """Estimated number of rows with ``low_v <= value <= high_v``."""
        total = 0.0
        n = len(self.buckets)
        for idx, (low, high, count, _ndv) in enumerate(self.buckets):
            if high_v is not None and low > high_v:
                continue
            if low_v is not None and high < low_v:
                continue
            width = high - low
            if width <= 0:
                total += count
                continue
            lo = max(low, low_v if low_v is not None else low)
            hi = min(high, high_v if high_v is not None else high)
            frac = (hi - lo) / width
            if idx == n - 1:
                frac = max(frac, 1.0 / width)
            total += count * max(0.0, min(1.0, frac))
        return total

    def estimate_lt(self, value: float) -> float:
        return self.estimate_range(None, value)

    def estimate_gt(self, value: float) -> float:
        return max(0.0, self.total_count - self.null_count - self.estimate_range(None, value))

    def join_estimate(self, other: "Histogram") -> float:
        """Estimated number of matching row pairs between two histograms.

        Rows are assumed uniform within a bucket.  For every overlapping pair
        of buckets we estimate the number of common values (overlap width) and
        the average frequency on each side, then multiply.
        """
        total = 0.0
        for l_low, l_high, l_count, l_ndv in self.buckets:
            if l_count <= 0 or l_ndv <= 0:
                continue
            l_width = max(l_high - l_low, 1.0)
            l_freq = l_count / l_ndv
            for r_low, r_high, r_count, r_ndv in other.buckets:
                if r_count <= 0 or r_ndv <= 0:
                    continue
                lo = max(l_low, r_low)
                hi = min(l_high, r_high)
                if hi < lo:
                    continue
                r_width = max(r_high - r_low, 1.0)
                r_freq = r_count / r_ndv
                # number of distinct values shared inside the overlap
                common = min(l_ndv * (hi - lo) / l_width,
                             r_ndv * (hi - lo) / r_width)
                total += l_freq * r_freq * max(1.0, common)
        return total


@dataclass
class ColumnStats:
    name: str
    table: str
    row_count: int
    ndv: int
    null_count: int
    min_value: Any = None
    max_value: Any = None
    is_numeric: bool = False
    histogram: Optional[Histogram] = None
    #: top-k (value, row_count) pairs for skewed columns.
    mcv: List[Tuple[Any, int]] = field(default_factory=list)
    mcv_mass: float = 0.0
    avg_width: int = 8

    @property
    def null_fraction(self) -> float:
        return self.null_count / self.row_count if self.row_count else 0.0

    # -- single-column selectivities: (sel, source, explanation) ----------
    def eq_selectivity(self, value: Any) -> Tuple[float, str, str]:
        if self.row_count <= 0:
            return 0.0, "default", "空表"
        if self.is_numeric and isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.histogram is not None:
                rows = self.histogram.estimate_eq(float(value))
                return min(1.0, max(0.0, rows / self.row_count)), "histogram", \
                    f"直方图桶内 1/桶ndv ≈ {rows:.1f} 行"
        for v, cnt in self.mcv:
            if v == value:
                return min(1.0, cnt / self.row_count), "mcv", f"高频值列表命中，真实频次 {cnt} 行"
        if self.mcv:
            rest_mass = max(0.1 / max(self.ndv, 1), 1.0 - self.mcv_mass - self.null_fraction)
            rest_ndv = max(1, self.ndv - len(self.mcv))
            return min(1.0, rest_mass / rest_ndv), "mcv", \
                f"非高频值按剩余 {rest_ndv} 个不同值均分"
        return 1.0 / max(self.ndv, 1), "ndv", f"无直方图，使用 1/ndv = 1/{max(self.ndv, 1)}"

    def range_selectivity(self, low: Any, high: Any) -> Tuple[float, str, str]:
        if self.row_count <= 0:
            return 0.0, "default", "空表"
        if self.histogram is not None and (low is None or isinstance(low, (int, float))) \
                and (high is None or isinstance(high, (int, float))):
            rows = self.histogram.estimate_range(
                float(low) if low is not None else None,
                float(high) if high is not None else None)
            return min(1.0, max(0.0, rows / self.row_count)), "histogram", \
                f"直方图区间比例 ≈ {rows:.1f} 行"
        if isinstance(self.min_value, (int, float)) and isinstance(self.max_value, (int, float)):
            span = max(float(self.max_value) - float(self.min_value), 1e-9)
            lo = float(low) if isinstance(low, (int, float)) else float(self.min_value)
            hi = float(high) if isinstance(high, (int, float)) else float(self.max_value)
            return min(1.0, max(0.0, (hi - lo) / span)), "ndv", \
                "无直方图，按 min/max 均匀分布估算"
        return 0.3, "default", "无区间统计，使用默认选择率 0.3"

    def lt_selectivity(self, value: Any) -> Tuple[float, str, str]:
        return self.range_selectivity(None, value)

    def gt_selectivity(self, value: Any) -> Tuple[float, str, str]:
        return self.range_selectivity(value, None)

    def in_selectivity(self, values: List[Any]) -> Tuple[float, str, str]:
        sels = [self.eq_selectivity(v) for v in values]
        total = min(1.0, sum(s for s, _, _ in sels))
        sources = {src for _, src, _ in sels}
        src = "histogram" if "histogram" in sources else (
            "mcv" if "mcv" in sources else "ndv")
        return total, src, f"IN 列表 {len(values)} 个值，选择率求和后截断"


@dataclass
class MultiColStats:
    """Multi-column statistics that correct the independence assumption."""

    table: str
    columns: Tuple[str, ...]
    ndv: int
    row_count: int
    #: most common (value tuple, count) pairs
    mcv: List[Tuple[Tuple[Any, ...], int]] = field(default_factory=list)
    mcv_mass: float = 0.0


@dataclass
class TableStats:
    name: str
    row_count: int
    page_count: int = 1
    avg_row_bytes: int = 48
    columns: Dict[str, ColumnStats] = field(default_factory=dict)
    multicol: Dict[Tuple[str, ...], MultiColStats] = field(default_factory=dict)
    indexes: List[List[str]] = field(default_factory=list)

    def column(self, name: str) -> Optional[ColumnStats]:
        return self.columns.get(name)

    def multicol_for(self, columns: Set[str]) -> Optional[MultiColStats]:
        key = tuple(sorted(columns))
        if key in self.multicol:
            return self.multicol[key]
        best = None
        for cols, mc in self.multicol.items():
            if set(cols) <= columns and len(cols) >= 2:
                if best is None or len(cols) > len(best.columns):
                    best = mc
        return best

    def has_index_prefix(self, column: str) -> bool:
        return any(cols and cols[0] == column for cols in self.indexes)

    def index_height(self) -> int:
        import math
        return max(1, int(math.log2(max(self.page_count, 2))) + 1)


# ---------------------------------------------------------------------------
# Query representation
# ---------------------------------------------------------------------------


@dataclass
class Predicate:
    kind: str                       # eq | range | in | like | null | notnull | expr
    text: str
    left_table: Optional[str] = None
    right_table: Optional[str] = None
    col_left: Optional[str] = None
    col_right: Optional[str] = None
    op: str = "="
    value: Any = None
    value2: Any = None
    values: Optional[List[Any]] = None
    selectivity: Optional[float] = None
    source: Optional[str] = None    # histogram | mcv | ndv | multicol | default
    note: str = ""
    origin: str = "where"           # where | on-inner | on-outer

    def is_join(self) -> bool:
        return self.left_table is not None and self.right_table is not None

    def tables(self) -> Set[str]:
        out: Set[str] = set()
        if self.left_table:
            out.add(self.left_table)
        if self.right_table:
            out.add(self.right_table)
        return out


@dataclass
class OuterJoin:
    """A LEFT / FULL outer join (RIGHT is normalised to LEFT at parse time).

    ``preserved`` is the side whose rows always survive; ``nullable`` is the
    side that can be NULL-extended.  For a FULL join the parser records the
    prefix as preserved and the new relation as nullable; both sides count as
    nullable when rows are counted.
    """

    id: int
    kind: str                        # left | full
    preserved: List[str]
    nullable: List[str]
    predicates: List[Predicate] = field(default_factory=list)
    degenerate: bool = False         # nullable side filtered in WHERE -> inner
    note: str = ""

    def describe(self) -> str:
        return f"{self.kind.upper()} JOIN {{{', '.join(self.nullable)}}}"


@dataclass
class SemiUnit:
    """A correlated subquery rewritten to a semi / anti join."""

    id: int
    kind: str                        # semi | anti
    outer_refs: Set[str]
    inner_relations: List[str]
    inner_joins: List[Predicate] = field(default_factory=list)
    correlation: List[Predicate] = field(default_factory=list)
    inner_filters: Dict[str, List[Predicate]] = field(default_factory=dict)
    inner_alias_tables: Dict[str, str] = field(default_factory=dict)
    original_sql: str = ""
    #: the original WHERE conjunct, e.g. "EXISTS (SELECT 1 FROM ...)".
    #: Used verbatim when the plan is executed on SQLite.
    predicate_sql: str = ""
    #: "exists" | "in" — how the subquery was written
    mode: str = "exists"
    #: parsed conjuncts of the subquery WHERE, used to regenerate its SQL
    inner_conjuncts: List[Any] = field(default_factory=list)
    note: str = ""
    inner_alias: str = ""

    @property
    def symbol(self) -> str:
        return "⋉" if self.kind == "semi" else "▷"

    def label(self) -> str:
        rel = "+".join(self.inner_relations)
        return f"{self.symbol} {rel}"

    def output_relations(self) -> List[str]:
        return []


@dataclass
class Query:
    sql: str
    #: main-query alias -> base table name
    tables: Dict[str, str] = field(default_factory=dict)
    #: original FROM order (after RIGHT-join normalisation)
    from_order: List[str] = field(default_factory=list)
    inner_joins: List[Predicate] = field(default_factory=list)
    local_filters: Dict[str, List[Predicate]] = field(default_factory=dict)
    outer_joins: List[OuterJoin] = field(default_factory=list)
    semi_units: List[SemiUnit] = field(default_factory=list)
    #: alias -> set(alias) adjacency (inner + outer + semi edges)
    graph: Dict[str, Set[str]] = field(default_factory=dict)
    #: reorderable stages; stages are evaluated in order
    stages: List["Stage"] = field(default_factory=list)
    #: semi units that are attached inside a stage (by relation availability)
    floating_units: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    original_sql: str = ""
    folded: Dict[str, str] = field(default_factory=dict)
    select_list: str = "*"
    group_by: Optional[str] = None
    order_by: Optional[str] = None
    limit: Optional[str] = None

    @property
    def relations(self) -> List[str]:
        return list(self.tables.keys())

    def add_edge(self, a: str, b: str) -> None:
        self.graph.setdefault(a, set()).add(b)
        self.graph.setdefault(b, set()).add(a)

    def edge_predicates(self) -> List[Tuple[str, str]]:
        seen = set()
        out = []
        for p in self.inner_joins:
            key = tuple(sorted((p.left_table, p.right_table)))
            if key not in seen:
                seen.add(key)
                out.append((p.left_table, p.right_table))
        for oj in self.outer_joins:
            for p in oj.predicates:
                if p.is_join():
                    key = tuple(sorted((p.left_table, p.right_table)))
                    if key not in seen:
                        seen.add(key)
                        out.append((p.left_table, p.right_table))
        for unit in self.semi_units:
            for r in sorted(unit.outer_refs):
                key = tuple(sorted((r, unit.inner_alias)))
                if key not in seen:
                    seen.add(key)
                    out.append((r, unit.inner_alias))
        return out


@dataclass
class Stage:
    """A block of atoms that may be freely reordered among themselves.

    ``outer`` is the outer join that attaches this stage to the accumulated
    prefix (evaluated in stage order), if any.
    """

    relations: List[str]
    semi_units: List[str] = field(default_factory=list)
    outer: Optional[OuterJoin] = None

    def describe(self) -> str:
        r = "{" + ", ".join(self.relations + self.semi_units) + "}"
        return f"{'outer' if self.outer else 'inner'} stage {r}"


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


@dataclass
class PlanNode:
    node_type: str                  # scan | join | outer | semi | anti
    relation: Optional[str] = None
    left: Optional["PlanNode"] = None
    right: Optional["PlanNode"] = None
    algorithm: Optional[str] = None  # nl | hash | merge
    predicates: List[Predicate] = field(default_factory=list)
    est_rows: float = 0.0
    est_width: float = 12.0
    cost: float = 0.0
    startup: float = 0.0
    output_relations: List[str] = field(default_factory=list)
    #: for merge join / display: "alias.column" keys the output is sorted on
    sorted_cols: Set[str] = field(default_factory=set)
    #: per-algorithm candidate costs at this node
    candidates: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def children(self) -> List["PlanNode"]:
        return [c for c in (self.left, self.right) if c is not None]

    def walk(self):
        yield self
        if self.left is not None:
            yield from self.left.walk()
        if self.right is not None:
            yield from self.right.walk()

    def relations(self) -> Set[str]:
        out = set(self.output_relations)
        if self.relation is not None:
            out.add(self.relation)
        return out

    def atom_order(self) -> List[str]:
        """Left-to-right evaluation order of scan atoms."""
        if self.node_type == "scan":
            return [self.relation] if self.relation else []
        if self.node_type in ("semi", "anti"):
            return self.left.atom_order() if self.left else []
        return (self.left.atom_order() if self.left else []) + \
               (self.right.atom_order() if self.right else [])

    def label(self) -> str:
        if self.node_type == "scan":
            return f"Scan {self.relation}"
        if self.node_type == "join":
            return f"{self.algorithm.upper()} join"
        if self.node_type == "outer":
            kind = self.extra.get("kind", "left")
            return f"{kind.upper()} {self.algorithm.upper()} join"
        if self.node_type == "semi":
            return f"Semi {self.algorithm.upper()}"
        if self.node_type == "anti":
            return f"Anti {self.algorithm.upper()}"
        return self.node_type


@dataclass
class PlanAlternative:
    order: List[str]
    cost: float
    rows: float
    algorithm_summary: str


@dataclass
class OrderResult:
    plan: PlanNode
    total_cost: float
    est_rows: float
    algorithm: str                  # exhaustive | greedy | brute
    order: List[str]
    subproblems: int = 0
    transitions: int = 0
    alternatives: List[PlanAlternative] = field(default_factory=list)
    greedy_plan: Optional[PlanNode] = None
    greedy_cost: Optional[float] = None
    greedy_order: Optional[List[str]] = None
    notes: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost(self) -> float:
        """Alias for :attr:`total_cost` (convenience for callers)."""
        return self.total_cost


@dataclass
class OptimizerConfig:
    #: above this many atoms the optimizer falls back to greedy
    exhaustive_threshold: int = 10
    #: when the greedy fallback triggers, still compute the exact optimum for
    #: comparison while the atom count is at most this (0 disables it)
    compare_exact_limit: int = 9
    algorithm: str = "auto"          # auto | exhaustive | greedy | brute
    #: cost model parameters
    seq_page_cost: float = 1.0
    random_page_cost: float = 4.0
    cpu_tuple_cost: float = 0.01
    cpu_operator_cost: float = 0.0025
    cpu_index_tuple_cost: float = 0.005
    #: estimation knobs
    use_histograms: bool = True
    use_multicolumn: bool = True
    default_filter_selectivity: float = 0.3
    default_join_selectivity: float = 0.1
    #: prune DP to connected subsets (exact for connected join graphs)
    connected_pruning: bool = True
    max_states: int = 200_000
    max_alternatives: int = 6
    #: include cartesian-product plans in the enumeration
    allow_cartesian: bool = False
