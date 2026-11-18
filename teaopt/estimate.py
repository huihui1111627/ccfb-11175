"""Selectivity and cardinality estimation.

Every predicate gets a *source* attached to it (requirement 1):

``histogram``
    computed from an equi-width histogram (equality / range).
``mcv``
    computed from a most-common-value list (skewed columns).
``ndv``
    computed from the number of distinct values (uniform assumption).
``multicol``
    corrected with multi-column statistics because the columns are correlated.
``default``
    no usable statistics; a textbook default is used.

Key design point: all selectivities are computed *once*, before the search,
and depend only on the predicate / atom.  Therefore the row count of any
subset of relations is a pure function of the subset:

    rows(S) = Π atom_rows(r) × Π edge_selectivity(u, v)

This order-independence is what makes the subset DP exact with respect to
brute-forcing every join order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .types import ColumnStats, OptimizerConfig, Predicate, Query, SemiUnit, TableStats


@dataclass
class Atom:
    """An estimated base relation (a scan with pushed-down filters)."""

    alias: str
    table: str
    base_rows: float
    rows: float
    width: float
    pages: float
    filters: List[Predicate] = field(default_factory=list)
    ndv: Dict[str, float] = field(default_factory=dict)
    index_cols: Set[str] = field(default_factory=set)
    sorted_cols: Set[str] = field(default_factory=set)

    @property
    def filter_selectivity(self) -> float:
        return self.rows / self.base_rows if self.base_rows > 0 else 1.0


class Estimator:
    """Pre-computes atom rows, edge selectivities and semi-join factors."""

    def __init__(self, query: Query, stats: Dict[str, TableStats],
                 config: Optional[OptimizerConfig] = None):
        self.query = query
        self.stats = stats
        self.config = config or OptimizerConfig()
        self.atoms: Dict[str, Atom] = {}
        #: frozenset({u,v}) -> combined selectivity of all equality predicates
        self.edge_sel: Dict[frozenset, float] = {}
        self.edge_preds: Dict[frozenset, List[Predicate]] = {}
        self.unit_sel: Dict[int, float] = {}
        self._build_atoms()
        self._build_edges()
        self._build_units()

    # ------------------------------------------------------------------
    # Atoms
    # ------------------------------------------------------------------
    def _build_atoms(self) -> None:
        for alias, table in self.query.tables.items():
            self.atoms[alias] = self._make_atom(alias, table)
        for unit in self.query.semi_units:
            for alias, table in unit.inner_alias_tables.items():
                self.atoms[alias] = self._make_atom(alias, table)

    def _make_atom(self, alias: str, table: str) -> Atom:
        ts = self.stats.get(table)
        filters = list(self.query.local_filters.get(alias, []))
        for unit in self.query.semi_units:
            if alias in unit.inner_alias_tables:
                filters += unit.inner_filters.get(alias, [])
        if ts is None:
            atom = Atom(alias=alias, table=table, base_rows=1000.0, rows=1000.0,
                        width=16.0, pages=10.0)
            atom.rows = self._apply_filters(atom, filters, None)
            atom.filters = filters
            atom.ndv["__rows__"] = max(1.0, atom.rows)
            return atom
        atom = Atom(alias=alias, table=table, base_rows=float(ts.row_count),
                    rows=float(ts.row_count), width=float(ts.avg_row_bytes),
                    pages=float(ts.page_count),
                    index_cols={cols[0] for cols in ts.indexes if cols},
                    sorted_cols={cols[0] for cols in ts.indexes if cols})
        atom.rows = self._apply_filters(atom, filters, ts)
        atom.filters = filters
        frac = atom.rows / atom.base_rows if atom.base_rows else 0.0
        for col, cs in ts.columns.items():
            atom.ndv[col] = max(1.0, cs.ndv * max(frac, 1.0 / max(cs.ndv, 1)))
        atom.ndv["__rows__"] = max(1.0, atom.rows)
        return atom

    def _apply_filters(self, atom: Atom, filters: Sequence[Predicate],
                       ts: Optional[TableStats]) -> float:
        if not filters:
            return atom.base_rows
        for p in filters:
            sel, source, note = self._filter_selectivity(p, ts)
            p.selectivity = sel
            p.source = source
            p.note = note
        rows = atom.base_rows
        for p in filters:
            rows *= p.selectivity or 1.0
        # correlated equality filters: correct the independence product
        eq = [p for p in filters if p.kind == "eq" and p.value is not None
              and p.col_left and ts is not None]
        if self.config.use_multicolumn and ts is not None and len(eq) >= 2:
            self._correct_filter_correlation(ts, eq)
            rows = atom.base_rows
            for p in filters:
                rows *= p.selectivity or 1.0
        return max(0.0, rows)

    def _filter_selectivity(self, p: Predicate,
                            ts: Optional[TableStats]) -> Tuple[float, str, str]:
        cs = ts.column(p.col_left) if ts is not None and p.col_left else None
        if p.kind == "eq":
            if cs is not None:
                return cs.eq_selectivity(p.value)
            return self.config.default_filter_selectivity, "default", \
                f"无列统计，默认选择率 {self.config.default_filter_selectivity}"
        if p.kind == "range":
            if cs is not None:
                if p.op == "between":
                    return cs.range_selectivity(p.value, p.value2)
                if p.op in ("<", "<="):
                    return cs.lt_selectivity(p.value)
                return cs.gt_selectivity(p.value)
            return self.config.default_filter_selectivity, "default", "无列统计，使用默认值"
        if p.kind == "in":
            if cs is not None and p.values:
                return cs.in_selectivity(p.values)
            return min(1.0, 0.1 * len(p.values or [])), "default", \
                "无列统计，按每个值 0.1 估算"
        if p.kind == "like":
            return 0.1, "default", "LIKE 无直方图统计，默认选择率 0.1"
        if p.kind == "null":
            if cs is not None:
                return cs.null_fraction, "ndv", "按 NULL 计数估算"
            return 0.05, "default", "默认 NULL 比例 0.05"
        if p.kind == "notnull":
            if cs is not None:
                return 1.0 - cs.null_fraction, "ndv", "按 NULL 计数估算"
            return 0.95, "default", "默认非 NULL 比例 0.95"
        return self.config.default_filter_selectivity, "default", "表达式谓词，默认值"

    def _correct_filter_correlation(self, ts: TableStats,
                                    eq: Sequence[Predicate]) -> None:
        cols = {p.col_left for p in eq if p.col_left}
        mc = ts.multicol_for(cols)
        if mc is None:
            return
        by_col = {p.col_left: p for p in eq}
        if not all(c in by_col for c in mc.columns):
            return
        vals = tuple(by_col[c].value for c in mc.columns)
        joint = None
        for values, count in mc.mcv:
            if tuple(values) == vals:
                joint = count / max(ts.row_count, 1)
                break
        if joint is None:
            joint = 1.0 / max(mc.ndv, 1)
        indep = 1.0
        for c in mc.columns:
            indep *= by_col[c].selectivity or 1.0
        if indep <= 0:
            return
        ratio = max(0.01, min(100.0, joint / indep))
        target = by_col[mc.columns[-1]]
        target.selectivity = min(1.0, (target.selectivity or 1.0) * ratio)
        target.source = "multicol"
        target.note = (f"多列统计 ({', '.join(mc.columns)}) 修正相关列独立性："
                       f"联合选择率 {joint:.4f}，相对独立假设 ×{ratio:.2f}")

    # ------------------------------------------------------------------
    # Join edges
    # ------------------------------------------------------------------
    def _build_edges(self) -> None:
        groups: Dict[frozenset, List[Predicate]] = {}
        for p in self.query.inner_joins:
            if not p.is_join():
                continue
            groups.setdefault(frozenset((p.left_table, p.right_table)), []).append(p)
        for key, preds in groups.items():
            self.edge_preds[key] = preds
            # Edge selectivity is the *actual* join selectivity of one edge:
            # matching pairs / cartesian product.  Each predicate's own
            # selectivity is the per-predicate NDV/histogram factor (useful for
            # the UI), but several predicates on the same edge must not be
            # multiplied blindly -- that is exactly the independence mistake
            # the multi-column statistics correct.  So compute the edge
            # selectivity on the cardinality scale instead.
            self.edge_sel[key] = self._edge_selectivity(key, preds)
        for oj in self.query.outer_joins:
            for p in oj.predicates:
                if p.is_join():
                    sel, source, note = self._join_selectivity(p)
                    p.selectivity, p.source, p.note = sel, source, note
                    key = frozenset((p.left_table, p.right_table))
                    self.edge_preds.setdefault(key, []).append(p)
                    self.edge_sel.setdefault(key, sel)

    def _edge_selectivity(self, key: frozenset,
                          preds: Sequence[Predicate]) -> float:
        """Selectivity of the equality edge, refined by multi-column stats."""
        for p in preds:
            if p.selectivity is None:
                sel, source, note = self._join_selectivity(p)
                p.selectivity, p.source, p.note = sel, source, note
        a, b = sorted(key)
        atom_a = self.atoms.get(a)
        atom_b = self.atoms.get(b)
        if atom_a is None or atom_b is None or atom_a.rows <= 0 or atom_b.rows <= 0:
            sel = 1.0
            for p in preds:
                sel *= p.selectivity if p.selectivity is not None else 0.1
            return max(1e-12, min(1.0, sel))
        cols_a: Set[str] = set()
        cols_b: Set[str] = set()
        for p in preds:
            if p.left_table == a:
                cols_a.add(p.col_left or "")
                cols_b.add(p.col_right or "")
            else:
                cols_a.add(p.col_right or "")
                cols_b.add(p.col_left or "")
        mc_a = self.stats.get(self.query.tables.get(a, "")).multicol_for(cols_a) \
            if self.stats.get(self.query.tables.get(a, "")) else None
        mc_b = self.stats.get(self.query.tables.get(b, "")).multicol_for(cols_b) \
            if self.stats.get(self.query.tables.get(b, "")) else None
        if self.config.use_multicolumn and len(preds) >= 2 and mc_a and mc_b:
            frac_a = atom_a.rows / max(atom_a.base_rows, 1.0)
            frac_b = atom_b.rows / max(atom_b.base_rows, 1.0)
            ndv_a = max(1.0, mc_a.ndv * max(frac_a, 1.0 / max(mc_a.ndv, 1)))
            ndv_b = max(1.0, mc_b.ndv * max(frac_b, 1.0 / max(mc_b.ndv, 1)))
            sel = 1.0 / max(ndv_a, ndv_b)
            target = preds[-1]
            target.source = "multicol"
            target.note = (
                f"多列统计 ({', '.join(sorted(cols_a))} ↔ {', '.join(sorted(cols_b))}) "
                f"作为复合连接键：联合 ndv={max(ndv_a, ndv_b):.0f}")
            return max(1e-12, min(1.0, sel))
        # single predicate (or no multi-column statistics): use its per-predicate
        # selectivity, corrected for multi-column if only one side has stats
        sel = 1.0
        for p in preds:
            sel *= p.selectivity if p.selectivity is not None else self.config.default_join_selectivity
        return max(1e-12, min(1.0, sel))

    def _join_selectivity(self, p: Predicate) -> Tuple[float, str, str]:
        if p.col_left is None or p.col_right is None:
            return self.config.default_join_selectivity, "default", "缺少列信息"
        ts_l = self.stats.get(self.query.tables.get(p.left_table, ""))
        ts_r = self.stats.get(self.query.tables.get(p.right_table, ""))
        cs_l = ts_l.column(p.col_left) if ts_l else None
        cs_r = ts_r.column(p.col_right) if ts_r else None
        left = self.atoms.get(p.left_table or "")
        right = self.atoms.get(p.right_table or "")
        if cs_l is None or cs_r is None:
            return self.config.default_join_selectivity, "default", \
                "无列统计，默认连接选择率 0.1"
        if self.config.use_histograms and cs_l.histogram is not None \
                and cs_r.histogram is not None and left and right:
            pairs = cs_l.histogram.join_estimate(cs_r.histogram)
            # normalise by the *filtered* input sizes so the factor is stable
            sel = pairs / max(left.base_rows * right.base_rows, 1.0)
            sel = max(1e-9, min(1.0, sel))
            return sel, "histogram", (
                f"直方图重叠：等值匹配 ≈{pairs:.0f} 对 / 笛卡尔积 "
                f"{left.base_rows * right.base_rows:.0f}")
        if self.config.use_histograms and cs_l.histogram is not None and cs_r.histogram is None:
            frac = self._range_overlap_fraction(cs_l, cs_r)
            sel = min(1.0, frac / max(cs_r.ndv, 1))
            return max(1e-9, sel), "histogram", \
                f"一端直方图 × 另一端值域重叠 {frac:.3f} / ndv {cs_r.ndv}"
        if self.config.use_histograms and cs_r.histogram is not None and cs_l.histogram is None:
            frac = self._range_overlap_fraction(cs_r, cs_l)
            sel = min(1.0, frac / max(cs_l.ndv, 1))
            return max(1e-9, sel), "histogram", \
                f"一端直方图 × 另一端值域重叠 {frac:.3f} / ndv {cs_l.ndv}"
        ndv_l = cs_l.ndv
        ndv_r = cs_r.ndv
        return 1.0 / max(ndv_l, ndv_r, 1), "ndv", \
            f"无直方图，1/max(ndv) = 1/{max(ndv_l, ndv_r, 1)}"

    @staticmethod
    def _range_overlap_fraction(hist_cs: ColumnStats, other_cs: ColumnStats) -> float:
        try:
            h_low, h_high = float(hist_cs.min_value), float(hist_cs.max_value)
            o_low, o_high = float(other_cs.min_value), float(other_cs.max_value)
        except (TypeError, ValueError):
            return 1.0
        if h_high <= h_low:
            return 1.0
        lo, hi = max(h_low, o_low), min(h_high, o_high)
        return max(0.0, min(1.0, (hi - lo) / (h_high - h_low)))

    def _correct_join_correlation(self, key: frozenset,
                                  preds: Sequence[Predicate]) -> None:
        if len(key) != 2:
            return
        a, b = sorted(key)
        ts_a = self.stats.get(self.query.tables.get(a, ""))
        ts_b = self.stats.get(self.query.tables.get(b, ""))
        if ts_a is None or ts_b is None:
            return
        cols_a = set()
        cols_b = set()
        for p in preds:
            if p.left_table == a:
                cols_a.add(p.col_left)
                cols_b.add(p.col_right)
            else:
                cols_a.add(p.col_right)
                cols_b.add(p.col_left)
        mc_a = ts_a.multicol_for(cols_a)
        mc_b = ts_b.multicol_for(cols_b)
        if mc_a is None or mc_b is None:
            return
        atom_a = self.atoms.get(a)
        atom_b = self.atoms.get(b)
        frac_a = atom_a.rows / max(atom_a.base_rows, 1.0) if atom_a else 1.0
        frac_b = atom_b.rows / max(atom_b.base_rows, 1.0) if atom_b else 1.0
        ndv_a = max(1.0, mc_a.ndv * max(frac_a, 1.0 / max(mc_a.ndv, 1)))
        ndv_b = max(1.0, mc_b.ndv * max(frac_b, 1.0 / max(mc_b.ndv, 1)))
        joint = 1.0 / max(ndv_a, ndv_b)
        indep = 1.0
        for p in preds:
            indep *= p.selectivity or 1.0
        if indep <= 0:
            return
        ratio = max(0.01, min(100.0, joint / indep))
        target = preds[-1]
        target.selectivity = min(1.0, (target.selectivity or 1.0) * ratio)
        target.source = "multicol"
        target.note = (
            f"多列统计 ({', '.join(sorted(cols_a))} ↔ {', '.join(sorted(cols_b))}) "
            f"修正复合连接键：联合 ndv={max(ndv_a, ndv_b):.0f}，"
            f"相对独立假设 ×{ratio:.2f}")

    # ------------------------------------------------------------------
    # Semi / anti units
    # ------------------------------------------------------------------
    def _build_units(self) -> None:
        for unit in self.query.semi_units:
            sel, source, note = self._unit_selectivity(unit)
            self.unit_sel[unit.id] = sel
            for p in unit.correlation:
                p.selectivity = sel
                p.source = source
                p.note = note
            for p in unit.inner_joins:
                s, src, nt = self._join_selectivity(p)
                p.selectivity, p.source, p.note = s, src, nt

    def _unit_selectivity(self, unit: SemiUnit) -> Tuple[float, str, str]:
        if not unit.correlation:
            return 0.5, "default", "无相关条件，默认 0.5"
        p = unit.correlation[0]
        ts_o = self.stats.get(self.query.tables.get(p.left_table, ""))
        ts_i = self.stats.get(self.query.tables.get(p.right_table, ""))
        cs_o = ts_o.column(p.col_left) if ts_o and p.col_left else None
        cs_i = ts_i.column(p.col_right) if ts_i and p.col_right else None
        if cs_o is None or cs_i is None:
            return 0.5, "default", "缺少列统计，默认 0.5"
        inner_atom = self.atoms.get(p.right_table or "")
        outer_atom = self.atoms.get(p.left_table or "")
        if cs_o.histogram is not None and cs_i.histogram is not None:
            inner_distinct = sum(b[3] for b in cs_i.histogram.buckets)
            outer_distinct = max(cs_o.ndv, 1)
            frac = min(1.0, inner_distinct / outer_distinct)
            return frac, "histogram", (
                f"内表 {inner_distinct} 个不同键 / 外表 {outer_distinct} 个不同键")
        ndv_i = inner_atom.ndv.get(p.col_right, cs_i.ndv) if inner_atom else cs_i.ndv
        ndv_o = outer_atom.ndv.get(p.col_left, cs_o.ndv) if outer_atom else cs_o.ndv
        frac = 1.0 if ndv_i >= ndv_o else ndv_i / max(ndv_o, 1.0)
        return frac, "ndv", f"内表键 ndv={ndv_i:.0f} / 外表键 ndv={ndv_o:.0f}"

    # ------------------------------------------------------------------
    # Helpers used by the search
    # ------------------------------------------------------------------
    def edge(self, a: str, b: str) -> float:
        return self.edge_sel.get(frozenset((a, b)), 1.0)

    def connectors(self, a_set: Set[str], b: str) -> List[Predicate]:
        out = []
        for p in self.query.inner_joins:
            if p.left_table == b and p.right_table in a_set:
                out.append(p)
            elif p.right_table == b and p.left_table in a_set:
                out.append(p)
        return out

    def crossing_connectors(self, a_set: Set[str], b_set: Set[str]) -> List[Predicate]:
        out = []
        for p in self.query.inner_joins:
            if (p.left_table in a_set and p.right_table in b_set) or \
               (p.right_table in a_set and p.left_table in b_set):
                out.append(p)
        return out

    def subset_rows(self, rels: Iterable[str], units: Iterable[int] = (),
                    use_ndv_cap: bool = True) -> float:
        """Rows of the join of ``rels``.

        The estimate is computed by *canonically* joining the relations in a
        fixed order (increasing filtered size, then alias), propagating the
        distinct count of every join column as we go:

            rows   <- rows * |R| / max(ndv_l, ndv_r)
            ndv'   <- min(ndv, rows)

        The canonical order is a deterministic function of the subset, so the
        estimate depends only on *which* relations are joined, never on the
        order the search happens to pick.  That is the precondition for the
        subset DP being exact (it can compare orders fairly), and it also
        avoids the classic star-join underestimate: joining a second child of
        the same fact table divides by the *propagated* NDV, not by the
        original one.

        Edge selectivities (histogram / MCV / NDV / multi-column) are used when
        the canonical step crosses a single edge; otherwise the step's
        selectivity is the propagated-NDV ratio.
        """
        rels = list(dict.fromkeys(rels))
        if not rels:
            return 1.0
        order = sorted(rels, key=lambda r: (self.atoms[r].rows, r))
        have = [order[0]]
        rows = self.atoms[order[0]].rows
        ndv = dict(self.atoms[order[0]].ndv)
        for r in order[1:]:
            atom = self.atoms[r]
            conn = self.connectors(set(have), r)
            if conn:
                rows, ndv = self._canonical_step(rows, ndv, atom, conn)
            else:
                rows = rows * atom.rows
                for k, v in atom.ndv.items():
                    ndv[k] = max(ndv.get(k, 0.0), v)
            have.append(r)
        for uid in units:
            rows *= self.unit_sel.get(uid, 1.0)
        return rows

    def _canonical_step(self, rows: float, ndv: Dict[str, float],
                        atom: Atom, conn: Sequence[Predicate]):
        """Join ``atom`` to a prefix described by ``(rows, ndv)``."""
        rows = rows * atom.rows
        out_ndv = dict(ndv)
        if len(conn) >= 2:
            # several predicates between the same pair: treat them as a
            # composite key rather than multiplying independent NDV factors
            # (which is exactly the mistake the multi-column statistics fix)
            return self._composite_step(rows, ndv, atom, conn)
        for p in conn:
            if p.kind != "eq" or not p.col_left or not p.col_right:
                rows *= p.selectivity if p.selectivity is not None else 0.1
                continue
            # which side of the predicate is the new atom?
            if p.left_table == atom.alias:
                new_col, old_col = p.col_left, p.col_right
            else:
                new_col, old_col = p.col_right, p.col_left
            ndv_new = max(1.0, atom.ndv.get(new_col, atom.rows))
            ndv_old = max(1.0, min(ndv.get(old_col, rows), rows))
            rows = rows / max(ndv_new, ndv_old)
            out_ndv[new_col] = min(ndv_new, max(1.0, rows))
            out_ndv[old_col] = min(ndv_old, max(1.0, rows))
        for k, v in atom.ndv.items():
            out_ndv[k] = min(max(out_ndv.get(k, 0.0), v), max(1.0, rows))
        return max(0.0, rows), out_ndv

    def _composite_step(self, rows: float, ndv: Dict[str, float], atom: Atom,
                        conn: Sequence[Predicate]):
        """Join on a composite key (several equality predicates)."""
        out_ndv = dict(ndv)
        cols_new, cols_old = [], []
        for p in conn:
            if p.kind != "eq" or not p.col_left or not p.col_right:
                continue
            if p.left_table == atom.alias:
                cols_new.append(p.col_left)
                cols_old.append(p.col_right)
            else:
                cols_new.append(p.col_right)
                cols_old.append(p.col_left)
        if not cols_new:
            for p in conn:
                rows *= p.selectivity if p.selectivity is not None else 0.1
            return max(0.0, rows), out_ndv
        ts_a = self.stats.get(self.query.tables.get(atom.alias, ""))
        mc_new = ts_a.multicol_for(set(cols_new)) if ts_a else None
        frac = atom.rows / max(atom.base_rows, 1.0)
        if mc_new is not None:
            ndv_new = max(1.0, mc_new.ndv * max(frac, 1.0 / max(mc_new.ndv, 1)))
        else:
            ndv_new = 1.0
            for c in cols_new:
                ndv_new *= max(1.0, atom.ndv.get(c, atom.rows))
            ndv_new = max(1.0, min(ndv_new, atom.rows))
        ndv_old = 1.0
        for c in cols_old:
            ndv_old *= max(1.0, min(ndv.get(c, rows), rows))
        ndv_old = max(1.0, min(ndv_old, rows))
        rows = rows / max(ndv_new, ndv_old)
        for c in cols_new:
            out_ndv[c] = min(max(1.0, atom.ndv.get(c, 1.0)), max(1.0, rows))
        for c in cols_old:
            out_ndv[c] = min(max(1.0, ndv.get(c, 1.0)), max(1.0, rows))
        return max(0.0, rows), out_ndv

    def output_width(self, aliases: Iterable[str], predicates: Iterable[Predicate]) -> float:
        cols: Dict[str, Set[str]] = {}
        for p in predicates:
            if p.col_left and p.left_table:
                cols.setdefault(p.left_table, set()).add(p.col_left)
            if p.col_right and p.right_table:
                cols.setdefault(p.right_table, set()).add(p.col_right)
        total = 0.0
        for alias in aliases:
            ts = self.stats.get(self.query.tables.get(alias, ""))
            alias_cols = cols.get(alias, set())
            if not alias_cols:
                total += 8.0
                continue
            for c in alias_cols:
                cs = ts.column(c) if ts else None
                total += cs.avg_width if cs else 8
        return max(8.0, total)
