"""Table / column / multi-column statistics collection from SQLite.

For a real deployment the statistics would come from ``ANALYZE`` or the
catalog.  Here we collect them with ordinary SQL: row counts, NULL counts,
distinct counts, min/max, equi-width histograms (64 buckets), top-k most
common values for skewed columns, and multi-column DISTINCT counts for a
configured list of correlated column pairs (requirement 3).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import ColumnStats, Histogram, MultiColStats, TableStats

HISTOGRAM_BUCKETS = 64
MCV_LIMIT = 12
SAMPLE_FOR_HISTOGRAM = 200_000


class StatisticsCollector:
    """Collect :class:`TableStats` from a SQLite connection.

    Parameters
    ----------
    conn:
        open ``sqlite3.Connection``.
    multicolumn:
        list of ``(table, [col_a, col_b, ...])`` for which multi-column
        statistics should be built.
    """

    def __init__(self, conn, multicolumn: Optional[Sequence[Tuple[str, Sequence[str]]]] = None,
                 histogram_buckets: int = HISTOGRAM_BUCKETS):
        self.conn = conn
        self.multicolumn = list(multicolumn or [])
        self.histogram_buckets = histogram_buckets
        self._cache: Dict[str, TableStats] = {}

    # -- public ------------------------------------------------------------
    def collect_all(self, tables: Iterable[str]) -> Dict[str, TableStats]:
        for t in tables:
            self.collect(t)
        return dict(self._cache)

    def collect(self, table: str) -> TableStats:
        if table in self._cache:
            return self._cache[table]
        cur = self.conn.cursor()
        row_count = cur.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        columns = [r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')]
        indexes = []
        for row in cur.execute(f'PRAGMA index_list("{table}")'):
            idx_name = row[1]
            cols = [r[2] for r in cur.execute(f'PRAGMA index_info("{idx_name}")')]
            indexes.append(cols)
        page_count = self._page_count(table, row_count)
        stats = TableStats(name=table, row_count=row_count, page_count=page_count,
                           indexes=indexes)
        for col in columns:
            stats.columns[col] = self._column_stats(table, col, row_count)
        self._cache[table] = stats
        for tbl, cols in self.multicolumn:
            if tbl == table:
                mc = self._multicol_stats(table, list(cols), row_count)
                if mc is not None:
                    stats.multicol[tuple(sorted(cols))] = mc
        return stats

    # -- internals ---------------------------------------------------------
    def _page_count(self, table: str, row_count: int) -> int:
        try:
            page_size = self.conn.execute("PRAGMA page_size").fetchone()[0]
            rows = self.conn.execute(
                "SELECT SUM(pgsize) FROM dbstat WHERE name = ?", (table,)).fetchone()[0]
            if rows:
                return max(1, int(math.ceil(rows / max(page_size, 1))))
        except Exception:
            pass
        return max(1, int(math.ceil(row_count * 48 / 4096)) + 1)

    def _column_stats(self, table: str, col: str, row_count: int) -> ColumnStats:
        cur = self.conn.cursor()
        qcol = f'"{col}"'
        row = cur.execute(
            f'SELECT COUNT({qcol}), MIN({qcol}), MAX({qcol}) FROM "{table}"').fetchone()
        non_null = row[0] or 0
        null_count = row_count - non_null
        min_v, max_v = row[1], row[2]
        distinct = cur.execute(
            f'SELECT COUNT(DISTINCT {qcol}) FROM "{table}"').fetchone()[0] or 0
        is_numeric = isinstance(min_v, (int, float)) and not isinstance(min_v, bool) \
            and isinstance(max_v, (int, float))
        cs = ColumnStats(name=col, table=table, row_count=row_count, ndv=distinct,
                         null_count=null_count, min_value=min_v, max_value=max_v,
                         is_numeric=is_numeric)
        if row_count == 0 or non_null == 0:
            return cs
        # histogram for numeric / date-like columns
        if is_numeric and distinct > 1 and isinstance(min_v, (int, float)):
            cs.histogram = self._histogram(table, col, float(min_v), float(max_v),
                                           non_null, distinct)
        # MCV for low-cardinality or skewed columns
        if distinct > 0 and (distinct <= 200 or not is_numeric):
            self._mcv(table, col, row_count, cs, limit=MCV_LIMIT)
        elif distinct > 0 and distinct <= 5000:
            # still record the top values when they are clearly skewed
            self._mcv(table, col, row_count, cs, limit=MCV_LIMIT,
                      only_if_skewed=True)
        return cs

    def _histogram(self, table: str, col: str, low: float, high: float,
                   non_null: int, distinct: int) -> Histogram:
        cur = self.conn.cursor()
        qcol = f'"{col}"'
        if high <= low:
            return Histogram(low=low, high=high, buckets=[(low, high, non_null, distinct)])
        width = (high - low) / self.histogram_buckets
        buckets: List[Tuple[float, float, int, int]] = []
        for i in range(self.histogram_buckets):
            b_low = low + i * width
            b_high = b_low + width
            if i == self.histogram_buckets - 1:
                cond = f"{qcol} >= ? AND {qcol} <= ?"
                params: Tuple[Any, ...] = (b_low, high)
            else:
                cond = f"{qcol} >= ? AND {qcol} < ?"
                params = (b_low, b_high)
            cnt, d = cur.execute(
                f'SELECT COUNT(*), COUNT(DISTINCT {qcol}) FROM "{table}" WHERE {cond}',
                params).fetchone()
            buckets.append((float(b_low), float(b_high), int(cnt), int(d or 0)))
        return Histogram(low=low, high=high, buckets=buckets,
                         null_count=0, total_count=non_null)

    def _mcv(self, table: str, col: str, row_count: int, cs: ColumnStats,
             limit: int, only_if_skewed: bool = False) -> None:
        cur = self.conn.cursor()
        qcol = f'"{col}"'
        rows = cur.execute(
            f'SELECT {qcol} AS v, COUNT(*) AS c FROM "{table}" '
            f'WHERE {qcol} IS NOT NULL GROUP BY {qcol} ORDER BY c DESC LIMIT ?',
            (limit,)).fetchall()
        if not rows:
            return
        top = [(r[0], int(r[1])) for r in rows]
        mass = sum(c for _, c in top) / row_count if row_count else 0.0
        if only_if_skewed:
            uniform = 1.0 / max(cs.ndv, 1)
            if top[0][1] / max(row_count, 1) < max(3.0 * uniform, 0.05):
                return
        cs.mcv = top
        cs.mcv_mass = mass

    def _multicol_stats(self, table: str, cols: List[str], row_count: int) -> Optional[MultiColStats]:
        if len(cols) < 2 or row_count == 0:
            return None
        # keep the SELECT order identical to the stored column order so MCV
        # tuples line up with ``MultiColStats.columns``
        cols = sorted(cols)
        cur = self.conn.cursor()
        qcols = ", ".join(f'"{c}"' for c in cols)
        ndv = cur.execute(
            f'SELECT COUNT(*) FROM (SELECT DISTINCT {qcols} FROM "{table}")').fetchone()[0]
        mc = MultiColStats(table=table, columns=tuple(sorted(cols)), ndv=int(ndv),
                           row_count=row_count)
        try:
            rows = cur.execute(
                f'SELECT {qcols}, COUNT(*) AS c FROM "{table}" GROUP BY {qcols} '
                f'ORDER BY c DESC LIMIT ?', (MCV_LIMIT,)).fetchall()
            mc.mcv = [(tuple(r[:-1]), int(r[-1])) for r in rows]
            mc.mcv_mass = sum(r[-1] for r in rows) / row_count if row_count else 0.0
        except Exception:
            pass
        return mc


def collect(conn, tables: Iterable[str],
            multicolumn: Optional[Sequence[Tuple[str, Sequence[str]]]] = None
            ) -> Dict[str, TableStats]:
    return StatisticsCollector(conn, multicolumn=multicolumn).collect_all(tables)
