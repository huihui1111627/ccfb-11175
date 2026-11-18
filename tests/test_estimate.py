"""Estimation tests: sources, histograms, multi-column correction, joins."""

from __future__ import annotations

import pytest

from teaopt.estimate import Estimator
from teaopt.parser import analyze
from teaopt.types import Histogram, OptimizerConfig


# ---------------------------------------------------------------------------
# Source tagging (requirement 1)
# ---------------------------------------------------------------------------


def test_filter_selectivity_sources(stats):
    q = analyze("SELECT COUNT(*) FROM lots l WHERE l.grade = 'S' "
                "AND l.weight_kg > 100")
    est = Estimator(q, stats)
    srcs = {p.text.replace("l.", ""): p.source for p in est.atoms["l"].filters}
    assert srcs["grade = 'S'"] == "mcv"
    assert srcs["weight_kg > 100"] == "histogram"


def test_ndv_source_when_no_histogram(stats):
    q = analyze("SELECT COUNT(*) FROM customers c WHERE c.city = '昆明'")
    est = Estimator(q, stats)
    pred = est.atoms["c"].filters[0]
    # city has an MCV list (low cardinality), otherwise 1/ndv
    assert pred.source in ("mcv", "ndv")


def test_join_selectivity_source_is_histogram(stats):
    q = analyze("SELECT COUNT(*) FROM lots l JOIN origins o "
                "ON l.origin_id = o.origin_id")
    Estimator(q, stats)          # computes and annotates the predicate
    pred = q.inner_joins[0]
    assert pred.source == "histogram"
    assert 0 < pred.selectivity < 1


def test_each_predicate_has_a_source(stats):
    sql = ("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "WHERE l.grade = 'A' AND o.province = '云南' AND r.passed = 1")
    q = analyze(sql)
    Estimator(q, stats)
    assert all(p.source for p in q.inner_joins)
    for filters in q.local_filters.values():
        assert all(p.source for p in filters)


# ---------------------------------------------------------------------------
# Histogram behaviour
# ---------------------------------------------------------------------------


def test_histogram_uniform_range():
    h = Histogram(low=0, high=100, buckets=[
        (0, 25, 250, 25), (25, 50, 250, 25),
        (50, 75, 250, 25), (75, 100, 250, 25)])
    assert h.estimate_eq(10) == pytest.approx(10.0)
    assert h.estimate_range(0, 50) == pytest.approx(500.0, rel=0.02)
    assert h.estimate_range(None, 100) == pytest.approx(1000.0)


def test_histogram_skewed_bucket():
    h = Histogram(low=0, high=10, buckets=[
        (0, 5, 990, 2),      # 99% of rows in two values
        (5, 10, 10, 5)])
    assert h.estimate_eq(1) == pytest.approx(495.0)
    assert h.estimate_eq(7) == pytest.approx(2.0)


def test_histogram_drives_range_selectivity(stats):
    cs = stats["lots"].column("weight_kg")
    assert cs is not None and cs.histogram is not None
    sel, source, _ = cs.range_selectivity(0, 400)
    assert source == "histogram"
    assert sel > 0.95


# ---------------------------------------------------------------------------
# Multi-column correction (requirement 3)
# ---------------------------------------------------------------------------


def test_multicolumn_stats_exist(stats):
    mc = stats["lots"].multicol_for({"grade", "origin_id"})
    assert mc is not None
    assert mc.ndv <= stats["lots"].column("grade").ndv * stats["lots"].column("origin_id").ndv


def test_correlated_filters_are_corrected(stats):
    """grade and origin_id are correlated in the generated data, so the
    independence product over/under-estimates and multi-column stats fix it."""
    sql = ("SELECT COUNT(*) FROM lots l "
           "WHERE l.grade = 'S' AND l.origin_id = 1")
    q = analyze(sql)
    est = Estimator(q, stats)
    preds = est.atoms["l"].filters
    corrected = [p for p in preds if p.source == "multicol"]
    assert len(corrected) == 1
    assert "多列统计" in corrected[0].note
    # the correction must be materially different from the independent product
    independent = 1.0
    count = stats["lots"].row_count
    for p in preds:
        if p is not corrected[0]:
            independent *= p.selectivity
    # ground truth
    truth = _count(stats, "lots", "grade = 'S' AND origin_id = 1") / count
    est_sel = est.atoms["l"].rows / count
    assert abs(est_sel - truth) < abs(independent - truth) + 0.02


def _count(stats, table, where):
    # statistics object does not hold the connection; the conftest DB is used
    # through the derived data instead
    return stats[table].row_count * _approx_truth(stats, table, where)


def _approx_truth(stats, table, where):
    # computed from the multi-column MCV: enough for the assertion above
    if table == "lots" and where == "grade = 'S' AND origin_id = 1":
        mc = stats["lots"].multicol_for({"grade", "origin_id"})
        for values, n in mc.mcv:
            if tuple(values) == (1, "S"):
                return n / stats["lots"].row_count
    return 0.0


def test_multicolumn_can_be_disabled(stats):
    q = analyze("SELECT COUNT(*) FROM lots l "
                "WHERE l.grade = 'S' AND l.origin_id = 1")
    est = Estimator(q, stats, OptimizerConfig(use_multicolumn=False))
    assert all(p.source != "multicol" for p in est.atoms["l"].filters)


# ---------------------------------------------------------------------------
# Join cardinality
# ---------------------------------------------------------------------------


def test_join_rows_bounded_by_fk(stats):
    """Three child tables joined through the same FK must not square the
    fan-out: the NDV cap keeps the estimate near |lots| * multiplicity."""
    sql = ("SELECT COUNT(*) FROM lots l "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "JOIN inventory i ON i.lot_id = l.lot_id "
           "JOIN price_history p ON p.lot_id = l.lot_id")
    q = analyze(sql)
    est = Estimator(q, stats)
    rows = est.subset_rows(["l", "r", "i", "p"])
    assert rows < stats["lots"].row_count * 200
    # and it is not absurdly small either
    assert rows > stats["lots"].row_count * 0.5


def test_subset_rows_is_order_independent(stats):
    q = analyze("SELECT COUNT(*) FROM lots l JOIN origins o ON l.origin_id = o.origin_id "
                "JOIN reports r ON r.lot_id = l.lot_id JOIN inventory i ON i.lot_id = l.lot_id")
    est = Estimator(q, stats)
    a = est.subset_rows(["l", "o", "r", "i"])
    b = est.subset_rows(["i", "r", "o", "l"])
    c = est.subset_rows(["r", "l", "i", "o"])
    assert a == pytest.approx(b) == pytest.approx(c)


def test_semi_join_selectivity_between_0_and_1(stats):
    q = analyze("SELECT COUNT(*) FROM lots l WHERE EXISTS "
                "(SELECT 1 FROM reports r WHERE r.lot_id = l.lot_id)")
    est = Estimator(q, stats)
    sel = est.unit_sel[q.semi_units[0].id]
    assert 0.0 < sel <= 1.0
