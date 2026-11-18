"""Parser and rewrite tests: join graph, outer joins, semi-join unnesting."""

from __future__ import annotations

import pytest

from teaopt.parser import ParseError, analyze


def test_inner_join_graph():
    q = analyze("SELECT * FROM a JOIN b ON a.x = b.x JOIN c ON c.x = b.x")
    assert set(q.tables) == {"a", "b", "c"}
    assert len(q.inner_joins) == 2
    assert q.stages[0].relations == ["a", "b", "c"]


def test_comma_join_is_flat():
    q = analyze("SELECT * FROM a, b, c WHERE a.x = b.x AND b.x = c.x")
    assert len(q.inner_joins) == 2
    assert len(q.stages) == 1


def test_left_join_creates_stage_boundary():
    q = analyze("SELECT * FROM a JOIN b ON a.x = b.x LEFT JOIN c ON c.x = b.x")
    assert len(q.outer_joins) == 1
    assert [s.relations for s in q.stages] == [["a", "b"], ["c"]]
    assert q.stages[1].outer is not None


def test_right_join_is_normalised():
    q = analyze("SELECT * FROM a RIGHT JOIN b ON a.x = b.x")
    assert q.outer_joins[0].kind == "left"
    assert q.outer_joins[0].preserved == ["b"]
    assert q.outer_joins[0].nullable == ["a"]
    assert "RIGHT JOIN" in " ".join(q.notes)


def test_outer_join_degenerates_when_nullable_side_filtered():
    q = analyze("SELECT * FROM a LEFT JOIN b ON a.x = b.x WHERE b.v > 3")
    assert q.outer_joins == []
    assert len(q.stages) == 1
    assert any("退化为内连接" in n for n in q.notes)


def test_is_null_keeps_outer_join():
    q = analyze("SELECT * FROM a LEFT JOIN b ON a.x = b.x WHERE b.v IS NULL")
    assert len(q.outer_joins) == 1
    assert [s.relations for s in q.stages] == [["a"], ["b"]]


def test_derived_table_is_folded():
    q = analyze("SELECT * FROM (SELECT * FROM lots WHERE grade = 'S') x "
                "JOIN origins o ON o.origin_id = x.origin_id")
    assert q.tables["x"] == "lots"
    assert q.folded == {"x": "lots"}
    assert q.local_filters["x"][0].text.replace('"', "") in (
        "grade = 'S'", "x.grade = 'S'")


def test_parse_error_on_unknown_alias():
    with pytest.raises(ParseError):
        analyze("SELECT * FROM a JOIN b ON z.x = b.x")


def test_exists_rewritten_to_semi():
    q = analyze("SELECT * FROM lots l WHERE EXISTS (SELECT 1 FROM reports r "
                "WHERE r.lot_id = l.lot_id AND r.passed = 1)")
    assert len(q.semi_units) == 1
    unit = q.semi_units[0]
    assert unit.kind == "semi"
    assert unit.outer_refs == {"l"}
    assert unit.inner_relations == ["r"]
    assert unit.correlation[0].col_left == "lot_id"
    assert "r" in q.tables


def test_not_exists_rewritten_to_anti():
    q = analyze("SELECT * FROM orders od WHERE NOT EXISTS "
                "(SELECT 1 FROM shipments sh WHERE sh.order_id = od.order_id)")
    assert q.semi_units[0].kind == "anti"


def test_in_subquery_rewritten_to_semi():
    q = analyze("SELECT * FROM orders od WHERE od.customer_id IN "
                "(SELECT cu.customer_id FROM customers cu WHERE cu.channel = '线上')")
    assert q.semi_units[0].kind == "semi"
    assert q.semi_units[0].mode == "in"


def test_uncorrelated_scalar_subquery_is_kept_as_filter():
    q = analyze("SELECT * FROM a WHERE a.x > (SELECT MAX(b.x) FROM b)")
    assert q.semi_units == []
    # a scalar subquery is not correlated, so it stays a ordinary filter on `a`
    assert [p.kind for p in q.local_filters["a"]] == ["range"]


def test_group_by_is_preserved():
    q = analyze("SELECT cu.city, COUNT(*) FROM orders od "
                "JOIN customers cu ON od.customer_id = cu.customer_id "
                "GROUP BY cu.city ORDER BY cu.city")
    assert q.group_by and "GROUP BY" in q.group_by.upper()
    assert q.order_by and "ORDER BY" in q.order_by.upper()


def test_rule5_stage_order_is_prefix_closed():
    """Any concatenation of stages is a legal order; moving a nullable side
    before its preserved side is not."""
    q = analyze("SELECT * FROM a JOIN b ON a.x = b.x LEFT JOIN c ON c.x = b.x")
    stage_of = {r: i for i, st in enumerate(q.stages) for r in st.relations}
    assert stage_of["c"] == 1
    assert stage_of["a"] == stage_of["b"] == 0


def test_mixed_on_predicates_are_split():
    """``ON k = k AND flag = 1``: the equality joins, the rest filters."""
    q = analyze("SELECT * FROM a JOIN b ON a.x = b.x AND b.flag = 1")
    assert len(q.inner_joins) == 1
    assert [p.text for p in q.local_filters["b"]] == ["b.flag = 1"]


def test_constant_where_predicate_on_single_table_is_accepted():
    # a constant predicate referencing no column is attached to the only table
    q = analyze("SELECT * FROM a WHERE 1 = 1")
    assert q.tables and not q.local_filters.get("a") == []
    # ... but on a multi-table query it is ambiguous and rejected
    with pytest.raises(ParseError):
        analyze("SELECT * FROM a JOIN b ON a.x = b.x WHERE 1 = 1")
