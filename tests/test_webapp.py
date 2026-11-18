"""Web API tests."""

from __future__ import annotations

import pytest

from teaopt.demo import MULTICOLUMN
from teaopt.service import QueryService
from teaopt.webapp import create_app


@pytest.fixture(scope="module")
def client(db_path, stats):
    app = create_app()
    # point the app at the shared fixture database so the module does not build
    # a second copy
    app.config["SERVICE"] = QueryService(db_path, MULTICOLUMN)
    app.testing = True
    with app.test_client() as c:
        yield c


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "查询编辑" in html and "连接图" in html and "计划树" in html
    assert "行数估算对比" in html and "枚举过程" in html


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_schema(client):
    data = client.get("/api/schema").get_json()
    assert len(data["tables"]) == 16
    lots = next(t for t in data["tables"] if t["name"] == "lots")
    assert lots["rows"] > 0
    assert any(c["histogram"] for c in lots["columns"])
    assert lots["multicol"]


def test_enumeration(client):
    data = client.get("/api/enumeration").get_json()
    assert data["shapes"] == ["chain", "star", "cycle", "clique"]
    for shape in data["shapes"]:
        assert data["table"][shape]


def test_optimize_returns_all_panes(client):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "WHERE o.province = '福建'")
    data = client.post("/api/optimize", json={"sql": sql}).get_json()
    for key in ("plan", "order", "cost", "est_rows", "alternatives", "notes",
                "predicates", "join_graph", "validation", "timing", "stages",
                "query", "brute_force", "threshold", "atom_count"):
        assert key in data, key
    assert data["validation"]["okay"]
    # predicate sources are present for the "selection-rate provenance" pane
    assert all(p["source"] for p in data["predicates"])
    # graph has nodes for each relation
    assert len(data["join_graph"]["nodes"]) == 3


def test_optimize_threshold_switch(client):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "JOIN inventory i ON i.lot_id = l.lot_id "
           "JOIN price_history p ON p.lot_id = l.lot_id "
           "JOIN purchases pu ON pu.lot_id = l.lot_id")
    exhaustive = client.post("/api/optimize", json={"sql": sql, "threshold": 10}).get_json()
    greedy = client.post("/api/optimize", json={"sql": sql, "threshold": 3,
                                               "run": False}).get_json()
    assert exhaustive["algorithm"].startswith("exhaustive")
    assert greedy["algorithm"] == "greedy"
    assert greedy["exhaustive_available"] is False


def test_optimize_rejects_bad_sql(client):
    r = client.post("/api/optimize", json={"sql": "SELECT FROM"})
    assert r.status_code == 400
    assert r.get_json()["error"]


def test_optimize_rejects_empty_sql(client):
    r = client.post("/api/optimize", json={"sql": ""})
    assert r.status_code == 400


def test_brute_force_endpoint(client):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id")
    data = client.post("/api/brute-force",
                       json={"sql": sql, "threshold": 10}).get_json()
    assert data["skipped"] is False
    assert data["match"] is True
    assert data["perms"] > 0


def test_brute_force_skips_large_queries(client):
    # 13 connected relations -> far beyond the 8-atom cap for enumeration
    sql = "SELECT COUNT(*) AS n FROM origins o"
    joins = {
        "lots": ("l", "l.origin_id = o.origin_id"),
        "reports": ("r", "r.lot_id = l.lot_id"),
        "inventory": ("i", "i.lot_id = l.lot_id"),
        "price_history": ("p", "p.lot_id = l.lot_id"),
        "purchases": ("pu", "pu.lot_id = l.lot_id"),
        "blend_lots": ("bl", "bl.lot_id = l.lot_id"),
        "blends": ("b", "b.blend_id = bl.blend_id"),
        "orders": ("od", "od.blend_id = b.blend_id"),
        "customers": ("cu", "cu.customer_id = od.customer_id"),
        "shipments": ("sh", "sh.order_id = od.order_id"),
        "warehouses": ("w", "w.warehouse_id = sh.warehouse_id"),
        "suppliers": ("s", "s.origin_id = o.origin_id"),
    }
    for table, (alias, cond) in joins.items():
        sql += f' JOIN "{table}" {alias} ON {cond}'
    data = client.post("/api/brute-force", json={"sql": sql}).get_json()
    assert data.get("skipped") is True, data
    assert "dp_cost" in data


def test_reverse_endpoint(client):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id")
    data = client.post("/api/reverse",
                       json={"sql": sql, "order": ["l", "o"]}).get_json()
    assert data["okay"], data.get("error")


def test_outer_join_query_through_api(client):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "LEFT JOIN inventory i ON i.lot_id = l.lot_id "
           "WHERE o.province = '福建'")
    data = client.post("/api/optimize", json={"sql": sql}).get_json()
    assert data["order"][-1] == "i"
    assert data["validation"]["okay"]


def test_counterpart_wall_clock_is_reported(client):
    sql = ("SELECT COUNT(*) AS n FROM lots l "
           "JOIN origins o ON l.origin_id = o.origin_id "
           "JOIN reports r ON r.lot_id = l.lot_id "
           "JOIN inventory i ON i.lot_id = l.lot_id "
           "JOIN price_history p ON p.lot_id = l.lot_id "
           "WHERE o.province = '福建'")
    data = client.post("/api/optimize", json={"sql": sql}).get_json()
    timing = data["timing"]
    assert timing.get("chosen_ms") is not None
    assert "slow_cost" in timing or "slow_ms" in timing
    assert "sqlite_ms" in timing
