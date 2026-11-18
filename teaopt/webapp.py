"""Flask web UI for the tea-blending query optimizer.

Pages
-----
``/``                single-page workspace with five panes
``/api/schema``      tables, columns, NDV, histograms, multi-column stats
``/api/optimize``    parse + optimise + execute one SQL query
``/api/enumeration`` sub-problem counts for chain / star / cycle / clique
``/api/brute-force`` compare the DP result with an exhaustive permutation search
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request

from .demo import MULTICOLUMN, TABLES
from .service import QueryService, ServiceError, ensure_demo_database
from .types import OptimizerConfig


EXAMPLES: List[Dict[str, str]] = [
    {
        "name": "链式：批次 → 产区 → 检测报告",
        "sql": ("SELECT l.lot_id, l.grade, o.province, r.score\n"
                "FROM lots l\n"
                "JOIN origins o ON l.origin_id = o.origin_id\n"
                "JOIN reports r ON r.lot_id = l.lot_id\n"
                "WHERE r.passed = 1 AND o.province = '福建' AND l.grade = 'S'"),
    },
    {
        "name": "星式：产区为中心的 5 表连接",
        "sql": ("SELECT o.province, COUNT(*) AS n\n"
                "FROM lots l\n"
                "JOIN origins o ON l.origin_id = o.origin_id\n"
                "JOIN inventory i ON i.lot_id = l.lot_id\n"
                "JOIN reports r ON r.lot_id = l.lot_id\n"
                "JOIN price_history p ON p.lot_id = l.lot_id\n"
                "WHERE o.province = '云南' AND r.passed = 1\n"
                "GROUP BY o.province"),
    },
    {
        "name": "相关子查询 → 半连接",
        "sql": ("SELECT COUNT(*) AS n\n"
                "FROM orders od\n"
                "JOIN customers cu ON od.customer_id = cu.customer_id\n"
                "WHERE NOT EXISTS (SELECT 1 FROM shipments sh\n"
                "                  WHERE sh.order_id = od.order_id)\n"
                "  AND cu.channel = '线上'"),
    },
    {
        "name": "外连接：合法换序约束",
        "sql": ("SELECT COUNT(*) AS n\n"
                "FROM lots l\n"
                "JOIN origins o ON l.origin_id = o.origin_id\n"
                "LEFT JOIN inventory i ON i.lot_id = l.lot_id\n"
                "WHERE o.province = '云南' AND l.grade = 'S'"),
    },
    {
        "name": "多列相关：产区 + 等级",
        "sql": ("SELECT COUNT(*) AS n\n"
                "FROM lots l\n"
                "JOIN origins o ON l.origin_id = o.origin_id\n"
                "JOIN reports r ON r.lot_id = l.lot_id\n"
                "WHERE l.grade = 'A' AND l.origin_id = 1 AND r.passed = 1"),
    },
    {
        "name": "大查询：12 表（默认阈值 10 → 贪心）",
        "sql": ("SELECT COUNT(*) AS n\n"
                "FROM lots l\n"
                "JOIN origins o ON l.origin_id = o.origin_id\n"
                "JOIN reports r ON r.lot_id = l.lot_id\n"
                "JOIN inventory i ON i.lot_id = l.lot_id\n"
                "JOIN warehouses w ON w.warehouse_id = i.warehouse_id\n"
                "JOIN price_history p ON p.lot_id = l.lot_id\n"
                "JOIN purchases pu ON pu.lot_id = l.lot_id\n"
                "JOIN suppliers s ON s.supplier_id = pu.supplier_id\n"
                "JOIN blend_lots bl ON bl.lot_id = l.lot_id\n"
                "JOIN blends b ON b.blend_id = bl.blend_id\n"
                "JOIN orders od ON od.blend_id = b.blend_id\n"
                "JOIN customers cu ON od.customer_id = cu.customer_id\n"
                "WHERE o.province = '云南'"),
    },
]


def create_app(root: Optional[str] = None) -> Flask:
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = ensure_demo_database(root)
    service = QueryService(db_path, MULTICOLUMN, TABLES)
    # warm up statistics so the first request is fast
    service.stats()

    app = Flask(__name__, static_folder=os.path.join(root, "web", "static"),
                template_folder=os.path.join(root, "web", "templates"))
    app.config["SERVICE"] = service

    @app.get("/")
    def index():
        return render_template("index.html", examples=EXAMPLES,
                               db_path=db_path)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "db": db_path, "tables": len(service.tables)})

    @app.get("/api/schema")
    def schema():
        return jsonify({"tables": service.schema(),
                        "multicolumn": [{"table": t, "columns": list(cols)}
                                        for t, cols in MULTICOLUMN]})

    @app.get("/api/examples")
    def examples():
        return jsonify({"examples": EXAMPLES})

    @app.get("/api/enumeration")
    def enumeration():
        sizes = request.args.get("sizes", "3,4,5,6,7,8")
        parsed = [int(s) for s in sizes.split(",") if s.strip().isdigit()]
        return jsonify(service.enumeration_stats(parsed or (3, 4, 5, 6, 7, 8)))

    @app.post("/api/optimize")
    def optimize():
        body = request.get_json(force=True, silent=True) or {}
        sql = (body.get("sql") or "").strip()
        if not sql:
            return jsonify({"error": "SQL 不能为空"}), 400
        config = _config_from_body(body)
        try:
            result = service.optimize(sql, config,
                                      run_on_sqlite=bool(body.get("run", True)))
        except ServiceError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - defensive
            return jsonify({"error": f"{type(exc).__name__}: {exc}",
                            "trace": traceback.format_exc()}), 500
        return jsonify(result)

    @app.post("/api/brute-force")
    def brute():
        body = request.get_json(force=True, silent=True) or {}
        sql = (body.get("sql") or "").strip()
        if not sql:
            return jsonify({"error": "SQL 不能为空"}), 400
        try:
            result = service.compare_with_brute_force(sql, _config_from_body(body))
        except ServiceError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}",
                            "trace": traceback.format_exc()}), 500
        return jsonify(result)

    @app.post("/api/reverse")
    def reverse():
        """Execute the query with a given join order (for timing comparison)."""
        body = request.get_json(force=True, silent=True) or {}
        sql = (body.get("sql") or "").strip()
        order = body.get("order") or []
        if not sql or not order:
            return jsonify({"error": "需要 sql 和 order"}), 400
        from .parser import analyze
        from .execute import execute, sql_for_order
        try:
            query = analyze(sql)
            with service.connect() as conn:
                order_sql = sql_for_order(conn, query, order)
                res = execute(conn, order_sql, sample=1, explain=False)
            return jsonify({"okay": res.okay, "error": res.error,
                            "sql": order_sql, "rows": res.rows})
        except Exception as exc:
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400

    return app


def _config_from_body(body: Dict[str, Any]) -> OptimizerConfig:
    cfg = OptimizerConfig()
    if "threshold" in body and body["threshold"] is not None:
        try:
            cfg.exhaustive_threshold = max(1, int(body["threshold"]))
        except (TypeError, ValueError):
            pass
    if body.get("algorithm") in ("auto", "exhaustive", "greedy"):
        cfg.algorithm = body["algorithm"]
    if "use_histograms" in body:
        cfg.use_histograms = bool(body["use_histograms"])
    if "use_multicolumn" in body:
        cfg.use_multicolumn = bool(body["use_multicolumn"])
    if "allow_cartesian" in body:
        cfg.allow_cartesian = bool(body["allow_cartesian"])
    for key in ("seq_page_cost", "random_page_cost", "cpu_tuple_cost"):
        if key in body:
            try:
                setattr(cfg, key, float(body[key]))
            except (TypeError, ValueError):
                pass
    return cfg


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)
