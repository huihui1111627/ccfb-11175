"""Synthetic tea-blending schema + data generator.

The domain: 茶叶拼配 (tea blending).  Every blended product is made from
*lots* (原料批次), which come from *origins* (产区), have *test reports*
(检测报告), *inventory* (库存) and *orders* (订单).  The demo uses 16
relations so the join graph is large enough to exercise the optimizer, and
the data intentionally contains

* **column correlation** – e.g. ``grade`` is a deterministic function of
  ``origin_id`` within a lot; ``passed`` strongly correlates with ``grade``;
  ``province`` is a function of ``origin_id``.
* **skew** – a few origins / customers / status values dominate.

The generator is deterministic given a seed, so the acceptance tests are
reproducible.
"""

from __future__ import annotations

import os
import random
import sqlite3
from typing import Dict, List, Sequence, Tuple

SCHEMA_SQL = """
CREATE TABLE origins (
    origin_id   INTEGER PRIMARY KEY,
    province    TEXT NOT NULL,
    region      TEXT NOT NULL,
    altitude    INTEGER NOT NULL
);
CREATE TABLE lots (
    lot_id      INTEGER PRIMARY KEY,
    origin_id   INTEGER NOT NULL,
    grade       TEXT NOT NULL,
    harvest_year INTEGER NOT NULL,
    weight_kg   REAL NOT NULL,
    price       REAL NOT NULL
);
CREATE TABLE inventory (
    inv_id      INTEGER PRIMARY KEY,
    lot_id      INTEGER NOT NULL,
    warehouse_id INTEGER NOT NULL,
    qty_kg      REAL NOT NULL,
    reserved_kg REAL NOT NULL
);
CREATE TABLE warehouses (
    warehouse_id INTEGER PRIMARY KEY,
    city        TEXT NOT NULL,
    capacity_kg REAL NOT NULL
);
CREATE TABLE reports (
    report_id   INTEGER PRIMARY KEY,
    lot_id      INTEGER NOT NULL,
    passed      INTEGER NOT NULL,
    inspector_id INTEGER NOT NULL,
    score       REAL NOT NULL
);
CREATE TABLE inspectors (
    inspector_id INTEGER PRIMARY KEY,
    lab         TEXT NOT NULL,
    certification TEXT NOT NULL
);
CREATE TABLE blends (
    blend_id    INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    style       TEXT NOT NULL,
    target_price REAL NOT NULL
);
CREATE TABLE blend_lots (
    blend_lot_id INTEGER PRIMARY KEY,
    blend_id    INTEGER NOT NULL,
    lot_id      INTEGER NOT NULL,
    ratio       REAL NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    blend_id    INTEGER NOT NULL,
    status      TEXT NOT NULL,
    order_date  TEXT NOT NULL,
    amount      REAL NOT NULL
);
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    channel     TEXT NOT NULL,
    city        TEXT NOT NULL,
    tier        TEXT NOT NULL
);
CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL,
    warehouse_id INTEGER NOT NULL,
    shipped_date TEXT NOT NULL,
    cost        REAL NOT NULL
);
CREATE TABLE suppliers (
    supplier_id INTEGER PRIMARY KEY,
    origin_id   INTEGER NOT NULL,
    contract_year INTEGER NOT NULL,
    rating      REAL NOT NULL
);
CREATE TABLE purchases (
    purchase_id INTEGER PRIMARY KEY,
    supplier_id INTEGER NOT NULL,
    lot_id      INTEGER NOT NULL,
    qty_kg      REAL NOT NULL,
    unit_price  REAL NOT NULL
);
CREATE TABLE grades (
    grade_code  TEXT PRIMARY KEY,
    tier        TEXT NOT NULL,
    min_score   REAL NOT NULL
);
CREATE TABLE warehouses_bak (
    warehouse_id INTEGER PRIMARY KEY,
    city        TEXT NOT NULL,
    capacity_kg REAL NOT NULL
);
CREATE TABLE price_history (
    ph_id       INTEGER PRIMARY KEY,
    lot_id      INTEGER NOT NULL,
    month       TEXT NOT NULL,
    price       REAL NOT NULL
);
"""

TABLES: List[str] = [
    "origins", "lots", "inventory", "warehouses", "reports", "inspectors",
    "blends", "blend_lots", "orders", "customers", "shipments", "suppliers",
    "purchases", "grades", "warehouses_bak", "price_history",
]

#: column pairs with known correlation, used for multi-column statistics
MULTICOLUMN: List[Tuple[str, Sequence[str]]] = [
    ("lots", ("origin_id", "grade")),
    ("origins", ("province", "region")),
    ("reports", ("lot_id", "passed")),
    ("orders", ("customer_id", "status")),
    ("customers", ("channel", "tier")),
    ("blend_lots", ("blend_id", "lot_id")),
    ("purchases", ("supplier_id", "lot_id")),
]

PROVINCES = ["云南", "福建", "浙江", "安徽", "四川", "湖南"]
REGIONS = {"云南": ["勐海", "临沧", "普洱"], "福建": ["武夷", "安溪", "福鼎"],
           "浙江": ["杭州", "绍兴"], "安徽": ["黄山", "六安"],
           "四川": ["雅安", "峨眉"], "湖南": ["安化", "长沙"]}
GRADES = ["S", "A", "B", "C"]
STYLES = ["普洱生茶", "普洱熟茶", "红茶", "白茶", "乌龙"]
STATUSES = ["created", "paid", "shipped", "cancelled", "refunded"]
CHANNELS = ["线上", "门店", "批发", "海外"]
TIERS = ["普通", "银卡", "金卡", "钻石"]
CITIES = ["昆明", "广州", "上海", "北京", "成都", "深圳", "杭州"]
LABS = ["SGS", "中检", "华测"]
CERTS = ["CMA", "CNAS", "ISO17025"]


class DataGenerator:
    """Deterministic generator for the tea-blending demo database."""

    def __init__(self, seed: int = 20261119, scale: float = 1.0):
        self.rng = random.Random(seed)
        self.seed = seed
        self.scale = scale
        self.ids: Dict[str, List[int]] = {}

    # sizes: chosen so that bad orders are visibly slower in SQLite while the
    # whole fixture still builds in a couple of seconds.
    def sizes(self) -> Dict[str, int]:
        s = self.scale
        return {
            "origins": 24,
            "lots": int(6000 * s),
            "inventory": int(9000 * s),
            "warehouses": 12,
            "reports": int(12000 * s),
            "inspectors": 30,
            "blends": 300,
            "blend_lots": int(7000 * s),
            "orders": int(16000 * s),
            "customers": int(2500 * s),
            "shipments": int(15000 * s),
            "suppliers": 60,
            "purchases": int(8000 * s),
            "grades": 4,
            "warehouses_bak": 12,
            "price_history": int(18000 * s),
        }

    # ------------------------------------------------------------------
    def build(self, conn: sqlite3.Connection, drop_existing: bool = False) -> Dict[str, int]:
        if drop_existing:
            existing = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'")]
            for name in existing:
                if name in TABLES:
                    conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        conn.executescript(SCHEMA_SQL)
        counts = self.sizes()
        rows: Dict[str, List[tuple]] = {}
        self._gen_origins(rows, counts)
        self._gen_warehouses(rows, counts)
        self._gen_customers(rows, counts)
        self._gen_inspectors(rows, counts)
        self._gen_blends(rows, counts)
        self._gen_grades(rows)
        self._gen_lots(rows, counts)
        self._gen_inventory(rows, counts)
        self._gen_reports(rows, counts)
        self._gen_blend_lots(rows, counts)
        self._gen_orders(rows, counts)
        self._gen_shipments(rows, counts)
        self._gen_suppliers(rows, counts)
        self._gen_purchases(rows, counts)
        self._gen_price_history(rows, counts)
        counts["warehouses_bak"] = counts["warehouses"]
        rows["warehouses_bak"] = list(rows["warehouses"])
        self._insert(conn, rows)
        self._create_indexes(conn)
        conn.commit()
        return {t: len(rows[t]) for t in TABLES}

    #: indexes a production schema would have on the join keys.  The optimizer
    #: reads them (page counts, index heights) and the execution harness needs
    #: them to make the join order matter.
    INDEXES = [
        ("ix_lots_origin", "lots", "origin_id"),
        ("ix_lots_grade", "lots", "grade"),
        ("ix_inv_lot", "inventory", "lot_id"),
        ("ix_reports_lot", "reports", "lot_id"),
        ("ix_reports_passed", "reports", "passed"),
        ("ix_blendlots_lot", "blend_lots", "lot_id"),
        ("ix_blendlots_blend", "blend_lots", "blend_id"),
        ("ix_orders_cust", "orders", "customer_id"),
        ("ix_orders_blend", "orders", "blend_id"),
        ("ix_orders_status", "orders", "status"),
        ("ix_ship_order", "shipments", "order_id"),
        ("ix_ship_wh", "shipments", "warehouse_id"),
        ("ix_pur_lot", "purchases", "lot_id"),
        ("ix_pur_sup", "purchases", "supplier_id"),
        ("ix_ph_lot", "price_history", "lot_id"),
        ("ix_sup_origin", "suppliers", "origin_id"),
    ]

    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        for name, table, col in self.INDEXES:
            conn.execute(f'CREATE INDEX IF NOT EXISTS "{name}" '
                         f'ON "{table}" ("{col}")')

    def _insert(self, conn: sqlite3.Connection, rows: Dict[str, List[tuple]]) -> None:
        order = ["origins", "warehouses", "customers", "inspectors", "blends",
                 "grades", "lots", "inventory", "reports", "blend_lots",
                 "orders", "shipments", "suppliers", "purchases",
                 "warehouses_bak", "price_history"]
        for t in order:
            data = rows.get(t, [])
            if not data:
                continue
            qs = ",".join("?" * len(data[0]))
            conn.executemany(f'INSERT INTO "{t}" VALUES ({qs})', data)

    # -- sampling --------------------------------------------------------
    def _uniform(self, n: int, k: int) -> List[int]:
        """Uniform key sampling: bounds the fan-out of child tables.

        The child tables of ``lots`` (inventory, reports, price_history) all
        join on ``lot_id``; if each of them were heavily skewed towards the
        same few lots, joining two of them would square the skew and produce
        tens of millions of rows.  Child FKs are therefore near-uniform.
        """
        return [self.rng.randrange(k) for _ in range(n)]

    def _skewed(self, n: int, k: int, alpha: float = 1.6) -> List[int]:
        """Zipf-ish sampling of 0..k-1: a few indices dominate."""
        weights = [1.0 / ((i + 1) ** alpha) for i in range(k)]
        total = sum(weights)
        cum = []
        acc = 0.0
        for w in weights:
            acc += w / total
            cum.append(acc)
        out = []
        for _ in range(n):
            x = self.rng.random()
            lo, hi = 0, k - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            out.append(lo)
        return out

    # -- per-table generators -------------------------------------------
    def _gen_origins(self, rows, counts):
        data = []
        for i in range(1, counts["origins"] + 1):
            province = PROVINCES[i % len(PROVINCES)]
            region = REGIONS[province][i % len(REGIONS[province])]
            altitude = 800 + (i * 137) % 1800
            data.append((i, province, region, altitude))
        rows["origins"] = data

    def _gen_warehouses(self, rows, counts):
        rows["warehouses"] = [
            (i, CITIES[i % len(CITIES)], 50000 + i * 5000)
            for i in range(1, counts["warehouses"] + 1)]

    def _gen_customers(self, rows, counts):
        n = counts["customers"]
        channels = self._skewed(n, len(CHANNELS), alpha=2.2)   # 线上 dominates
        data = []
        for i in range(n):
            channel = CHANNELS[channels[i]]
            # correlation: 线上 customers are mostly 普通/银卡, 钻石 is rare
            if channel == "线上":
                tier = TIERS[0] if self.rng.random() < 0.8 else TIERS[1]
            elif channel == "批发":
                tier = TIERS[1] if self.rng.random() < 0.7 else TIERS[2]
            else:
                tier = TIERS[self.rng.randrange(len(TIERS))]
            data.append((i + 1, channel, CITIES[i % len(CITIES)], tier))
        rows["customers"] = data

    def _gen_inspectors(self, rows, counts):
        rows["inspectors"] = [
            (i, LABS[i % len(LABS)], CERTS[(i // 3) % len(CERTS)])
            for i in range(1, counts["inspectors"] + 1)]

    def _gen_blends(self, rows, counts):
        data = []
        for i in range(1, counts["blends"] + 1):
            style = STYLES[self._skewed(1, len(STYLES), 1.2)[0]]
            data.append((i, f"拼配-{style}-{i:04d}", style,
                         80 + (i * 7) % 400))
        rows["blends"] = data

    def _gen_grades(self, rows):
        rows["grades"] = [("S", "特级", 95.0), ("A", "一级", 85.0),
                          ("B", "二级", 70.0), ("C", "三级", 50.0)]

    def _gen_lots(self, rows, counts):
        n = counts["lots"]
        # origin skew: a handful of origins supply most lots
        origin_idx = self._skewed(n, counts["origins"], alpha=1.3)
        data = []
        for i in range(n):
            origin_id = origin_idx[i] + 1
            # CORRELATION: grade is a function of origin (cheap origins -> low grade)
            # and of weight, so grade and origin_id are far from independent.
            bucket = (origin_id * 7) % 100
            r = self.rng.random()
            if bucket < 20:
                grade = "S" if r < 0.5 else "A"
            elif bucket < 50:
                grade = "A" if r < 0.6 else "B"
            elif bucket < 80:
                grade = "B" if r < 0.7 else "C"
            else:
                grade = "C" if r < 0.6 else "B"
            harvest = self.rng.choice([2020, 2021, 2022, 2023, 2024, 2025])
            weight = round(self.rng.uniform(5, 400), 2)
            price = round(40 + weight * 0.3 + {"S": 300, "A": 150, "B": 60, "C": 10}[grade]
                          * self.rng.uniform(0.8, 1.2), 2)
            data.append((i + 1, origin_id, grade, harvest, weight, price))
        rows["lots"] = data
        self.ids["origins"] = list(range(1, counts["origins"] + 1))

    def _gen_inventory(self, rows, counts):
        n = counts["inventory"]
        lot_idx = self._uniform(n, counts["lots"])
        w_idx = self._skewed(n, counts["warehouses"], alpha=1.5)
        data = []
        for i in range(n):
            qty = round(self.rng.uniform(0, 5000), 2)
            data.append((i + 1, lot_idx[i] + 1, w_idx[i] + 1, qty,
                         round(qty * self.rng.uniform(0, 0.6), 2)))
        rows["inventory"] = data

    def _gen_reports(self, rows, counts):
        n = counts["reports"]
        lot_idx = self._uniform(n, counts["lots"])
        insp_idx = self._skewed(n, counts["inspectors"], alpha=1.8)
        data = []
        for i in range(n):
            lot_id = lot_idx[i] + 1
            # CORRELATION: score and passed both depend on the lot's grade.
            grade = rows["lots"][lot_id - 1][2]
            base = {"S": 94.0, "A": 86.0, "B": 74.0, "C": 58.0}[grade]
            score = round(max(0.0, min(100.0, base + self.rng.gauss(0, 7))), 1)
            passed = 1 if score >= 70 else 0
            data.append((i + 1, lot_id, passed, insp_idx[i] + 1, score))
        rows["reports"] = data

    def _gen_blend_lots(self, rows, counts):
        n = counts["blend_lots"]
        blend_idx = self._uniform(n, counts["blends"])
        lot_idx = self._uniform(n, counts["lots"])
        data = []
        for i in range(n):
            data.append((i + 1, blend_idx[i] + 1, lot_idx[i] + 1,
                         round(self.rng.uniform(0.01, 0.9), 3)))
        rows["blend_lots"] = data

    def _gen_orders(self, rows, counts):
        n = counts["orders"]
        cust_idx = self._skewed(n, counts["customers"], alpha=1.7)
        blend_idx = self._skewed(n, counts["blends"], alpha=1.0)
        status_idx = self._skewed(n, len(STATUSES), alpha=2.5)  # created dominates
        data = []
        for i in range(n):
            # time dimension: newest orders dominate (realistic for a shop)
            r = self.rng.random()
            if r < 0.45:
                year = 2026
            elif r < 0.7:
                year = 2025
            elif r < 0.9:
                year = 2024
            else:
                year = 2023
            month = self.rng.randint(1, 12)
            day = self.rng.randint(1, 28)
            amount = round(self.rng.uniform(100, 80000), 2)
            data.append((i + 1, cust_idx[i] + 1, blend_idx[i] + 1,
                         STATUSES[status_idx[i]],
                         f"{year:04d}-{month:02d}-{day:02d}", amount))
        rows["orders"] = data

    def _gen_shipments(self, rows, counts):
        n = counts["shipments"]
        order_idx = self._uniform(n, counts["orders"])
        w_idx = self._skewed(n, counts["warehouses"], alpha=1.4)
        data = []
        for i in range(n):
            month = self.rng.randint(1, 12)
            day = self.rng.randint(1, 28)
            data.append((i + 1, order_idx[i] + 1, w_idx[i] + 1,
                         f"2026-{month:02d}-{day:02d}",
                         round(self.rng.uniform(10, 2000), 2)))
        rows["shipments"] = data

    def _gen_suppliers(self, rows, counts):
        data = []
        for i in range(1, counts["suppliers"] + 1):
            data.append((i, (i * 5) % counts["origins"] + 1,
                         self.rng.choice([2023, 2024, 2025]),
                         round(self.rng.uniform(1, 5), 2)))
        rows["suppliers"] = data

    def _gen_purchases(self, rows, counts):
        n = counts["purchases"]
        sup_idx = self._uniform(n, counts["suppliers"])
        lot_idx = self._uniform(n, counts["lots"])
        data = []
        for i in range(n):
            qty = round(self.rng.uniform(10, 3000), 2)
            data.append((i + 1, sup_idx[i] + 1, lot_idx[i] + 1, qty,
                         round(qty * self.rng.uniform(10, 90), 2)))
        rows["purchases"] = data

    def _gen_price_history(self, rows, counts):
        n = counts["price_history"]
        lot_idx = self._uniform(n, counts["lots"])
        data = []
        for i in range(n):
            month = f"2026-{self.rng.randint(1, 12):02d}"
            data.append((i + 1, lot_idx[i] + 1, month,
                         round(self.rng.uniform(20, 500), 2)))
        rows["price_history"] = data


def create_database(path: str, seed: int = 20261119, scale: float = 1.0,
                    force: bool = False) -> Dict[str, int]:
    """Create (or rebuild) the demo database.

    ``force=True`` drops and recreates the demo tables, so the command is
    idempotent and can be re-run to reset the data.
    """
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    conn = sqlite3.connect(path)
    try:
        gen = DataGenerator(seed=seed, scale=scale)
        counts = gen.build(conn, drop_existing=force or exists)
    finally:
        conn.close()
    return counts


if __name__ == "__main__":  # pragma: no cover
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "tea.db"
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    counts = create_database(target, scale=scale, force=True)
    for name, n in counts.items():
        print(f"  {name:16s} {n:>8,d}")
    print(f"已生成 {target}")
