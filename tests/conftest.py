"""Shared fixtures for the test-suite.

The tests build the demo database once per session into a temporary directory
and share read-only connections plus the collected statistics.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from typing import Dict, Iterator

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from teaopt.demo import MULTICOLUMN, TABLES, create_database  # noqa: E402
from teaopt.stats import collect  # noqa: E402
from teaopt.types import TableStats  # noqa: E402


@pytest.fixture(scope="session")
def db_path(tmp_path_factory) -> str:
    path = str(tmp_path_factory.mktemp("db") / "tea.db")
    create_database(path, seed=20261119, scale=1.0)
    return path


@pytest.fixture(scope="session")
def conn(db_path) -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(db_path)
    c.execute("PRAGMA cache_size = -64000")
    yield c
    c.close()


@pytest.fixture(scope="session")
def stats(conn) -> Dict[str, TableStats]:
    return collect(conn, TABLES, MULTICOLUMN)


@pytest.fixture()
def fresh_conn(db_path) -> Iterator[sqlite3.Connection]:
    c = sqlite3.connect(db_path)
    yield c
    c.close()


@pytest.fixture(scope="session")
def service(db_path, stats):
    """A QueryService bound to the shared demo database."""
    from teaopt.service import QueryService
    from teaopt.demo import MULTICOLUMN
    return QueryService(db_path, MULTICOLUMN, TABLES)
