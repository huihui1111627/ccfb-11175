"""Tea-blending multi-table join query optimizer.

Public entry points:

* :func:`teaopt.parser.analyze` – SQL -> :class:`teaopt.types.Query`
* :func:`teaopt.search.Optimizer` – join-order search
* :func:`teaopt.execute.PipelineBuilder` – plan -> executable SQLite SQL
* :class:`teaopt.service.QueryService` – end-to-end service used by the UI
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
