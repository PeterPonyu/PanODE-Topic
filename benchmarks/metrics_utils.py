"""Metrics computation for benchmark evaluation.

**Thin compatibility shim** — the canonical implementation now lives in
``eval_lib.metrics.battery``.  This module re-exports the public API so
that existing ``from benchmarks.metrics_utils import …`` statements
continue to work without modification.

Re-exports
----------
compute_metrics            — clustering + DRE + LSE + DREX + LSEX
compute_latent_diagnostics — latent collapse / redundancy stats
convergence_diagnostics    — training convergence from loss history
METRIC_COLUMNS             — canonical column order (38 columns)
METRIC_GROUPS              — metric grouping for visualisation
PUBLICATION_METRIC_GROUPS  — K_max-excluded groups for publication figures
"""

import sys
from pathlib import Path

# Ensure the project root is importable (benchmarks/ may be run standalone)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval_lib.metrics.battery import (          # noqa: F401 — re-export
    compute_metrics,
    compute_latent_diagnostics,
    convergence_diagnostics,
    DataSplitter,
    METRIC_COLUMNS,
    METRIC_GROUPS,
    PUBLICATION_METRIC_GROUPS)

# Backward-compat alias — old code references the underscore-prefixed name
_convergence_diagnostics = convergence_diagnostics
