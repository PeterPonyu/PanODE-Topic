"""eval_lib.baselines — External baseline model wrappers and registry.

Submodules
----------
registry    EXTERNAL_MODELS dict mapping model names → factory + params
models/     BaseModel interface + 12 model wrapper implementations
"""

from .registry import (
    EXTERNAL_MODELS,
    MODEL_TAXONOMY,
    CLASSICAL_BASELINES,
    DEEP_GRAPH_BASELINES,
    list_external_models)
from .models import BaseModel
