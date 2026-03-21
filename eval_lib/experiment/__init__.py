"""eval_lib.experiment — Experiment infrastructure and templates.

Importable modules
------------------
config      ExperimentConfig dataclass
merge       MergedExperimentConfig (cross-experiment result merger)

Templates (copy to project ``experiments/`` and customise)
----------------------------------------------------------
templates/run_experiment.py
templates/visualize_experiment.py
"""

from .config import ExperimentConfig
from .merge import MergedExperimentConfig
