"""eval_lib.viz — Publication-quality figure generation from experiment results.

Submodules
----------
rea      RigorousExperimentalAnalyzer — statistical comparison figures
loss     Training / validation loss curve visualisation
layout   Centralised layout constants and adaptive sizing helpers
"""

from .rea import (
    RigorousExperimentalAnalyzer,
    create_publication_figure,
    _apply_font)
from .loss import (
    plot_training_curves,
    plot_aggregated_loss)
from .layout import (
    MIN_XTICK_FONTSIZE,
    MIN_XTICK_FONTSIZE_PER_GROUP,
    FIG_WIDTH_PER_METRIC,
    MAX_METHODS_PER_FIGURE,
    MAX_HEIGHT_TO_WIDTH,
    MIN_HEIGHT_TO_WIDTH,
    clamp_xtick_fontsize,
    needs_method_split,
    clamp_aspect_ratio,
    assert_no_label_overlap)
