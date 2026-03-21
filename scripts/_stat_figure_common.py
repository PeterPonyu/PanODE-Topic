"""Shared utilities for statistical figure generation scripts."""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from matplotlib.patches import Patch  # noqa: F401 – re-exported for convenience


def setup_fonts():
    """Register Arial fonts from the portable font directory resolution."""
    _FONT_CANDIDATES = [
        os.environ.get("PANODE_FONT_DIR", ""),
        str(Path(__file__).resolve().parent.parent / "fonts"),
        str(Path.home() / "Desktop" / "fonts"),
    ]
    font_dir = next(
        (Path(c) for c in _FONT_CANDIDATES if c and Path(c).is_dir()), None
    )
    if font_dir is not None:
        for f in font_dir.glob("Arial*.ttf"):
            fm.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10


# Series -> set of structured model names (for filtering)
SERIES_STRUCTURED = {
    "dpmm": {"DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive"},
    "topic": {"Topic-Base", "Topic-Transformer", "Topic-Contrastive"},
}

# Series -> set of all model names (structured + pure) for that article
SERIES_MODELS = {
    "dpmm": {"DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive",
             "Pure-AE", "Pure-Transformer-AE", "Pure-Contrastive-AE"},
    "topic": {"Topic-Base", "Topic-Transformer", "Topic-Contrastive",
              "Pure-VAE", "Pure-Transformer-VAE", "Pure-Contrastive-VAE"},
}


def color_dpmm(model):
    """Return bar color for a DPMM-series model."""
    if "Pure" in model:
        return "#9ECAE1"
    return "#E6550D"


def color_topic(model):
    """Return bar color for a Topic-series model."""
    if "Pure" in model:
        return "#C7E9C0"
    return "#756BB1"
