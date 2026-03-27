"""Generate individual subplot PNGs for Figure 9 (GO Enrichment).

Produces:
  Panel A — Workflow
  Panel B — GO enrichment dot plots (datasets × models)

Enrichment data is loaded from the same bio-validation results as Figure 6.
Term descriptions and titles use a consistent font size.
Panel spacing is enforced to prevent overlap.

Output: benchmarks/paper_figures/{series}/subplots/fig9/

Usage:
    python -m benchmarks.figure_generators.gen_fig9_subplots --series dpmm
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.visualization import (
    apply_style, style_axes, add_panel_label, save_with_vcd,
    bind_figure_region, LayoutRegion)

from benchmarks.figure_generators.subplot_style import (
    apply_subplot_style, save_subplot, build_manifest,
    FIGSIZE_ENRICHMENT, SUBPLOT_DPI,
    FONTSIZE_TITLE, FONTSIZE_TICK, FONTSIZE_LABEL,
    FONTSIZE_CAPTION)
from benchmarks.figure_generators.common import (
    MODEL_SHORT_NAMES,
    PRIOR_MODELS_DPMM, PRIOR_MODELS_TOPIC,
    REPRESENTATIVE_DATASETS, BIO_RESULTS
)

_COMP_PALETTE = [
    "#4E79A7", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1",
    "#9C755F", "#BAB0AC", "#86BCB6", "#A0CBE8", "#CFCFCF",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Bio data loading (shared with Fig 6)
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_bio_paths(model, dataset, results_dir):
    tag = f"{model}_{dataset}"
    candidates = []
    model_dir = results_dir / model.replace("/", "_")
    if model_dir.is_dir():
        candidates.append(model_dir)
    candidates.append(results_dir)
    for d in candidates:
        lp = d / f"{tag}_latent_data.npz"
        ip = d / f"{tag}_importance.npz"
        if lp.exists() and ip.exists():
            return lp, ip, d
    return None, None, None


def _load_dataset_bio(models, dataset, results_dir):
    model_data = {}
    for model in models:
        latent_path, importance_path, bio_dir = _resolve_bio_paths(
            model, dataset, results_dir)
        tag = f"{model}_{dataset}"
        if latent_path is None:
            continue
        try:
            ld = np.load(latent_path, allow_pickle=True)
            imp = np.load(importance_path, allow_pickle=True)
            importance = imp["importance"]
            components = ld.get("components")
            K_total = (components.shape[1] if components is not None
                       else importance.shape[0])
            enrichments = {}
            for k in range(K_total):
                csv_path = bio_dir / f"{tag}_enrichment_comp{k}.csv"
                if csv_path.exists():
                    edf = pd.read_csv(csv_path)
                    if len(edf) > 0:
                        enrichments[k] = edf
            model_data[model] = {
                "importance": importance,
                "enrichments": enrichments,
            }
        except Exception:
            pass
    return model_data if model_data else None


def _discover_bio_datasets(models, results_dir):
    found = {}
    for ds in REPRESENTATIVE_DATASETS:
        md = _load_dataset_bio(models, ds, results_dir)
        if md:
            found[ds] = md
    return found


def _load_beta_enrichments(models, results_dir):
    """Load beta-based enrichment CSVs for Topic models.

    Returns ``{dataset: {model: {k: DataFrame, ...}, ...}, ...}``
    Only populated for Topic models that have ``*_beta_enrich_comp{k}.csv``.
    """
    found = {}
    for ds in REPRESENTATIVE_DATASETS:
        ds_data = {}
        for model in models:
            if not model.startswith("Topic"):
                continue
            tag = f"{model}_{ds}"
            candidates = [
                results_dir / model,
                results_dir,
            ]
            enrichments = {}
            for parent in candidates:
                for k in range(20):  # up to 20 topics
                    csv_path = parent / f"{tag}_beta_enrich_comp{k}.csv"
                    if csv_path.exists():
                        edf = pd.read_csv(csv_path)
                        if len(edf) > 0:
                            enrichments[k] = edf
                if enrichments:
                    break
            if enrichments:
                ds_data[model] = {"enrichments": enrichments}
        if ds_data:
            found[ds] = ds_data
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# Enrichment dotplot generator (improved spacing & consistent fonts)
# ═══════════════════════════════════════════════════════════════════════════════

def _wrap_term(text, max_chars=28):
    """Wrap a long GO term description into two lines at a word boundary.

    If the text is longer than *max_chars*, splits at the nearest space
    before the limit and joins the halves with ``\\n``.  This saves
    horizontal space in the dot-plot y-axis labels.
    """
    if len(text) <= max_chars:
        return text
    # Try to split near the midpoint at a word boundary
    mid = len(text) // 2
    # Search outward from the midpoint for a space
    left = text.rfind(" ", 0, mid + 5)
    right = text.find(" ", mid - 5)
    if left == -1 and right == -1:
        return text  # no good split point
    # Pick the split closest to midpoint
    if left == -1:
        split = right
    elif right == -1:
        split = left
    else:
        split = left if (mid - left) <= (right - mid) else right
    return text[:split] + "\n" + text[split + 1:]


def gen_enrichment_dotplot(md, comp_prefix, ds_name, model, out_path,
                           top_n=10):
    """Generate one GO enrichment dot plot subplot PNG.

    Layout:
    - GO term labels are placed as y-axis tick labels (right-aligned)
      to avoid collision with scatter dots and ensure VCD-safe spacing.
    - Dots are never clipped (``clip_on=False``).
    - Title is left-aligned to conserve header space.
    - Long GO term names are wrapped into two lines at word boundaries.
    - Vertical spacing is increased to prevent text_overlap between
      adjacent GO term labels.
    """
    short = MODEL_SHORT_NAMES.get(model, model)
    enrichments = md.get("enrichments", {})

    all_terms = []
    for k, edf in enrichments.items():
        if "Adjusted P-value" not in edf.columns or "Term" not in edf.columns:
            continue
        for _, row in edf.head(6).iterrows():
            pval = row["Adjusted P-value"]
            if pval <= 0 or pval >= 0.05:
                continue
            term_raw = str(row["Term"])
            term_short = term_raw.split("(GO:")[0].strip()
            overlap_str = str(row.get("Overlap", "0/1"))
            try:
                gene_count = int(overlap_str.split("/")[0])
            except (ValueError, IndexError):
                gene_count = 1
            all_terms.append({
                "term": term_short, "comp": k,
                "nlp": -np.log10(max(pval, 1e-30)),
                "pval": pval, "gene_count": gene_count,
            })

    # Wider + taller figure to prevent xlabel/xtick truncation at bottom
    # Height scaled proportionally to term count
    fig_w = FIGSIZE_ENRICHMENT[0] * 2.0
    y_spacing = 1.4  # increased from 1.0 to prevent label overlap
    fig_h = max(FIGSIZE_ENRICHMENT[1] * 1.3 * (top_n / 10),
                y_spacing * top_n * 0.22 + 1.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    layout = bind_figure_region(fig, (0.08, 0.10, 0.95, 0.92))
    ax = layout.add_axes(fig)
    style_axes(ax)

    if not all_terms:
        ax.text(0.5, 0.5, "No significant terms",
                transform=ax.transAxes, ha="center",
                fontsize=FONTSIZE_TITLE)
        ax.set_title(f"{short} — {ds_name}",
                     fontsize=FONTSIZE_TITLE, loc="left",
                     fontweight="normal")
        save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)
        return

    best = {}
    for t in all_terms:
        key = t["term"]
        if key not in best or t["nlp"] > best[key]["nlp"]:
            best[key] = t
    term_list = sorted(best.values(), key=lambda x: -x["nlp"])[:top_n]

    y_pos = np.arange(len(term_list)) * y_spacing
    nlps = [t["nlp"] for t in term_list]
    gcs = [t["gene_count"] for t in term_list]
    comp_ids = [t["comp"] for t in term_list]
    comp_colors = [_COMP_PALETTE[c % len(_COMP_PALETTE)] for c in comp_ids]
    labels = [_wrap_term(t["term"], max_chars=32) for t in term_list]

    mn, mx = max(min(gcs), 1), max(gcs)
    sizes = ([15 + 100 * (g - mn) / (mx - mn) for g in gcs]
             if mx > mn else [50] * len(gcs))

    # ── Scatter (never clipped) ─────────────────────────────────────────
    ax.scatter(nlps, y_pos, s=sizes, c=comp_colors,
               edgecolors="white", linewidth=0.4, alpha=0.88,
               zorder=3, clip_on=False)

    # ── Place term labels right beside each dot as text annotations ─────
    term_fs = FONTSIZE_TICK
    # Compute a small offset in data coordinates (~2% of x range)
    x_max_val = max(nlps) if nlps else 1.0
    x_offset = max(x_max_val * 0.03, 0.15)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (nlps[i] + x_offset, y_pos[i]),
                    fontsize=term_fs, color="#333333",
                    va="center", ha="left", linespacing=0.90,
                    annotation_clip=False)

    # Hide y-axis ticks/labels — terms are now inline annotations
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', length=0, pad=0)
    ax.spines["left"].set_visible(False)
    ax.invert_yaxis()

    # ── X-axis ──────────────────────────────────────────────────────────
    x_fs = FONTSIZE_LABEL
    ax.set_xlabel("$-\\log_{10}$(adj. p)", fontsize=x_fs)
    ax.set_title(f"{short} — {ds_name}", fontsize=FONTSIZE_TITLE,
                 loc="left", fontweight="normal", pad=4)
    ax.tick_params(axis='x', labelsize=x_fs, rotation=0)
    ax.axvline(-np.log10(0.05), color="gray", ls="--", lw=0.6, alpha=0.6)
    ax.grid(axis="x", alpha=0.2, linewidth=0.4)

    # Prune the uppermost x-tick so no label sits at the axes right edge
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))

    # y-axis padding
    n_terms = len(term_list)
    ax.set_ylim(y_pos[-1] + 0.8, y_pos[0] - 0.8)

    # x-axis padding — generous right margin for inline term annotations
    xlim = ax.get_xlim()
    x_rng = xlim[1] - xlim[0]
    pad_right = max(x_rng * 0.65, 2.0)  # more room for annotation text
    ax.set_xlim(xlim[0], xlim[1] + pad_right)

    # Do NOT call subplots_adjust here — let tight_layout in save_subplot
    # handle margins.  The wider figure + generous xlim padding ensures
    # all text fits within the canvas.

    save_with_vcd(fig, out_path, dpi=SUBPLOT_DPI, close=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate(out_dir):
    """Generate all subplot PNGs for Figure 9 (GO Enrichment)."""
    series = "topic"
    print(f"\n  Figure 9 subplots ({series})")
    sub_dir = out_dir / "fig9"
    sub_dir.mkdir(parents=True, exist_ok=True)
    apply_subplot_style()

    models = PRIOR_MODELS_TOPIC if series == "topic" else PRIOR_MODELS_DPMM
    comp_prefix = "Topic" if series == "topic" else "Dim"
    results_dir = BIO_RESULTS

    bio_datasets = _discover_bio_datasets(models, results_dir)
    if not bio_datasets:
        print("    No bio data found.")
        return build_manifest(sub_dir, {})

    avail_models = [m for m in models
                    if any(m in bio_datasets.get(ds, {}) for ds in bio_datasets)]

    # Panel B — enrichment dot plots (perturbation-based)
    # Topic: top_n=7 (compact; 9 plots across 3 datasets × 3 models)
    # DPMM:  top_n=10
    perturb_top_n = 5 if series == "topic" else 10
    enrich_files = {}
    for ds_name in bio_datasets:
        ds_enrich = []
        for model in avail_models:
            md = bio_datasets[ds_name].get(model)
            if md is None:
                continue
            safe_m = model.replace("/", "_")
            fname = f"enrich_{ds_name}_{safe_m}.png"
            gen_enrichment_dotplot(md, comp_prefix, ds_name, model,
                                  sub_dir / fname,
                                  top_n=perturb_top_n)
            ds_enrich.append({"file": fname, "model": model})
        enrich_files[ds_name] = ds_enrich

    # ── Panel C — beta-based enrichment (Topic series, best variant only) ──
    beta_enrich_files = {}
    if series == "topic":
        # Only show the best-performing Topic variant's beta enrichment
        _BEST_TOPIC_VARIANT = "Topic-FM-Transformer"
        beta_data = _load_beta_enrichments(models, results_dir)
        for ds_name, ds_models in beta_data.items():
            ds_beta = []
            md = ds_models.get(_BEST_TOPIC_VARIANT)
            if md is None:
                continue
            safe_m = _BEST_TOPIC_VARIANT.replace("/", "_")
            fname = f"beta_enrich_{ds_name}_{safe_m}.png"
            gen_enrichment_dotplot(md, "Topic", ds_name, _BEST_TOPIC_VARIANT,
                                  sub_dir / fname,
                                  top_n=10)
            ds_beta.append({"file": fname, "model": _BEST_TOPIC_VARIANT})
            if ds_beta:
                beta_enrich_files[ds_name] = ds_beta

    manifest = build_manifest(sub_dir, {
        "panelA": enrich_files,
        "panelBeta": beta_enrich_files,
        "models": avail_models,
        "datasets": list(bio_datasets.keys()),
    })
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Figure 9 subplots")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    series = "topic"
    out = (Path(args.output_dir) if args.output_dir
           else ROOT / "benchmarks" / "paper_figures" / "topic" / "subplots")
    out.mkdir(parents=True, exist_ok=True)
    generate(out)
