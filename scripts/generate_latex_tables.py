#!/usr/bin/env python
"""Generate LaTeX tables for DPMM and Topic manuscripts from statistical exports."""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = PROJECT_ROOT / "benchmarks" / "benchmark_results" / "statistical_exports"

DPMM_MODELS = [
    "Pure-AE", "Pure-Transformer-AE", "Pure-Contrastive-AE",
    "DPMM-Base", "DPMM-Transformer", "DPMM-Contrastive",
]
TOPIC_MODELS = [
    "Pure-VAE", "Pure-Transformer-VAE", "Pure-Contrastive-VAE",
    "Topic-Base", "Topic-Transformer", "Topic-Contrastive",
]

SERIES_CONFIG = {
    "dpmm": {
        "models": DPMM_MODELS,
        "structured_prefix": "DPMM",
        "output_dir": PROJECT_ROOT / "article" / "dpmm" / "tables",
        "label_prefix": "dpmm",
        "series_name": "DPMM",
    },
    "topic": {
        "models": TOPIC_MODELS,
        "structured_prefix": "Topic",
        "output_dir": PROJECT_ROOT / "article" / "topic" / "tables",
        "label_prefix": "topic",
        "series_name": "Topic",
    },
}

# Core metrics for Wilcoxon filtering
CORE_METRICS = ["NMI", "ARI", "ASW", "DAV"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(val, decimals=3):
    """Format a numeric value to a fixed number of decimal places."""
    if pd.isna(val):
        return "--"
    return f"{val:.{decimals}f}"


def _fmt_int(val):
    """Format a numeric value as an integer."""
    if pd.isna(val):
        return "--"
    return f"{int(val):,}"


def _fmt_pval(val):
    """Format a p-value: scientific notation if <0.001, otherwise 3 decimals."""
    if pd.isna(val):
        return "--"
    if val < 0.001:
        return f"{val:.2e}"
    return f"{val:.3f}"


def _bold(text):
    """Wrap text in LaTeX bold."""
    return r"\textbf{" + str(text) + "}"


def _bold_best(series, lower_is_better=False, decimals=3):
    """Return a list of formatted strings with the best value bolded."""
    vals = series.values
    if lower_is_better:
        best_idx = series.idxmin()
    else:
        best_idx = series.idxmax()
    result = []
    for idx, val in series.items():
        formatted = _fmt(val, decimals)
        if idx == best_idx:
            formatted = _bold(formatted)
        result.append(formatted)
    return result


def _escape_model(name):
    """Escape model names for LaTeX (hyphens are fine, underscores need escaping)."""
    return name.replace("_", r"\_")


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_variant_ranking(series_key, config):
    """Generate individual variant ranking table."""
    df = pd.read_csv(EXPORTS_DIR / "individual_variant_ranking.csv")
    df = df[df["Model"].isin(config["models"])].copy()
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)

    cols_higher = ["NMI", "ARI", "ASW", "Score"]
    cols_lower = ["DAV"]

    # Build formatted columns
    formatted = {"Model": [_escape_model(m) for m in df["Model"]]}
    for col in cols_higher:
        formatted[col] = _bold_best(df[col], lower_is_better=False)
    for col in cols_lower:
        formatted[col + r" $\downarrow$"] = _bold_best(df[col], lower_is_better=True)

    # Assemble column order for display
    display_cols = ["Model", "NMI", "ARI", "ASW", r"DAV $\downarrow$", "Score"]
    n_rows = len(df)

    header = " & ".join(display_cols)
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Individual variant ranking (" + config["series_name"]
                 + r" series).} Models ranked by composite score across 56 datasets."
                 + r" Best value per column in bold.}")
    lines.append(r"\label{tab:" + config["label_prefix"] + r"_variant_ranking}")
    lines.append(r"\begin{tabular}{l" + "c" * (len(display_cols) - 1) + "}")
    lines.append(r"\toprule")
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for i in range(n_rows):
        row_vals = [
            formatted["Model"][i],
            formatted["NMI"][i],
            formatted["ARI"][i],
            formatted["ASW"][i],
            formatted[r"DAV $\downarrow$"][i],
            formatted["Score"][i],
        ]
        lines.append(" & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_wilcoxon_core(series_key, config):
    """Generate core-metric Wilcoxon test table."""
    df = pd.read_csv(EXPORTS_DIR / "pairwise_wilcoxon.csv")

    # Filter to core metrics and relevant structured prefix
    df = df[df["Metric"].isin(CORE_METRICS)].copy()
    df = df[df["Structured"].str.startswith(config["structured_prefix"])].copy()
    df = df.sort_values(["Structured", "Metric"]).reset_index(drop=True)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Wilcoxon signed-rank tests on core metrics ("
                 + config["series_name"]
                 + r" series).} Pairwise structured vs.\ pure comparisons across 56 datasets."
                 + r" W/L/T = wins/losses/ties; Sig.\ at $\alpha{=}0.05$.}")
    lines.append(r"\label{tab:" + config["label_prefix"] + r"_wilcoxon_core}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Structured vs Pure & Metric & W/L/T & $p$-value & Sig. & Cliff's $\delta$ & Effect \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        pair_label = _escape_model(row["Structured"]) + " vs " + _escape_model(row["Pure"])
        metric = row["Metric"]
        wlt = f"{int(row['Wins'])}/{int(row['Losses'])}/{int(row['Ties'])}"
        pval = _fmt_pval(row["p_value"])
        sig = "Yes" if row["Significant_005"] == "Yes" else "No"
        cliff = _fmt(row["Cliffs_delta"], 3)
        effect = row["Effect_size"]
        lines.append(
            f"{pair_label} & {metric} & {wlt} & {pval} & {sig} & {cliff} & {effect} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_knn(series_key, config):
    """Generate kNN downstream accuracy table."""
    df = pd.read_csv(EXPORTS_DIR / "knn_summary_by_model.csv")
    df = df[df["Model"].isin(config["models"])].copy()
    df = df.sort_values("Mean_Acc", ascending=False).reset_index(drop=True)

    # Bold best values
    acc_best = _bold_best(df["Mean_Acc"], lower_is_better=False)
    f1_best = _bold_best(df["Mean_F1"], lower_is_better=False)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{$k$NN downstream classification ("
                 + config["series_name"]
                 + r" series).} Mean accuracy and macro-F1 across "
                 + str(int(df["N_datasets"].iloc[0]))
                 + r" datasets. Best value per column in bold.}")
    lines.append(r"\label{tab:" + config["label_prefix"] + r"_knn}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Accuracy & F1-macro \\")
    lines.append(r"\midrule")

    for i in range(len(df)):
        model = _escape_model(df.iloc[i]["Model"])
        acc = acc_best[i]
        f1 = f1_best[i]
        lines.append(f"{model} & {acc} & {f1} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_runtime(series_key, config):
    """Generate runtime analysis table."""
    df = pd.read_csv(EXPORTS_DIR / "runtime_analysis.csv")
    df = df[df["Model"].isin(config["models"])].copy()
    df = df.sort_values("Mean_Time_s", ascending=True).reset_index(drop=True)

    # Determine best (lowest) for time, sec/epoch, GPU; lowest params
    time_best = _bold_best(df["Mean_Time_s"], lower_is_better=True, decimals=1)
    spe_best = _bold_best(df["Mean_SecPerEpoch"], lower_is_better=True, decimals=4)
    gpu_best = _bold_best(df["Mean_PeakGPU_MB"], lower_is_better=True, decimals=1)
    params_formatted = _bold_best_int(df["NumParams"], lower_is_better=True)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{Runtime analysis ("
                 + config["series_name"]
                 + r" series).} Mean training time, throughput, peak GPU memory,"
                 + r" and parameter count across 56 datasets. Best value per column in bold.}")
    lines.append(r"\label{tab:" + config["label_prefix"] + r"_runtime}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Time (s) & Sec/Epoch & GPU (MB) & Params \\")
    lines.append(r"\midrule")

    for i in range(len(df)):
        model = _escape_model(df.iloc[i]["Model"])
        lines.append(
            f"{model} & {time_best[i]} & {spe_best[i]} & {gpu_best[i]} & {params_formatted[i]} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _bold_best_int(series, lower_is_better=False):
    """Return a list of formatted integer strings with the best value bolded."""
    if lower_is_better:
        best_idx = series.idxmin()
    else:
        best_idx = series.idxmax()
    result = []
    for idx, val in series.items():
        formatted = _fmt_int(val)
        if idx == best_idx:
            formatted = _bold(formatted)
        result.append(formatted)
    return result


def generate_external_winrate(series_key, config):
    """Generate external win-rate summary table."""
    df = pd.read_csv(EXPORTS_DIR / "per_variant_external_summary.csv")
    df = df[df["Model"].isin(config["models"])].copy()
    df = df.sort_values("Core_win_rate", ascending=False).reset_index(drop=True)

    # Bold best (highest) win rate
    wr_best = _bold_best(df["Core_win_rate"], lower_is_better=False)
    sig_best = _bold_best_int(df["Significant_all"], lower_is_better=False)
    core_sig_best = _bold_best_int(df["Core_significant"], lower_is_better=False)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{External win-rate summary ("
                 + config["series_name"]
                 + r" series).} Each variant is compared against 23 external baselines"
                 + r" via Wilcoxon signed-rank tests across 56 datasets."
                 + r" Core win rate = fraction of core-metric comparisons significantly won."
                 + r" Best value per column in bold.}")
    lines.append(r"\label{tab:" + config["label_prefix"] + r"_external_winrate}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Total tests & Sig.\ (all) & Core sig. & Core total & Core win rate \\")
    lines.append(r"\midrule")

    for i in range(len(df)):
        model = _escape_model(df.iloc[i]["Model"])
        total = _fmt_int(df.iloc[i]["Total_tests"])
        sig_all = sig_best[i]
        core_sig = core_sig_best[i]
        core_total = _fmt_int(df.iloc[i]["Core_total"])
        wr = wr_best[i]
        lines.append(
            f"{model} & {total} & {sig_all} & {core_sig} & {core_total} & {wr} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TABLE_GENERATORS = {
    "variant_ranking.tex": generate_variant_ranking,
    "wilcoxon_core.tex": generate_wilcoxon_core,
    "knn.tex": generate_knn,
    "runtime.tex": generate_runtime,
    "external_winrate.tex": generate_external_winrate,
}


def main():
    # Verify exports directory exists
    if not EXPORTS_DIR.is_dir():
        print(f"ERROR: Exports directory not found: {EXPORTS_DIR}", file=sys.stderr)
        sys.exit(1)

    for series_key, config in SERIES_CONFIG.items():
        out_dir = config["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Generating tables for {config['series_name']} series -> {out_dir}")
        print(f"{'='*60}")

        for filename, generator in TABLE_GENERATORS.items():
            content = generator(series_key, config)
            out_path = out_dir / filename
            out_path.write_text(content + "\n", encoding="utf-8")
            print(f"  wrote {out_path.relative_to(PROJECT_ROOT)}")

    print("\nDone. All tables generated successfully.")


if __name__ == "__main__":
    main()
