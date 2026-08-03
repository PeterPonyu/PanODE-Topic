#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
  library(gtable)
  library(tikzDevice)
})

`%||%` <- function(x, y) if (is.null(x) || length(x) == 0 || is.na(x)) y else x

args_file <- commandArgs(trailingOnly = FALSE)
file_arg <- args_file[grep("^--file=", args_file)][1] %||% "scripts/build_std_figures.R"
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg), mustWork = FALSE))
repo_root <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
source(file.path(script_dir, "figure_standard.R"))

out_dir <- file.path(repo_root, "article", "topic", "figures_std")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

dims <- data.frame(
  old_target = character(),
  new_target = character(),
  plot_w_in = numeric(),
  plot_h_in = numeric(),
  canvas_w_in = numeric(),
  canvas_h_in = numeric(),
  stringsAsFactors = FALSE
)
contradictions <- character()

record_dim <- function(old_target, new_file, fitted, plot_w, plot_h) {
  dims[nrow(dims) + 1, ] <<- list(
    old_target = old_target,
    new_target = paste0("figures_std/", basename(new_file)),
    plot_w_in = plot_w,
    plot_h_in = plot_h,
    canvas_w_in = fitted$w_in,
    canvas_h_in = fitted$h_in
  )
}

save_std <- function(plot, old_target, name, plot_w, plot_h) {
  path <- file.path(out_dir, paste0(name, ".tex"))
  fitted <- save_tikz_std(plot, path, plot_w_in = plot_w, plot_h_in = plot_h,
                          base_size = STD_BASE_SIZE, sanitize = TRUE)
  record_dim(old_target, path, fitted, plot_w, plot_h)
  invisible(path)
}

read_metric_tables <- function(subdir = "ablation") {
  tables_dir <- file.path(repo_root, "experiments", "results", "topic", subdir, "tables")
  files <- sort(list.files(tables_dir, pattern = "_df[.]csv$", full.names = TRUE))
  if (!length(files)) stop("No CSV metric tables found in ", tables_dir)
  rows <- lapply(files, function(path) {
    x <- read.csv(path, check.names = FALSE)
    x$dataset <- sub("_df$", "", tools::file_path_sans_ext(basename(path)))
    x
  })
  do.call(rbind, rows)
}

method_levels <- c(
  "Pure-VAE", "Pure-Transformer-VAE", "Pure-Contrastive-VAE",
  "Topic-FM-Base", "Topic-FM-Transformer", "Topic-FM-Contrastive"
)
method_labels <- c(
  "Pure-VAE" = "Pure\nVAE",
  "Pure-Transformer-VAE" = "Pure\nTrans",
  "Pure-Contrastive-VAE" = "Pure\nContr",
  "Topic-FM-Base" = "TFM\nBase",
  "Topic-FM-Transformer" = "TFM\nTrans",
  "Topic-FM-Contrastive" = "TFM\nContr"
)
method_cols <- c(
  "Pure-VAE" = "#0072B2",
  "Pure-Transformer-VAE" = "#E69F00",
  "Pure-Contrastive-VAE" = "#009E73",
  "Topic-FM-Base" = "#CC79A7",
  "Topic-FM-Transformer" = "#D55E00",
  "Topic-FM-Contrastive" = "#56B4E9"
)

metric_specs <- data.frame(
  metric = c("NMI", "ARI", "ASW", "DAV", "CAL", "COR",
             "DRE_umap_distance_correlation", "DRE_umap_Q_local", "DRE_umap_Q_global", "DRE_umap_overall_quality", "DRE_umap_k_max",
             "DRE_tsne_distance_correlation", "DRE_tsne_Q_local", "DRE_tsne_Q_global", "DRE_tsne_overall_quality", "DRE_tsne_k_max",
             "LSE_anisotropy_score", "LSE_core_quality", "LSE_manifold_dimensionality", "LSE_noise_resilience", "LSE_overall_quality", "LSE_participation_ratio", "LSE_spectral_decay_rate", "LSE_trajectory_directionality",
             "DREX_continuity", "DREX_distance_pearson", "DREX_distance_spearman", "DREX_local_scale_quality", "DREX_neighborhood_symmetry", "DREX_overall_quality", "DREX_trustworthiness",
             "LSEX_entropy_stability", "LSEX_local_curvature", "LSEX_overall_quality", "LSEX_radial_concentration", "LSEX_two_hop_connectivity"),
  label = c("NMI", "ARI", "ASW", "DAV↓", "CAL", "COR",
            "UMAP dist.", "UMAP local", "UMAP global", "UMAP overall", "UMAP k-max",
            "t-SNE dist.", "t-SNE local", "t-SNE global", "t-SNE overall", "t-SNE k-max",
            "Anisotropy", "Core qual.", "Manifold dim.", "Noise resil.", "LSE overall", "Participation", "Spectral decay", "Trajectory",
            "DREX cont.", "DREX Pearson", "DREX Spearman", "DREX local", "DREX symmetry", "DREX overall", "DREX trust",
            "LSEX entropy", "LSEX curvature", "LSEX overall", "LSEX radial", "LSEX 2-hop"),
  suite = c(rep("Clustering", 6), rep("DRE-UMAP", 5), rep("DRE-tSNE", 5), rep("LSE", 8), rep("DREX", 7), rep("LSEX", 5)),
  lower_better = c(FALSE, FALSE, FALSE, TRUE, FALSE, FALSE,
                   FALSE, FALSE, FALSE, FALSE, FALSE,
                   FALSE, FALSE, FALSE, FALSE, FALSE,
                   rep(FALSE, 8), rep(FALSE, 7), rep(FALSE, 5)),
  stringsAsFactors = FALSE
)

make_architecture <- function() {
  blocks <- data.frame(
    variant = factor(c("(a) Base", "(b) Transformer", "(c) Contrastive", "(d) GAT"),
                     levels = c("(a) Base", "(b) Transformer", "(c) Contrastive", "(d) GAT")),
    encoder = c("MLP\nencoder", "Self-attention\nencoder", "MoCo\ncontrastive head", "Graph attention\nencoder"),
    prior = "Dirichlet topic VAE\nK = 10 simplex",
    flow = "Flow matching\npre-softmax RK",
    decoder = "Decoder beta\nprograms",
    stringsAsFactors = FALSE
  )
  nodes <- do.call(rbind, lapply(seq_len(nrow(blocks)), function(i) {
    data.frame(
      variant = blocks$variant[i],
      stage = factor(c("Input", "Encoder", "Topic", "Flow", "Decoder"),
                     levels = c("Input", "Encoder", "Topic", "Flow", "Decoder")),
      label = c("Cells x genes", blocks$encoder[i], blocks$prior[i], blocks$flow[i], blocks$decoder[i]),
      x = 1:5,
      y = 5 - i,
      stringsAsFactors = FALSE
    )
  }))
  edges <- do.call(rbind, lapply(split(nodes, nodes$variant), function(d) {
    data.frame(x = d$x[-nrow(d)] + 0.33, xend = d$x[-1] - 0.33,
               y = d$y[-nrow(d)], yend = d$y[-1])
  }))
  ggplot() +
    geom_segment(data = edges, aes(x = x, xend = xend, y = y, yend = yend),
                 arrow = arrow(length = unit(0.08, "in")), linewidth = 0.35, color = "#4B5563") +
    geom_label(data = nodes, aes(x = x, y = y, label = label, fill = stage),
               linewidth = 0.22, label.r = unit(0.05, "in"), size = 2.1, lineheight = 0.85,
               color = "#111827") +
    scale_fill_manual(values = c("Input" = "#F3F4F6", "Encoder" = "#DBEAFE", "Topic" = "#D1FAE5", "Flow" = "#FCE7F3", "Decoder" = "#FEF3C7"), guide = "none") +
    scale_x_continuous(breaks = 1:5, labels = c("Input", "Encoder", "Topic posterior", "Refinement", "Readout"), expand = expansion(mult = c(0.09, 0.12))) +
    scale_y_continuous(breaks = 1:4, labels = rev(levels(blocks$variant)), expand = expansion(mult = c(0.08, 0.08))) +
    labs(x = NULL, y = NULL) +
    theme(panel.grid.minor = element_blank(), panel.grid.major.y = element_blank(),
          axis.text.y = element_text(face = "bold"), axis.ticks = element_blank(),
          legend.position = "none")
}

make_crossdataset <- function(all_df) {
  metrics <- c("NMI", "ARI", "ASW", "DAV")
  keep <- all_df[all_df$method %in% c("Pure-VAE", "Topic-FM-Base", "Topic-FM-Transformer", "Topic-FM-Contrastive"), ]
  pairs <- list(c("NMI", "ASW"), c("NMI", "DAV"), c("ARI", "ASW"), c("ARI", "DAV"))
  out <- do.call(rbind, lapply(pairs, function(pair) {
    d <- keep[, c("dataset", "method", pair)]
    names(d)[3:4] <- c("x", "y")
    d$panel <- paste(pair[1], "vs", pair[2])
    d$xlab <- pair[1]
    d$ylab <- pair[2]
    d
  }))
  out$method <- factor(out$method, levels = c("Pure-VAE", "Topic-FM-Base", "Topic-FM-Transformer", "Topic-FM-Contrastive"))
  ggplot(out, aes(x = x, y = y, color = method)) +
    geom_point(alpha = 0.65, size = 1.2) +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.35, alpha = 0.8) +
    facet_wrap(~panel, scales = "free", ncol = 2) +
    scale_color_manual(values = method_cols, labels = method_labels, drop = FALSE, name = "Model") +
    labs(x = "Concordance metric", y = "Geometry metric") +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())
}

make_uniform_grid <- function(all_df) {
  available <- metric_specs[metric_specs$metric %in% names(all_df), ]
  keep <- all_df[all_df$method %in% c("Pure-VAE", "Topic-FM-Base", "Topic-FM-Transformer", "Topic-FM-Contrastive"), ]
  long <- do.call(rbind, lapply(seq_len(nrow(available)), function(i) {
    m <- available$metric[i]
    data.frame(
      dataset = keep$dataset,
      method = keep$method,
      value = keep[[m]],
      metric = available$label[i],
      suite = available$suite[i],
      stringsAsFactors = FALSE
    )
  }))
  long <- long[is.finite(long$value), ]
  long$method <- factor(long$method, levels = c("Pure-VAE", "Topic-FM-Base", "Topic-FM-Transformer", "Topic-FM-Contrastive"))
  long$metric <- factor(long$metric, levels = unique(available$label))
  grid_labels <- c("Pure-VAE" = "PV", "Topic-FM-Base" = "Base",
                   "Topic-FM-Transformer" = "Trans", "Topic-FM-Contrastive" = "Contr")
  ggplot(long, aes(x = method, y = value, fill = method)) +
    geom_boxplot(width = 0.58, outlier.size = 0.25, linewidth = 0.18) +
    stat_summary(fun = median, geom = "point", shape = 23, size = 0.75, fill = "white", color = "#111827", stroke = 0.2) +
    facet_wrap(~metric, scales = "free_y", ncol = 6) +
    scale_x_discrete(labels = grid_labels) +
    scale_fill_manual(values = method_cols, guide = "none") +
    labs(x = NULL, y = "Metric value") +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1, size = 4.4),
          strip.text = element_text(size = 5.8),
          panel.spacing = unit(0.065, "in"),
          panel.grid.minor = element_blank())
}

check_contradictions <- function(all_df) {
  core <- c("NMI", "ARI", "ASW", "DAV")
  means <- aggregate(all_df[, core], list(method = all_df$method), mean, na.rm = TRUE)
  pv <- means[means$method == "Pure-VAE", ]
  tfm <- means[means$method %in% c("Topic-FM-Base", "Topic-FM-Transformer", "Topic-FM-Contrastive"), ]
  if (nrow(pv) == 1 && nrow(tfm) > 0) {
    for (m in c("NMI", "ARI")) {
      best <- tfm$method[which.max(tfm[[m]])]
      best_val <- max(tfm[[m]], na.rm = TRUE)
      if (is.finite(pv[[m]]) && is.finite(best_val) && pv[[m]] > best_val) {
        contradictions <<- c(contradictions, sprintf("Internal ablation CSVs: Pure-VAE mean %s %.3f exceeds best Topic-FM mean %.3f (%s), contradicting text/table claims of Topic-FM improvement on every core metric.", m, pv[[m]], best_val, best))
      }
    }
  }
}

all_df <- read_metric_tables("ablation")
check_contradictions(all_df)

save_std(make_architecture(), "Fig1_arch_topic", "Fig1_arch_topic_std", 6.65, 3.00)
save_std(make_crossdataset(all_df), "Fig5_crossdataset_topic", "Fig5_crossdataset_topic_std", 2.25, 1.65)
save_std(make_uniform_grid(all_df), "ablation/figures/uniform_grid", "uniform_grid_topic_std", 0.74, 1.05)

write.csv(dims, file.path(out_dir, "figure_dimensions.csv"), row.names = FALSE)
writeLines(contradictions, file.path(out_dir, "figure_contradictions.txt"))
print(dims)
if (length(contradictions)) {
  message("Contradictions flagged:")
  message(paste(contradictions, collapse = "\n"))
}
