## figure_standard.R — 统一 figure 标准(2026-08-03)
##
## 核心规则(用户指定):每个 panel 的画布宽高由该 panel 内文本的最大延展决定。
## 实现方式:plot 区域设为绝对尺寸,边距由 gtable 实测(y 轴 tick 最宽标签、
## 轴标题、旋转 x 标签、注释、panel tag 的真实渲染宽度)——画布 = plot 区域 +
## 文本最大延展,不裁剪也不多留。
##
## 用法:
##   source("figure_standard.R")
##   save_tikz_std(p, "fig_x.tex", plot_w_in = 2.4, plot_h_in = 1.8)
##   save_pdf_std(p, "fig_x.pdf", plot_w_in = 2.4, plot_h_in = 1.8)  # markdown 手稿用
##
## 约定:base font 8pt;单栏 panel 目标宽 2.2-2.6in,双栏通栏图按手稿栏宽缩放;
## sanitize=TRUE(tikz);输出 tex 用 \\input{},pdf 用 \\includegraphics。

library(ggplot2)
library(grid)

STD_BASE_SIZE <- 8  # pt,全组合统一

std_theme <- function(base_size = STD_BASE_SIZE) {
  theme_bw(base_size = base_size) +
    theme(
      plot.tag        = element_text(size = base_size + 2, face = "bold"),
      plot.tag.position = "topleft",
      plot.title      = element_text(size = base_size + 1),
      axis.title      = element_text(size = base_size),
      axis.text       = element_text(size = base_size - 1),
      legend.text     = element_text(size = base_size - 1),
      legend.title    = element_text(size = base_size),
      strip.text      = element_text(size = base_size),
      plot.margin     = margin(2, 2, 2, 2)
    )
}

## --- 内部:把 gtable 的 panel 区域锁成绝对尺寸 --------------------------
.lock_panel_size <- function(g, plot_w_in, plot_h_in) {
  is_panel <- grepl("^panel", g$layout$name)
  if (!any(is_panel)) stop("no panel cell found — pass a ggplot, not a composed object")
  cols <- unique(g$layout$l[is_panel])
  rows <- unique(g$layout$t[is_panel])
  g$widths[cols]  <- unit(plot_w_in, "in")
  g$heights[rows] <- unit(plot_h_in, "in")
  g
}

## --- 实测:画布尺寸 = panel 绝对尺寸 + 全部文本延展 ---------------------
measure_canvas <- function(g) {
  list(
    w_in = convertWidth(sum(g$widths), "in", valueOnly = TRUE) + 0.08,
    h_in = convertHeight(sum(g$heights), "in", valueOnly = TRUE) + 0.08
  )
}

## 返回 list(grob, w_in, h_in);patchwork 组合请先对子图分别 fit 再 wrap
fit_panel <- function(p, plot_w_in, plot_h_in, base_size = STD_BASE_SIZE) {
  g <- ggplotGrob(p + std_theme(base_size))
  g <- .lock_panel_size(g, plot_w_in, plot_h_in)
  dims <- measure_canvas(g)
  c(list(grob = g), dims)
}

save_tikz_std <- function(p, file, plot_w_in, plot_h_in, base_size = STD_BASE_SIZE,
                          sanitize = TRUE) {
  fitted <- fit_panel(p, plot_w_in, plot_h_in, base_size)
  tikzDevice::tikz(file, width = fitted$w_in, height = fitted$h_in,
                   sanitize = sanitize, standAlone = FALSE)
  grid.draw(fitted$grob)
  dev.off()
  message(sprintf("%s: panel %.2fx%.2f in -> canvas %.2fx%.2f in",
                  basename(file), plot_w_in, plot_h_in, fitted$w_in, fitted$h_in))
  invisible(fitted)
}

save_pdf_std <- function(p, file, plot_w_in, plot_h_in, base_size = STD_BASE_SIZE) {
  fitted <- fit_panel(p, plot_w_in, plot_h_in, base_size)
  pdf(file, width = fitted$w_in, height = fitted$h_in, useDingbats = FALSE)
  grid.draw(fitted$grob)
  dev.off()
  invisible(fitted)
}

## --- 组合图标准(2026-08-03 增补)----------------------------------------
## 问题:patchwork/cowplot 的自动排版不识别各子图的注释文本量(y tick 最宽
## 标签、旋转 x 标签、strip/legend),长文本子图会被压 plot 区域或裁剪。
## 规则:逐子图独立实测(保留各子图自有 theme),列宽 = 该列各子图实测宽
## 最大值,行高同理;画布 = 各列宽之和 x 各行高之和。

## 保留子图自有 theme 的 fit(tessera 模式):std_theme 先应用,p$theme 后挂回
fit_panel_keep <- function(p, plot_w_in, plot_h_in, base_size = STD_BASE_SIZE) {
  th <- p$theme
  g <- ggplotGrob(p + std_theme(base_size) + th)
  g <- .lock_panel_size(g, plot_w_in, plot_h_in)
  c(list(grob = g), measure_canvas(g))
}

## panels: ggplot 列表(按行优先顺序);ncol: 列数
## 返回 list(patchwork, w_in, h_in, cell_w, cell_h)
compose_std <- function(panels, ncol, plot_w_in, plot_h_in,
                        base_size = STD_BASE_SIZE) {
  if (!requireNamespace("patchwork", quietly = TRUE)) stop("patchwork required")
  fits <- lapply(panels, fit_panel_keep, plot_w_in, plot_h_in, base_size)
  n <- length(fits)
  nrow <- ceiling(n / ncol)
  idx <- matrix(c(seq_len(n), rep(NA, nrow * ncol - n)), nrow = nrow, byrow = TRUE)
  col_w <- vapply(seq_len(ncol), function(j)
    max(vapply(idx[, j][!is.na(idx[, j])], function(i) fits[[i]]$w_in, numeric(1))),
    numeric(1))
  row_h <- vapply(seq_len(nrow), function(i)
    max(vapply(idx[i, ][!is.na(idx[i, ])], function(j) fits[[j]]$h_in, numeric(1))),
    numeric(1))
  cells <- lapply(fits, function(f) patchwork::wrap_elements(full = f$grob))
  comp <- Reduce(`|`, cells[seq_len(min(ncol, n))])
  if (nrow > 1) {
    rows <- vector("list", nrow)
    for (i in seq_len(nrow)) {
      row_cells <- cells[idx[i, ][!is.na(idx[i, ])]]
      rows[[i]] <- Reduce(`|`, row_cells)
    }
    comp <- Reduce(`/`, rows)
  }
  list(patchwork = comp, w_in = sum(col_w), h_in = sum(row_h),
       cell_w = col_w, cell_h = row_h, fits = fits)
}

save_pdf_composed <- function(panels, file, ncol, plot_w_in, plot_h_in,
                              base_size = STD_BASE_SIZE) {
  c <- compose_std(panels, ncol, plot_w_in, plot_h_in, base_size)
  pdf(file, width = c$w_in, height = c$h_in, useDingbats = FALSE)
  print(c$patchwork)
  dev.off()
  message(sprintf("%s: %d panels -> canvas %.2fx%.2f in",
                  basename(file), length(panels), c$w_in, c$h_in))
  invisible(c)
}

save_tikz_composed <- function(panels, file, ncol, plot_w_in, plot_h_in,
                               base_size = STD_BASE_SIZE, sanitize = TRUE) {
  c <- compose_std(panels, ncol, plot_w_in, plot_h_in, base_size)
  tikzDevice::tikz(file, width = c$w_in, height = c$h_in,
                   sanitize = sanitize, standAlone = FALSE)
  print(c$patchwork)
  dev.off()
  invisible(c)
}
