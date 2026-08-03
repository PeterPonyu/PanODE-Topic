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
    w_in = convertWidth(sum(g$widths), "in", valueOnly = TRUE) + 0.04,
    h_in = convertHeight(sum(g$heights), "in", valueOnly = TRUE) + 0.04
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
