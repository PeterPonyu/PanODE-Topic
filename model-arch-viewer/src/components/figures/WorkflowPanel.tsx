"use client";

import React from "react";
import { PanelSection } from "./FigureComponents";

/* ╔══════════════════════════════════════════════════════════════════════════╗
   ║  WorkflowPanel — Panel A for Figures 2–7                               ║
   ║                                                                        ║
   ║  Renders a horizontal left-to-right flowchart showing the              ║
   ║  data-to-visualization pipeline for each figure.                       ║
   ║  Entirely browser-rendered (no matplotlib PNG dependency).             ║
   ║                                                                        ║
   ║  Each step includes a schematic SVG icon for visual clarity.           ║
   ║  The last step shows the methodological output (NOT panel refs).       ║
   ║                                                                        ║
   ║  Color scheme: neutral blue (#E8F0FE fill, #4A7CBF border) to avoid   ║
   ║  conflict with model-specific palette (DPMM=orange, Topic=purple).    ║
   ╚══════════════════════════════════════════════════════════════════════════╝ */

// ─── Step data type ──────────────────────────────────────────────────────────
interface Step {
  label: string;
  sub?: string;
  icon?: string;  // icon key for schematic SVG
}

// ─── Schematic SVG Icons (minimalistic, publication-friendly) ────────────────
function StepIcon({ icon }: { icon?: string }) {
  if (!icon) return null;
  const size = 18;
  const c = "#4A7CBF";
  const c2 = "#6A9FD8";

  const icons: Record<string, React.ReactNode> = {
    // Cell/dataset: stylised cell cluster (3 circles)
    cells: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <circle cx="7" cy="8" r="4" fill={c2} opacity="0.5" stroke={c} strokeWidth="0.6"/>
        <circle cx="13" cy="8" r="4" fill={c2} opacity="0.5" stroke={c} strokeWidth="0.6"/>
        <circle cx="10" cy="14" r="4" fill={c2} opacity="0.5" stroke={c} strokeWidth="0.6"/>
      </svg>
    ),
    // Preprocess funnel
    preprocess: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <polygon points="3,4 17,4 12,11 12,17 8,17 8,11" fill={c2} opacity="0.5" stroke={c} strokeWidth="0.7"/>
        <line x1="5" y1="7" x2="15" y2="7" stroke={c} strokeWidth="0.5" opacity="0.4"/>
      </svg>
    ),
    // Train: neural network icon (3 nodes → 2 nodes → 1 node)
    train: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <circle cx="4" cy="5" r="2" fill={c2} stroke={c} strokeWidth="0.5"/>
        <circle cx="4" cy="10" r="2" fill={c2} stroke={c} strokeWidth="0.5"/>
        <circle cx="4" cy="15" r="2" fill={c2} stroke={c} strokeWidth="0.5"/>
        <circle cx="11" cy="7" r="2" fill={c2} stroke={c} strokeWidth="0.5"/>
        <circle cx="11" cy="13" r="2" fill={c2} stroke={c} strokeWidth="0.5"/>
        <circle cx="17" cy="10" r="2" fill={c} stroke={c} strokeWidth="0.5"/>
        <line x1="6" y1="5" x2="9" y2="7" stroke={c} strokeWidth="0.4" opacity="0.5"/>
        <line x1="6" y1="10" x2="9" y2="7" stroke={c} strokeWidth="0.4" opacity="0.5"/>
        <line x1="6" y1="10" x2="9" y2="13" stroke={c} strokeWidth="0.4" opacity="0.5"/>
        <line x1="6" y1="15" x2="9" y2="13" stroke={c} strokeWidth="0.4" opacity="0.5"/>
        <line x1="13" y1="7" x2="15" y2="10" stroke={c} strokeWidth="0.4" opacity="0.5"/>
        <line x1="13" y1="13" x2="15" y2="10" stroke={c} strokeWidth="0.4" opacity="0.5"/>
      </svg>
    ),
    // Evaluate: checklist / ruler
    evaluate: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <rect x="4" y="2" width="12" height="16" rx="1" fill="none" stroke={c} strokeWidth="0.7"/>
        <line x1="7" y1="6" x2="14" y2="6" stroke={c} strokeWidth="0.6"/>
        <line x1="7" y1="9" x2="14" y2="9" stroke={c} strokeWidth="0.6"/>
        <line x1="7" y1="12" x2="14" y2="12" stroke={c} strokeWidth="0.6"/>
        <line x1="7" y1="15" x2="11" y2="15" stroke={c} strokeWidth="0.6"/>
        <polyline points="5,5.5 5.8,6.5 7,5" fill="none" stroke="#59A14F" strokeWidth="0.7"/>
        <polyline points="5,8.5 5.8,9.5 7,8" fill="none" stroke="#59A14F" strokeWidth="0.7"/>
        <polyline points="5,11.5 5.8,12.5 7,11" fill="none" stroke="#59A14F" strokeWidth="0.7"/>
      </svg>
    ),
    // Compare / trade-off: balance / comparison
    compare: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <rect x="3" y="12" width="5" height="6" rx="0.5" fill={c2} opacity="0.6" stroke={c} strokeWidth="0.5"/>
        <rect x="12" y="8" width="5" height="10" rx="0.5" fill={c} opacity="0.5" stroke={c} strokeWidth="0.5"/>
        <line x1="3" y1="5" x2="17" y2="5" stroke={c} strokeWidth="0.5" strokeDasharray="1.5,1"/>
        <polygon points="8,3 12,3 10,6" fill={c} opacity="0.4"/>
      </svg>
    ),
    // Sweep: slider / dial
    sweep: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <line x1="3" y1="6" x2="17" y2="6" stroke={c} strokeWidth="0.7"/>
        <circle cx="12" cy="6" r="2.5" fill={c2} stroke={c} strokeWidth="0.6"/>
        <line x1="3" y1="14" x2="17" y2="14" stroke={c} strokeWidth="0.7"/>
        <circle cx="7" cy="14" r="2.5" fill={c2} stroke={c} strokeWidth="0.6"/>
      </svg>
    ),
    // Metric chart: mini bar chart
    metrics: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <rect x="3" y="10" width="3" height="8" rx="0.3" fill={c2} stroke={c} strokeWidth="0.4"/>
        <rect x="8" y="5" width="3" height="13" rx="0.3" fill={c} opacity="0.6" stroke={c} strokeWidth="0.4"/>
        <rect x="13" y="8" width="3" height="10" rx="0.3" fill={c2} stroke={c} strokeWidth="0.4"/>
      </svg>
    ),
    // UMAP: scatter cloud
    umap: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <circle cx="6" cy="7" r="1.5" fill="#E74C3C" opacity="0.7"/>
        <circle cx="8" cy="5" r="1.5" fill="#E74C3C" opacity="0.7"/>
        <circle cx="5" cy="9" r="1.5" fill="#E74C3C" opacity="0.7"/>
        <circle cx="14" cy="13" r="1.5" fill="#3498DB" opacity="0.7"/>
        <circle cx="12" cy="15" r="1.5" fill="#3498DB" opacity="0.7"/>
        <circle cx="15" cy="11" r="1.5" fill="#3498DB" opacity="0.7"/>
        <circle cx="10" cy="10" r="1.2" fill="#2ECC71" opacity="0.6"/>
        <circle cx="9" cy="12" r="1.2" fill="#2ECC71" opacity="0.6"/>
      </svg>
    ),
    // Scatter/hull: convex hull
    scatter: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <polygon points="4,14 7,5 14,4 17,10 13,16 6,16" fill={c2} opacity="0.15" stroke={c} strokeWidth="0.6" strokeDasharray="2,1"/>
        <circle cx="7" cy="5" r="1.5" fill="#E74C3C" opacity="0.7"/>
        <circle cx="14" cy="4" r="1.5" fill="#E74C3C" opacity="0.7"/>
        <circle cx="10" cy="10" r="1.5" fill="#3498DB" opacity="0.7"/>
        <circle cx="6" cy="13" r="1.5" fill="#3498DB" opacity="0.7"/>
        <circle cx="15" cy="12" r="1.5" fill="#E74C3C" opacity="0.7"/>
      </svg>
    ),
    // Gradient / saliency: heatmap-like grid
    gradient: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <rect x="2" y="4" width="4" height="4" fill="#FFFFB2" stroke={c} strokeWidth="0.3"/>
        <rect x="6" y="4" width="4" height="4" fill="#FECC5C" stroke={c} strokeWidth="0.3"/>
        <rect x="10" y="4" width="4" height="4" fill="#FD8D3C" stroke={c} strokeWidth="0.3"/>
        <rect x="14" y="4" width="4" height="4" fill="#E31A1C" stroke={c} strokeWidth="0.3"/>
        <rect x="2" y="8" width="4" height="4" fill="#FD8D3C" stroke={c} strokeWidth="0.3"/>
        <rect x="6" y="8" width="4" height="4" fill="#E31A1C" stroke={c} strokeWidth="0.3"/>
        <rect x="10" y="8" width="4" height="4" fill="#FECC5C" stroke={c} strokeWidth="0.3"/>
        <rect x="14" y="8" width="4" height="4" fill="#FFFFB2" stroke={c} strokeWidth="0.3"/>
        <rect x="2" y="12" width="4" height="4" fill="#FECC5C" stroke={c} strokeWidth="0.3"/>
        <rect x="6" y="12" width="4" height="4" fill="#FFFFB2" stroke={c} strokeWidth="0.3"/>
        <rect x="10" y="12" width="4" height="4" fill="#FD8D3C" stroke={c} strokeWidth="0.3"/>
        <rect x="14" y="12" width="4" height="4" fill="#E31A1C" stroke={c} strokeWidth="0.3"/>
      </svg>
    ),
    // Enrichment: GO term dots
    enrichment: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <line x1="5" y1="5" x2="5" y2="17" stroke={c} strokeWidth="0.5"/>
        <line x1="5" y1="17" x2="18" y2="17" stroke={c} strokeWidth="0.5"/>
        <circle cx="10" cy="7" r="3" fill="#59A14F" opacity="0.5" stroke="#59A14F" strokeWidth="0.4"/>
        <circle cx="14" cy="11" r="2" fill="#E15759" opacity="0.5" stroke="#E15759" strokeWidth="0.4"/>
        <circle cx="8" cy="13" r="2.5" fill="#4E79A7" opacity="0.5" stroke="#4E79A7" strokeWidth="0.4"/>
      </svg>
    ),
    // Latent: dimensionality / embedding
    latent: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <rect x="2" y="3" width="6" height="14" rx="1" fill="none" stroke={c} strokeWidth="0.6"/>
        <line x1="5" y1="6" x2="5" y2="14" stroke={c} strokeWidth="0.5" strokeDasharray="1,1"/>
        <path d="M11,10 L18,5" stroke={c} strokeWidth="0.6" fill="none" markerEnd=""/>
        <path d="M11,10 L18,15" stroke={c} strokeWidth="0.6" fill="none"/>
        <circle cx="18" cy="5" r="1.5" fill={c2} stroke={c} strokeWidth="0.4"/>
        <circle cx="18" cy="15" r="1.5" fill={c2} stroke={c} strokeWidth="0.4"/>
        <circle cx="11" cy="10" r="2" fill={c} opacity="0.5" stroke={c} strokeWidth="0.4"/>
      </svg>
    ),
    // Perturbation: gene knock-out / delta
    perturb: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <circle cx="10" cy="10" r="7" fill="none" stroke={c} strokeWidth="0.7"/>
        <line x1="5" y1="5" x2="15" y2="15" stroke="#E74C3C" strokeWidth="1" opacity="0.7"/>
        <text x="10" y="13" textAnchor="middle" fontSize="8" fill="#1f2937" fontWeight="bold">G</text>
      </svg>
    ),
    // Differential: delta comparison
    diff: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <text x="10" y="14" textAnchor="middle" fontSize="14" fill="#1f2937" fontWeight="bold" fontFamily="serif">Δ</text>
      </svg>
    ),
    // Heatmap icon
    heatmap: (
      <svg width={size} height={size} viewBox="0 0 20 20">
        <rect x="3" y="3" width="14" height="14" rx="1" fill="none" stroke={c} strokeWidth="0.6"/>
        <rect x="4" y="4" width="4" height="4" fill="#FDD49E"/>
        <rect x="8" y="4" width="4" height="4" fill="#FD8D3C"/>
        <rect x="12" y="4" width="4" height="4" fill="#E6550D"/>
        <rect x="4" y="8" width="4" height="4" fill="#E6550D"/>
        <rect x="8" y="8" width="4" height="4" fill="#FDD49E"/>
        <rect x="12" y="8" width="4" height="4" fill="#FD8D3C"/>
        <rect x="4" y="12" width="4" height="4" fill="#FD8D3C"/>
        <rect x="8" y="12" width="4" height="4" fill="#E6550D"/>
        <rect x="12" y="12" width="4" height="4" fill="#FDD49E"/>
      </svg>
    ),
  };

  return (
    <div style={{ display: "flex", justifyContent: "center", marginBottom: "2px" }}>
      {icons[icon] ?? null}
    </div>
  );
}

// ─── Arrow connector ─────────────────────────────────────────────────────────
function Arrow() {
  return (
    <div style={{ display: "flex", alignItems: "center", flexShrink: 0, padding: "0 1px" }}>
      <svg width="16" height="10" viewBox="0 0 16 10" style={{ display: "block" }}>
        <line x1="0" y1="5" x2="10" y2="5" stroke="#6A6A6A" strokeWidth="1.2" />
        <polygon points="10,1.5 16,5 10,8.5" fill="#6A6A6A" />
      </svg>
    </div>
  );
}

// ─── Per-figure per-series step definitions ──────────────────────────────────
// NOTE: The last step describes the *methodological output*, NOT panel labels.
function getWorkflowSteps(figNum: number, series: string): Step[] {
  const isDpmm = series === "dpmm";
  const modelFamily = isDpmm ? "DPMM" : "Topic";
  const baselineFamily = isDpmm ? "Pure-AE" : "Pure-VAE";
  const baseModel = isDpmm ? "DPMM-Base" : "Topic-Base";

  const STEPS: Record<number, Step[]> = {
    2: [
      { label: "12 scRNA-seq", sub: "datasets", icon: "cells" },
      { label: "Preprocess", sub: "HVG · norm · log1p", icon: "preprocess" },
      { label: "Train 6 Models", sub: `3 ${modelFamily} + 3 ${baselineFamily}`, icon: "train" },
      { label: "Evaluate", sub: `${isDpmm ? "41" : "6"} metrics × 12 datasets`, icon: "evaluate" },
      { label: "Ablation Analysis", sub: "UMAP · metrics · efficiency", icon: "compare" },
    ],
    3: [
      { label: "12 scRNA-seq", sub: "datasets", icon: "cells" },
      { label: baseModel, sub: "single architecture", icon: "train" },
      { label: "Sweep 10 HPs", sub: "one factor at a time", icon: "sweep" },
      { label: `${isDpmm ? "6" : "4"} Core Metrics`, sub: "per sweep value", icon: "metrics" },
      { label: "Sensitivity Profile", sub: "parameter robustness", icon: "evaluate" },
    ],
    4: [
      { label: "3 Repr. Datasets", sub: "setty · endo · dentate", icon: "cells" },
      { label: "HP Sweep", sub: "10 parameters", icon: "sweep" },
      { label: "Latent Extract", sub: "per sweep point", icon: "latent" },
      { label: "KMeans + UMAP", sub: "cluster & project", icon: "umap" },
      { label: "Geometry Trends", sub: "embedding evolution", icon: "scatter" },
    ],
    5: [
      { label: "12 scRNA-seq", sub: "datasets", icon: "cells" },
      { label: "Train 6 Models", sub: `3 ${modelFamily} + 3 ${baselineFamily}`, icon: "train" },
      { label: `${isDpmm ? "41" : "6"} Metrics`, sub: "per model × dataset", icon: "metrics" },
      { label: "Pairwise Scatter", sub: "convex hulls", icon: "scatter" },
      { label: "Trade-off Landscape", sub: "multi-objective", icon: "compare" },
    ],
    6: [
      { label: "3 Repr. Datasets", sub: "setty · endo · dentate", icon: "cells" },
      { label: `Train 3 ${modelFamily}`, sub: "Base · Trans · Contr", icon: "train" },
      { label: "Gradient Saliency", sub: "∂loss / ∂gene", icon: "gradient" },
      { label: "Gene Ranking", sub: "top-K per component", icon: "heatmap" },
    ],
    7: [
      { label: "Salient Genes", sub: "per component", icon: "gradient" },
      { label: "GO / BP Query", sub: "Enrichr API", icon: "enrichment" },
      { label: "Filter adj. p < 0.05", sub: "Bonferroni", icon: "evaluate" },
      { label: "Enrichment Map", sub: "dotplot & terms", icon: "enrichment" },
    ],
  };

  return STEPS[figNum] ?? [];
}

// ─── Component ───────────────────────────────────────────────────────────────
interface WorkflowPanelProps {
  figNum: number;
  series: string;
}

export default function WorkflowPanel({ figNum, series }: WorkflowPanelProps) {
  const steps = getWorkflowSteps(figNum, series);
  if (steps.length === 0) return null;

  return (
    <PanelSection label="A" title="Workflow">
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "3px",
          padding: "4px 2px",
        }}
      >
        {steps.map((step, i) => (
          <React.Fragment key={i}>
            {/* Rounded box with icon */}
            <div
              style={{
                flex: 1,
                minWidth: 0,
                backgroundColor: "#E8F0FE",
                border: "1px solid #4A7CBF",
                borderRadius: "5px",
                padding: "4px 4px 5px",
                textAlign: "center",
              }}
            >
              <StepIcon icon={step.icon} />
              <div
                style={{
                  fontSize: "9px",
                  fontWeight: "bold",
                  color: "#1A1A1A",
                  fontFamily: "Arial, sans-serif",
                  lineHeight: 1.2,
                }}
              >
                {step.label}
              </div>
              {step.sub && (
                <div
                  style={{
                    fontSize: "7px",
                    color: "#555555",
                    fontFamily: "Arial, sans-serif",
                    lineHeight: 1.2,
                    marginTop: "1px",
                  }}
                >
                  {step.sub}
                </div>
              )}
            </div>
            {/* Arrow to next box */}
            {i < steps.length - 1 && <Arrow />}
          </React.Fragment>
        ))}
      </div>
    </PanelSection>
  );
}
