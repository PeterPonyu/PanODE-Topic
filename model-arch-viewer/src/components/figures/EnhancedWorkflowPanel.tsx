"use client";

import React from "react";
import { PanelSection } from "./FigureComponents";

/* ╔══════════════════════════════════════════════════════════════════════════╗
   ║  EnhancedWorkflowPanel — Compact icon-driven workflows for Fig 2-5     ║
   ║                                                                        ║
   ║  Compact horizontal layouts with minimal padding, using small icons   ║
   ║  instead of large colored blocks. Maximizes information density.       ║
   ╚══════════════════════════════════════════════════════════════════════════╝ */

// ─── Styling ─────────────────────────────────────────────────────────────────
const FONT = "Arial, sans-serif";
const CLR = {
  text: "#1A1A1A",
  textSub: "#555555",
  arrow: "#6A6A6A",
};

// ─── Icon Library (16x16 SVG icons) ──────────────────────────────────────────
const Icons = {
  // Dataset (cell cluster)
  cells: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <circle cx="7" cy="8" r="3.5" fill="#6A9FD8" opacity="0.6" stroke="#4A7CBF" strokeWidth="0.7"/>
      <circle cx="13" cy="8" r="3.5" fill="#6A9FD8" opacity="0.6" stroke="#4A7CBF" strokeWidth="0.7"/>
      <circle cx="10" cy="14" r="3.5" fill="#6A9FD8" opacity="0.6" stroke="#4A7CBF" strokeWidth="0.7"/>
    </svg>
  ),
  
  // Preprocessing funnel
  preprocess: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <polygon points="3,4 17,4 13,11 13,17 7,17 7,11" fill="#A8DADC" opacity="0.7" stroke="#457B9D" strokeWidth="0.8"/>
      <line x1="5" y1="7" x2="15" y2="7" stroke="#1D3557" strokeWidth="0.5"/>
    </svg>
  ),
  
  // Neural network
  train: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <circle cx="4" cy="6" r="1.8" fill="#457B9D" stroke="#1D3557" strokeWidth="0.5"/>
      <circle cx="4" cy="10" r="1.8" fill="#457B9D" stroke="#1D3557" strokeWidth="0.5"/>
      <circle cx="4" cy="14" r="1.8" fill="#457B9D" stroke="#1D3557" strokeWidth="0.5"/>
      <circle cx="11" cy="8" r="1.8" fill="#F1FAEE" stroke="#457B9D" strokeWidth="0.5"/>
      <circle cx="11" cy="12" r="1.8" fill="#F1FAEE" stroke="#457B9D" strokeWidth="0.5"/>
      <circle cx="17" cy="10" r="1.8" fill="#E63946" stroke="#A8DADC" strokeWidth="0.5"/>
      <line x1="6" y1="6" x2="9" y2="8" stroke="#457B9D" strokeWidth="0.4" opacity="0.5"/>
      <line x1="6" y1="10" x2="9" y2="8" stroke="#457B9D" strokeWidth="0.4" opacity="0.5"/>
      <line x1="6" y1="10" x2="9" y2="12" stroke="#457B9D" strokeWidth="0.4" opacity="0.5"/>
      <line x1="6" y1="14" x2="9" y2="12" stroke="#457B9D" strokeWidth="0.4" opacity="0.5"/>
      <line x1="13" y1="8" x2="15" y2="10" stroke="#E63946" strokeWidth="0.5" opacity="0.6"/>
      <line x1="13" y1="12" x2="15" y2="10" stroke="#E63946" strokeWidth="0.5" opacity="0.6"/>
    </svg>
  ),
  
  // Checklist evaluation
  evaluate: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="4" y="3" width="12" height="14" rx="1" fill="none" stroke="#457B9D" strokeWidth="0.8"/>
      <polyline points="6,6 7,7 9,5" fill="none" stroke="#2A9D8F" strokeWidth="0.8"/>
      <polyline points="6,9 7,10 9,8" fill="none" stroke="#2A9D8F" strokeWidth="0.8"/>
      <polyline points="6,12 7,13 9,11" fill="none" stroke="#2A9D8F" strokeWidth="0.8"/>
      <line x1="10" y1="6" x2="14" y2="6" stroke="#457B9D" strokeWidth="0.6"/>
      <line x1="10" y1="9" x2="14" y2="9" stroke="#457B9D" strokeWidth="0.6"/>
      <line x1="10" y1="12" x2="14" y2="12" stroke="#457B9D" strokeWidth="0.6"/>
    </svg>
  ),
  
  // Compare/balance
  compare: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="3" y="13" width="5" height="5" rx="0.5" fill="#E76F51" opacity="0.6" stroke="#D62828" strokeWidth="0.6"/>
      <rect x="12" y="9" width="5" height="9" rx="0.5" fill="#2A9D8F" opacity="0.6" stroke="#1D7874" strokeWidth="0.6"/>
      <line x1="3" y1="6" x2="17" y2="6" stroke="#457B9D" strokeWidth="0.6" strokeDasharray="1.5,1"/>
      <polygon points="8,4 12,4 10,6.5" fill="#457B9D" opacity="0.5"/>
    </svg>
  ),
  
  // Hyperparameter sweep/dial
  sweep: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <line x1="3" y1="7" x2="17" y2="7" stroke="#457B9D" strokeWidth="0.8"/>
      <circle cx="12" cy="7" r="2.2" fill="#E76F51" stroke="#D62828" strokeWidth="0.6"/>
      <line x1="3" y1="13" x2="17" y2="13" stroke="#457B9D" strokeWidth="0.8"/>
      <circle cx="7" cy="13" r="2.2" fill="#2A9D8F" stroke="#1D7874" strokeWidth="0.6"/>
    </svg>
  ),
  
  // Metrics bar chart
  metrics: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="3" y="11" width="3" height="7" rx="0.4" fill="#457B9D" stroke="#1D3557" strokeWidth="0.5"/>
      <rect x="8" y="6" width="3" height="12" rx="0.4" fill="#E63946" stroke="#A8DADC" strokeWidth="0.5"/>
      <rect x="13" y="9" width="3" height="9" rx="0.4" fill="#2A9D8F" stroke="#1D7874" strokeWidth="0.5"/>
    </svg>
  ),
  
  // UMAP scatter
  umap: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <circle cx="6" cy="7" r="1.4" fill="#E63946" opacity="0.7"/>
      <circle cx="8" cy="5" r="1.4" fill="#E63946" opacity="0.7"/>
      <circle cx="5" cy="9" r="1.4" fill="#E63946" opacity="0.7"/>
      <circle cx="14" cy="13" r="1.4" fill="#457B9D" opacity="0.7"/>
      <circle cx="12" cy="15" r="1.4" fill="#457B9D" opacity="0.7"/>
      <circle cx="15" cy="11" r="1.4" fill="#457B9D" opacity="0.7"/>
      <circle cx="10" cy="10" r="1.2" fill="#2A9D8F" opacity="0.7"/>
    </svg>
  ),
  
  // Scatter with hull
  scatter: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <polygon points="4,14 7,5 14,4 17,10 13,16 6,16" fill="#A8DADC" opacity="0.2" stroke="#457B9D" strokeWidth="0.7" strokeDasharray="1.5,1"/>
      <circle cx="7" cy="5" r="1.3" fill="#E63946" opacity="0.8"/>
      <circle cx="14" cy="4" r="1.3" fill="#E63946" opacity="0.8"/>
      <circle cx="10" cy="10" r="1.3" fill="#457B9D" opacity="0.8"/>
      <circle cx="6" cy="13" r="1.3" fill="#457B9D" opacity="0.8"/>
      <circle cx="15" cy="12" r="1.3" fill="#2A9D8F" opacity="0.8"/>
    </svg>
  ),
  
  // Heatmap/gradient
  gradient: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="3" y="4" width="4" height="4" fill="#FEE5D9"/>
      <rect x="7" y="4" width="4" height="4" fill="#FCAE91"/>
      <rect x="11" y="4" width="4" height="4" fill="#FB6A4A"/>
      <rect x="15" y="4" width="2" height="4" fill="#CB181D"/>
      <rect x="3" y="8" width="4" height="4" fill="#FB6A4A"/>
      <rect x="7" y="8" width="4" height="4" fill="#CB181D"/>
      <rect x="11" y="8" width="4" height="4" fill="#FCAE91"/>
      <rect x="15" y="8" width="2" height="4" fill="#FEE5D9"/>
      <rect x="3" y="12" width="4" height="4" fill="#FCAE91"/>
      <rect x="7" y="12" width="4" height="4" fill="#FEE5D9"/>
      <rect x="11" y="12" width="4" height="4" fill="#FB6A4A"/>
      <rect x="15" y="12" width="2" height="4" fill="#A50F15"/>
    </svg>
  ),
  
  // Latent space
  latent: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="2" y="4" width="6" height="12" rx="0.8" fill="none" stroke="#457B9D" strokeWidth="0.7"/>
      <line x1="5" y1="7" x2="5" y2="13" stroke="#457B9D" strokeWidth="0.5" strokeDasharray="0.8,0.8"/>
      <line x1="11" y1="10" x2="16" y2="6" stroke="#E63946" strokeWidth="0.6"/>
      <line x1="11" y1="10" x2="16" y2="14" stroke="#2A9D8F" strokeWidth="0.6"/>
      <circle cx="16" cy="6" r="1.5" fill="#E63946" opacity="0.6"/>
      <circle cx="16" cy="14" r="1.5" fill="#2A9D8F" opacity="0.6"/>
      <circle cx="11" cy="10" r="1.8" fill="#457B9D" opacity="0.5" stroke="#1D3557" strokeWidth="0.5"/>
    </svg>
  ),
  
  // Model architecture
  model: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="3" y="4" width="5" height="12" rx="1" fill="#E76F51" opacity="0.3" stroke="#D62828" strokeWidth="0.7"/>
      <rect x="9" y="6" width="4" height="8" rx="0.8" fill="#457B9D" opacity="0.3" stroke="#1D3557" strokeWidth="0.7"/>
      <rect x="14" y="8" width="3" height="4" rx="0.6" fill="#2A9D8F" opacity="0.3" stroke="#1D7874" strokeWidth="0.7"/>
      <line x1="5.5" y1="7" x2="5.5" y2="10" stroke="#D62828" strokeWidth="0.5"/>
      <line x1="11" y1="8.5" x2="11" y2="11.5" stroke="#1D3557" strokeWidth="0.5"/>
    </svg>
  ),
  
  // Ablation/split
  ablation: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="2" y="3" width="7" height="14" rx="1" fill="#E76F51" opacity="0.25" stroke="#D62828" strokeWidth="0.8"/>
      <rect x="11" y="3" width="7" height="14" rx="1" fill="#457B9D" opacity="0.25" stroke="#1D3557" strokeWidth="0.8"/>
      <text x="5.5" y="11.5" textAnchor="middle" fontSize="7" fill="#1f2937" fontWeight="bold">+</text>
      <text x="14.5" y="11.5" textAnchor="middle" fontSize="7" fill="#1f2937" fontWeight="bold">−</text>
    </svg>
  ),
  
  // Time/evolution
  time: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <line x1="2" y1="10" x2="18" y2="10" stroke="#457B9D" strokeWidth="0.8"/>
      <circle cx="4" cy="10" r="1.5" fill="#A7C4D8" stroke="#457B9D" strokeWidth="0.6"/>
      <circle cx="8" cy="10" r="1.5" fill="#78A8C8" stroke="#457B9D" strokeWidth="0.6"/>
      <circle cx="12" cy="10" r="1.5" fill="#4A7CBF" stroke="#1D3557" strokeWidth="0.6"/>
      <circle cx="16" cy="10" r="1.5" fill="#E63946" stroke="#A50F15" strokeWidth="0.6"/>
      <polygon points="17,5 19,5 18,8" fill="#E63946" opacity="0.5"/>
    </svg>
  ),
  
  // Multi-objective
  pareto: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <circle cx="6" cy="14" r="2" fill="#2A9D8F" opacity="0.6" stroke="#1D7874" strokeWidth="0.6"/>
      <circle cx="10" cy="10" r="2" fill="#E76F51" opacity="0.6" stroke="#D62828" strokeWidth="0.6"/>
      <circle cx="14" cy="6" r="2" fill="#457B9D" opacity="0.6" stroke="#1D3557" strokeWidth="0.6"/>
      <path d="M6,14 Q8,12 10,10 T14,6" fill="none" stroke="#1D3557" strokeWidth="0.7" strokeDasharray="1,1"/>
      <text x="16" y="5" fontSize="8" fill="#1f2937">↑</text>
    </svg>
  ),
};

// ─── Phase Component (Grouping Steps) ────────────────────────────────────────
function Phase({ label, color, children, details }: { label: string, color: string, children: React.ReactNode, details?: string }) {
  return (
    <div style={{
      display: "flex", 
      flexDirection: "column",
      border: `1px solid ${color}40`, 
      backgroundColor: `${color}08`, 
      borderRadius: "4px",
      margin: "0 2px",
      padding: "3px 2px 2px 2px",
      position: "relative",
      flexGrow: 1,
    }}>
       {/* Phase Label Badge */}
      <div style={{
        position: "absolute",
        top: "-6px",
        left: "4px",
        backgroundColor: color,
        color: "#fff",
        fontSize: "6.5px",
        fontWeight: "bold",
        padding: "0.5px 4px",
        borderRadius: "2px",
        lineHeight: "1.1",
        zIndex: 1,
      }}>
        {label}
      </div>

      {/* Steps Container */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-around", width: "100%" }}>
        {children}
      </div>

      {/* Detailed Description Footer */}
      {details && (
        <div style={{
           fontSize: "6px", 
           color: CLR.textSub, 
           marginTop: "3px", 
           textAlign: "center",
           borderTop: `0.5px solid ${color}30`, 
           paddingTop: "1.5px",
           lineHeight: "1.2",
           fontStyle: "italic",
           whiteSpace: "pre-wrap"
        }}>
          {details}
        </div>
      )}
    </div>
  );
}

// ─── Compact Step Component ──────────────────────────────────────────────────
function Step({ icon, label, sub }: { icon: React.ReactNode; label: string; sub?: string }) {
  return (
    <div style={{ 
      flex: 1, 
      minWidth: 0, 
      textAlign: "center",
      padding: "2px 3px", // Increased padding slightly
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      alignItems: "center"
    }}>
      <div style={{ display: "flex", justifyContent: "center", marginBottom: "1.5px" }}>
        {icon}
      </div>
      <div style={{ 
        fontSize: "7.5px", 
        fontWeight: "bold", 
        color: CLR.text, 
        fontFamily: FONT,
        lineHeight: 1.1,
      }}>
        {label}
      </div>
      {sub && (
        <div style={{ 
          fontSize: "6px", 
          color: CLR.textSub, 
          fontFamily: FONT,
          lineHeight: 1.05,
          marginTop: "1px",
        }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function Arrow() {
  return (
    <div style={{ display: "flex", alignItems: "center", flexShrink: 0, padding: "0 2px" }}>
      <svg width="10" height="6" viewBox="0 0 10 6" style={{ display: "block", opacity: 0.6 }}>
        <line x1="0" y1="3" x2="7" y2="3" stroke={CLR.arrow} strokeWidth="0.8" />
        <polygon points="7,1 10,3 7,5" fill={CLR.arrow} />
      </svg>
    </div>
  );
}


// ═══════════════════════════════════════════════════════════════════════════════
// Figure 2: Compact Ablation Flow (with split icon showing DPMM vs Pure)
// ═══════════════════════════════════════════════════════════════════════════════
function Fig2Flow({ series }: { series: string }) {
  const isDpmm = series === "dpmm";
  const modelFamily = isDpmm ? "DPMM" : "Topic";
  const baselineFamily = isDpmm ? "Pure-AE" : "Pure-VAE";

  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>
      
      <Phase label="INPUT DATA" color="#457B9D" details="Top-3k HVGs · log₁p normalisation · 4 species">
        <Step icon={Icons.cells} label="12 scRNA-seq" sub="Datasets" />
        <Arrow />
        <Step icon={Icons.preprocess} label="Preprocessing" sub="HVG · LogNorm" />
      </Phase>

      <Arrow />

      <Phase label="MODEL ABLATION" color="#E63946" details={isDpmm ? 'latent d=10 · enc=[256,128] · dropout=0.2 · wd=0 · lr=1e-3 · 1000 ep · batch=128' : 'K=10 · enc=128 · λ_KL=0.01 (Topic) / 1.0 (Pure) · drop=0.0 · wd=1e-3 · lr=1e-3 · 1000 ep'}>
        <Step icon={Icons.model} label="Architectures" sub="Base/Trans/Contr" />
        <Arrow />
        <Step icon={Icons.ablation} label="Ablation" sub={isDpmm ? "±DPMM · ±CL" : "±Dir KL · ±CL"} />
        <Arrow />
        <Step icon={Icons.train} label="Training" sub="5 seeds" />
      </Phase>

      <Arrow />

      <Phase label="EVALUATION" color="#2A9D8F" details={isDpmm ? "NMI · ARI · ASW · cLISI · iLISI · kBET · runtime" : "NMI · ARI · ASW · DAV · COR · CAL"}>
        <Step icon={Icons.evaluate} label={isDpmm ? "41 Metrics" : "6 Metrics"} sub={isDpmm ? "Bio/Batch/Time" : "Clust/Corr"} />
        <Arrow />
        <Step icon={Icons.compare} label="Validation" sub="Wilcoxon tests" />
      </Phase>

    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Figure 3: Compact Sensitivity (10 HP sweeps)
// ═══════════════════════════════════════════════════════════════════════════════
function Fig3Flow({ series }: { series: string }) {
  const isDpmm = series === "dpmm";
  const baseModel = isDpmm ? "DPMM-Base" : "Topic-Base";

  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>
      
      <Phase label="SETUP" color="#457B9D" details={isDpmm ? `${baseModel} — d=10, enc=[256,128], drop=0.2, warmup=0.9` : `${baseModel} — K=10, enc=128, λ_KL=0.01, drop=0.0`}>
        <Step icon={Icons.cells} label="12 Datasets" sub="Test Suite" />
        <Arrow />
        <Step icon={Icons.model} label={baseModel} sub="Default Config" />
      </Phase>

      <Arrow />

      <Phase label="SENSITIVITY ANALYSIS" color="#E76F51" details={isDpmm ? "K · α · conc · lr · drop · epochs · batch · warmup · wd · clip" : "K · s · fb · lr · drop · epochs · batch · β_kl · wd · clip"}>
        <Step icon={Icons.sweep} label="Prior Sweeps" sub={isDpmm ? "K/α/conc" : "K/s/fb"} />
        <Arrow />
        <Step icon={Icons.sweep} label="Train Sweeps" sub="lr/drop/epochs" />
        <Arrow />
        <Step icon={Icons.train} label="Re-Training" sub="1200+ Runs" />
      </Phase>

      <Arrow />

      <Phase label="ROBUSTNESS" color="#2A9D8F" details="Metric variance ≤ 5% across stable HP ranges">
        <Step icon={Icons.metrics} label="Metric Var" sub="NMI/ARI/ASW" />
        <Arrow />
        <Step icon={Icons.evaluate} label="Stability" sub="Range check" />
      </Phase>

    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Figure 4: Compact Training Dynamics (embedding evolution)
// ═══════════════════════════════════════════════════════════════════════════════
function Fig4Flow({ series }: { series: string }) {
  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>
      
      <Phase label="CASE STUDY" color="#457B9D" details={`${series === 'dpmm' ? 'DPMM-Base (det. AE + GMM prior)' : 'Topic-Base (VAE + LN prior)'} — 3 scale tiers`}>
        <Step icon={Icons.cells} label="3 Datasets" sub="Small/Med/Large" />
        <Arrow />
        <Step icon={Icons.sweep} label="Hyperparams" sub="Best config" />
      </Phase>

      <Arrow />

      <Phase label="DYNAMICS TRACKING" color="#9C27B0" details={`Snapshot ${series === 'dpmm' ? 'z ∈ ℝ^d' : 'θ ∈ Δ^{K-1}'} at ep 1, 10, 50, 100, …, final`}>
        <Step icon={Icons.train} label="Epoch Loop" sub="0 → MaxEpoch" />
        <Arrow />
        <Step icon={Icons.latent} label={series === 'dpmm' ? "Latent z" : "Latent θ"} sub="Snapshot/Epoch" />
        <Arrow />
        <Step icon={Icons.time} label="Trajectory" sub="Loss + metrics" />
      </Phase>

      <Arrow />

      <Phase label="CONVERGENCE" color="#2A9D8F" details="UMAP/PCA coloured by cell type — cluster separation">
        <Step icon={Icons.umap} label="Project" sub="UMAP/PCA" />
        <Arrow />
        <Step icon={Icons.evaluate} label="Evolution" sub="Cluster sep." />
      </Phase>

    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Figure 5: Compact Trade-off (Pareto frontier)
// ═══════════════════════════════════════════════════════════════════════════════
function Fig5Flow({ series }: { series: string }) {
  const isDpmm = series === "dpmm";
  const modelFamily = isDpmm ? "DPMM" : "Topic";
  const baselineFamily = isDpmm ? "Pure-AE" : "Pure-VAE";

  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>
      
      <Phase label="MULTI-MODEL" color="#457B9D" details={`${modelFamily}(Base/Trans/Contr.) vs ${baselineFamily}(Base/Trans/Contr.)`}>
        <Step icon={Icons.cells} label="12 Datasets" sub="Diverse types" />
        <Arrow />
        <Step icon={Icons.train} label="6 Models" sub={`${modelFamily} & ${baselineFamily}`} />
      </Phase>

      <Arrow />

      <Phase label="METRIC SPACE" color="#E63946" details={isDpmm ? 'NMI · ARI · ASW · DAV + DRE · LSE · DREX · LSEX + efficiency' : 'NMI · ARI · ASW · DAV + COR · CAL — same training config as Fig. 2'}>
        <Step icon={Icons.metrics} label="Bio Metrics" sub="NMI/ARI/ASW" />
        <Arrow />
        <Step icon={Icons.metrics} label={isDpmm ? "Recon Metrics" : "Geom Metrics"} sub={isDpmm ? "DRE/LSE/..." : "DAV/COR/CAL"} />
        <Arrow />
        <Step icon={Icons.time} label="Time/Mem" sub="s · MB" />
      </Phase>

      <Arrow />

      <Phase label="PARETO FRONT" color="#2A9D8F" details="Non-dominated frontier: bio-conservation vs batch-removal">
        <Step icon={Icons.scatter} label="Pairwise" sub="Metric vs Metric" />
        <Arrow />
        <Step icon={Icons.pareto} label="Frontier" sub="Best balance" />
      </Phase>

    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Figure 10: External Model Benchmark Comparison
// ═══════════════════════════════════════════════════════════════════════════════
function Fig10Flow({ series }: { series: string }) {
  const isDpmm = series === "dpmm";
  const modelFamily = isDpmm ? "DPMM" : "Topic";

  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>

      <Phase label="UNIFIED MODELS" color="#457B9D" details={`11 external baselines + Best-${modelFamily} + Best-${isDpmm ? 'Topic' : 'DPMM'}`}>
        <Step icon={Icons.model} label="11 External" sub="Baselines" />
        <Arrow />
        <Step icon={Icons.ablation} label={`Best-${modelFamily}`} sub="Internal Ref" />
      </Phase>

      <Arrow />

      <Phase label="UNIFIED PIPELINE" color="#E76F51" details={`Same preproc (3k HVG, log1p) · 12 datasets · d=10, lr=1e-3, 1000 ep (${isDpmm ? 'scDiff d=64, CLEAR d=128' : 'scDiff d=64, CLEAR/scGCC d=128'})`}>
        <Step icon={Icons.preprocess} label="Same Preproc" sub="HVG 3k · LogNorm" />
        <Arrow />
        <Step icon={Icons.cells} label="12 Datasets" sub="Unified eval" />
        <Arrow />
        <Step icon={Icons.train} label="Train / Infer" sub="Per model" />
      </Phase>

      <Arrow />

      <Phase label="COMPARISON" color="#2A9D8F" details={`${isDpmm ? "41" : "6"} metrics · Composite score · Ranking`}>
        <Step icon={Icons.evaluate} label={isDpmm ? "41 Metrics" : "6 Metrics"} sub={isDpmm ? "NMI/ARI/ASW/..." : "NMI/ARI/ASW/DAV"} />
        <Arrow />
        <Step icon={Icons.compare} label="Ranking" sub="Composite Score" />
      </Phase>

    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Export
// ═══════════════════════════════════════════════════════════════════════════════
interface EnhancedWorkflowPanelProps {
  figNum: 2 | 3 | 4 | 5 | 10;
  series: string;
}

export default function EnhancedWorkflowPanel({ figNum, series }: EnhancedWorkflowPanelProps) {
  const flows: Record<number, React.ReactNode> = {
    2: <Fig2Flow series={series} />,
    3: <Fig3Flow series={series} />,
    4: <Fig4Flow series={series} />,
    5: <Fig5Flow series={series} />,
    10: <Fig10Flow series={series} />,
  };

  return (
    <PanelSection label="A" title="Workflow">
      {flows[figNum]}
    </PanelSection>
  );
}
