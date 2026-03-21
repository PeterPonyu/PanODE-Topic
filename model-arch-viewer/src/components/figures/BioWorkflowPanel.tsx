"use client";

import React from "react";
import { PanelSection } from "./FigureComponents";

/* ╔══════════════════════════════════════════════════════════════════════════╗
   ║  BioWorkflowPanel — Detailed architecture-style workflow for Fig 6 & 7 ║
   ║                                                                        ║
   ║  Upgraded to match EnhancedWorkflowPanel styled with Phase groupings,   ║
   ║  compact icons, and detailed descriptions.                            ║
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

  // Trained Model
  model: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="3" y="4" width="5" height="12" rx="1" fill="#E76F51" opacity="0.3" stroke="#D62828" strokeWidth="0.7"/>
      <rect x="9" y="6" width="4" height="8" rx="0.8" fill="#457B9D" opacity="0.3" stroke="#1D3557" strokeWidth="0.7"/>
      <rect x="14" y="8" width="3" height="4" rx="0.6" fill="#2A9D8F" opacity="0.3" stroke="#1D7874" strokeWidth="0.7"/>
      <line x1="5.5" y1="7" x2="5.5" y2="10" stroke="#D62828" strokeWidth="0.5"/>
      <line x1="11" y1="8.5" x2="11" y2="11.5" stroke="#1D3557" strokeWidth="0.5"/>
    </svg>
  ),

  // Perturbation Importance
  gradient: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <defs>
        <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" style={{stopColor:"#FEE5D9", stopOpacity:1}} />
          <stop offset="100%" style={{stopColor:"#A50F15", stopOpacity:1}} />
        </linearGradient>
      </defs>
      <rect x="2" y="5" width="16" height="10" fill="url(#grad1)" stroke="#A50F15" strokeWidth="0.5" />
      <text x="10" y="13" textAnchor="middle" fontSize="6" fontWeight="bold" fill="#fff">Δx̂</text>
    </svg>
  ),

  // Heatmap
  heatmap: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <rect x="3" y="3" width="4.5" height="4.5" fill="#FDD49E"/>
      <rect x="7.5" y="3" width="4.5" height="4.5" fill="#FD8D3C"/>
      <rect x="12" y="3" width="4.5" height="4.5" fill="#E6550D"/>
      <rect x="3" y="7.5" width="4.5" height="4.5" fill="#E6550D"/>
      <rect x="7.5" y="7.5" width="4.5" height="4.5" fill="#FDD49E"/>
      <rect x="12" y="7.5" width="4.5" height="4.5" fill="#FD8D3C"/>
      <rect x="3" y="12" width="4.5" height="4.5" fill="#FD8D3C"/>
      <rect x="7.5" y="12" width="4.5" height="4.5" fill="#E6550D"/>
      <rect x="12" y="12" width="4.5" height="4.5" fill="#FDD49E"/>
    </svg>
  ),

  // Filter / Top N
  filter: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <polygon points="2,4 18,4 10,14" fill="#A8DADC" stroke="#457B9D" strokeWidth="0.8"/>
      <line x1="10" y1="14" x2="10" y2="18" stroke="#457B9D" strokeWidth="0.8"/>
      <circle cx="10" cy="18" r="1" fill="#457B9D"/>
    </svg>
  ),

  // Database / API (Enrichr)
  database: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <ellipse cx="10" cy="5" rx="7" ry="2" fill="#E76F51" opacity="0.4" stroke="#D62828" strokeWidth="0.7"/>
      <path d="M3,5 v6 a7,2 0 0,0 14,0 v-6" fill="#E76F51" opacity="0.2" stroke="#D62828" strokeWidth="0.7"/>
      <path d="M3,11 v4 a7,2 0 0,0 14,0 v-4" fill="#E76F51" opacity="0.4" stroke="#D62828" strokeWidth="0.7"/>
      <text x="10" y="10" textAnchor="middle" fontSize="5" fill="#1f2937">GO</text>
    </svg>
  ),

  // Dot Plot (Enrichment)
  dotplot: (
    <svg width="16" height="16" viewBox="0 0 20 20">
      <line x1="4" y1="16" x2="18" y2="16" stroke="#555" strokeWidth="0.5"/>
      <line x1="4" y1="4" x2="4" y2="16" stroke="#555" strokeWidth="0.5"/>
      <circle cx="8" cy="12" r="1.5" fill="#4E79A7"/>
      <circle cx="12" cy="8" r="2.5" fill="#F28E2B"/>
      <circle cx="16" cy="5" r="2" fill="#E15759"/>
    </svg>
  ),
  
  // Statistical Test
  stats: (
     <svg width="16" height="16" viewBox="0 0 20 20">
      <text x="10" y="12" textAnchor="middle" fontSize="9" fontWeight="bold" fontFamily="serif" fill="#1f2937">P</text>
      <text x="14" y="9" textAnchor="middle" fontSize="6" fontWeight="bold" fontFamily="serif" fill="#1f2937">val</text>
    </svg>
  )
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
      padding: "2px 3px", 
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
// Figure 6: Gradient Saliency Workflow
// ═══════════════════════════════════════════════════════════════════════════════
function Fig6Flow({ series }: { series: string }) {
  const family = series === "dpmm" ? "DPMM" : "Topic";
  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>
      
      <Phase label="SETUP" color="#457B9D" details="Input scRNA-seq (normalized) & trained models">
        <Step icon={Icons.cells} label="3 Datasets" sub="Setty/Endo/Dentate" />
        <Arrow />
        <Step icon={Icons.model} label={`${family} Models`} sub="Pre-trained" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          z = f_θ(x), x̂ = g_φ(z)
        </div>
      </Phase>

      <Arrow />

      <Phase label="INTERPRETABILITY" color="#9C27B0" details="Forward-pass perturbation analysis (perturb z_k, decode, measure Δx̂)">
        <Step icon={Icons.gradient} label="Perturbation" sub="Perturb z_k → Δx̂" />
        <Arrow />
        <Step icon={Icons.heatmap} label="Heatmap" sub="Gene × Comp" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          I_kj = |g_φ(z+δe_k) − g_φ(z)|_j
        </div>
      </Phase>

      <Arrow />

      <Phase label="DISCOVERY" color="#2A9D8F" details="Genes grouped by dominant component, then ranked within each group (block-diagonal ordering)">
        <Step icon={Icons.filter} label="Top Genes" sub="argmax(I_k) grouping" />
        <Arrow />
        <Step icon={Icons.dotplot} label="Block-Diag." sub="Marker validation" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          sort by (argmax_k, −I_kj)
        </div>
      </Phase>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Figure 7: Latent–Gene Correlation Workflow
// ═══════════════════════════════════════════════════════════════════════════════
function Fig7Flow({ series }: { series: string }) {
  const family = series === "dpmm" ? "DPMM" : "Topic";
  const compLabel = series === "dpmm" ? "Latent Dim" : "Topic";
  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>
      <Phase label="LATENT EXTRACTION" color="#457B9D" details="Extract latent representations from trained models">
        <Step icon={Icons.model} label={`${family} Models`} sub="Pre-trained" />
        <Arrow />
        <Step icon={Icons.cells} label="Latent Matrix" sub="Z ∈ ℝⁿˣᵈ" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          z_i = f_θ(x_i)
        </div>
      </Phase>

      <Arrow />

      <Phase label="CORRELATION" color="#E76F51" details="Pearson r per gene × latent dim; genes ordered by dominant component (block-diagonal)">
        <Step icon={Icons.gradient} label="Pearson r" sub={`Gene × ${compLabel}`} />
        <Arrow />
        <Step icon={Icons.filter} label="Top 30 Genes" sub="argmax |r| grouping" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          sort by (argmax_k |r_kj|, −|r|)
        </div>
      </Phase>

      <Arrow />

      <Phase label="VISUALIZATION" color="#2A9D8F" details="Block-diagonal heatmap reveals per-component gene signatures">
        <Step icon={Icons.heatmap} label="Block-Diag. Map" sub="coolwarm ±r" />
        <Arrow />
        <Step icon={Icons.dotplot} label="Cross-validate" sub="vs. Perturbation (Fig 6)" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          block-diag heatmap
        </div>
      </Phase>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Figure 8: Latent UMAP Projection Workflow
// ═══════════════════════════════════════════════════════════════════════════════
function Fig8Flow({ series }: { series: string }) {
  const family = series === "dpmm" ? "DPMM" : "Topic";
  const compLabel = series === "dpmm" ? "Dim" : "Topic";
  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>
      <Phase label="LATENT EXTRACTION" color="#457B9D" details="Encode test cells via trained models">
        <Step icon={Icons.model} label={`${family} Encoder`} sub="f_θ(x) → z" />
        <Arrow />
        <Step icon={Icons.cells} label="Latent Z" sub="Z ∈ ℝⁿˣᵈ" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          z_i = μ_θ(x_i)
        </div>
      </Phase>

      <Arrow />

      <Phase label="UMAP EMBEDDING" color="#9C27B0" details="Dimensionality reduction preserving local structure">
        <Step icon={Icons.preprocess} label="UMAP" sub="n_neighbors=15" />
        <Arrow />
        <Step icon={Icons.cells} label="2D Coords" sub="ℝⁿˣ²" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          U = UMAP(Z; k=15, d_min=0.3)
        </div>
      </Phase>

      <Arrow />

      <Phase label="GENE CORRELATION" color="#E76F51" details="Pearson r between latent dimensions and gene expression">
        <Step icon={Icons.gradient} label="Pearson r" sub={`r(z_k, x_j)`} />
        <Arrow />
        <Step icon={Icons.filter} label="argmax r" sub="r > 0 only" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          g*_k = argmax_j r(z_k, x_j)
        </div>
      </Phase>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Figure 9: GO Enrichment Workflow
// ═══════════════════════════════════════════════════════════════════════════════
function Fig9Flow({ series }: { series: string }) {
  return (
    <div style={{ display: "flex", alignItems: "stretch", gap: "3px", padding: "4px 1px 1px 1px", width: "100%" }}>
      <Phase label="GENE SELECTION" color="#E76F51" details="Extract top drivers from perturbation importance">
        <Step icon={Icons.heatmap} label="Importance" sub="From Figure 6" />
        <Arrow />
        <Step icon={Icons.filter} label="Top 50 Genes" sub="Per component" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          G_k = top_K(I_k)
        </div>
      </Phase>

      <Arrow />

      <Phase label="ENRICHMENT" color="#457B9D" details="Query databases for biological pathways">
        <Step icon={Icons.database} label="Enrichr API" sub="GO Biol. Process" />
        <Arrow />
        <Step icon={Icons.stats} label="Fisher Test" sub="P-value &lt; 0.05" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          P(X ≥ k)
        </div>
      </Phase>

      <Arrow />

      <Phase label="VALIDATION" color="#2A9D8F" details="Confirm biological relevance of terms">
        <Step icon={Icons.dotplot} label="Dot Plot" sub="-log(P) · Ratio" />
        <Arrow />
        <Step icon={Icons.cells} label="Bio Context" sub="Literature Support" />
        <div style={{ fontSize: "5.5px", color: CLR.textSub, marginTop: "1px", fontStyle: "italic", fontFamily: "serif" }}>
          -log₁₀(p_adj)
        </div>
      </Phase>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Export
// ═══════════════════════════════════════════════════════════════════════════════
interface BioWorkflowPanelProps {
  figNum: 6 | 7 | 8 | 9;
  series: string;
}

export default function BioWorkflowPanel({ figNum, series }: BioWorkflowPanelProps) {
  const flows: Record<number, React.ReactNode> = {
    6: <Fig6Flow series={series} />,
    7: <Fig7Flow series={series} />,   // Latent–gene Pearson correlation
    8: <Fig8Flow series={series} />,   // Latent UMAP projections
    9: <Fig9Flow series={series} />,   // GO enrichment
  };

  return (
    <PanelSection label="A" title="Workflow">
      {flows[figNum]}
    </PanelSection>
  );
}
