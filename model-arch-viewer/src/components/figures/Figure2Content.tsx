"use client";

import React, { useEffect, useState } from "react";
import {
  FigureContainer,
  PanelSection,
  PanelColumns,
  SubplotGrid,
  SubplotImage,
  subplotPath,
} from "@/components/figures/FigureComponents";
import EnhancedWorkflowPanel from "@/components/figures/EnhancedWorkflowPanel";

/*  Figure 2: Base vs Ablation — 17:21 multi-column panel layout
 *
 *  Panel A: Workflow (rendered by WorkflowPanel)
 *  Row 1: [Panel B (Indep UMAPs)] | [Panel B2 (Joint UMAPs)] | [Panel C (Core)] | [Panel D (Efficiency)]
 *  Row 2: Panel E — Extended metrics (full width, 8 columns)
 *
 *  Data: 5-seed combined CSV (60 data points per model in boxplots).
 *  Core boxplots include Wilcoxon significance brackets (structured vs pure counterpart).
 */

interface Props {
  series: string;
}

interface Manifest {
  panelA?: string[];
  panelA_legend?: string;
  panelA_joint?: string[];
  panelA_joint_legend?: string;
  panelB?: string[];
  panelC?: string[];
  panelD?: string[];
  image_sizes?: Record<string, { w: number; h: number }>;
}

function Figure2Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig2/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setManifest)
      .catch((e) => setError(e.message));
  }, [series]);

  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!manifest) return <div className="p-4 text-gray-400">Loading…</div>;

  const {
    panelA = [], panelA_legend, panelA_joint = [], panelA_joint_legend,
    panelB = [], panelC = [], panelD = [],
    image_sizes = {},
  } = manifest;
  const sz = (f: string) => image_sizes[f] ?? {};

  return (
    <FigureContainer figureId="figure2" series={series}>
      {/* Panel A: Workflow */}
      <EnhancedWorkflowPanel figNum={2} series={series} />

      {/* Row 1: Panel B (Indep UMAPs) | Panel B2 (Joint UMAPs) | Panel C (Core) | Panel D (Efficiency)
         — B and B2 are co-equal main panels; layout adapts if joint data is absent */}
      <PanelColumns columns={panelA_joint.length > 0 ? 4 : 3} gap="3px">
        {/* Panel B: per-dataset independent training, latents concatenated */}
        <PanelSection label="B" title="Latent Space (Independent)">
          <SubplotGrid columns={3} gap="1px">
            {panelA.map((f) => (
              <SubplotImage key={f} src={subplotPath(series, 2, f)} alt={f}
                naturalWidth={sz(f).w} naturalHeight={sz(f).h} />
            ))}
          </SubplotGrid>
          {panelA_legend && (
            <img
              src={subplotPath(series, 2, panelA_legend)}
              alt="UMAP legend"
              className="w-full h-auto mt-0"
              style={{ imageRendering: "auto" }}
              loading="eager"
            />
          )}
        </PanelSection>

        {/* Panel B2: Joint model trained on all datasets simultaneously */}
        {panelA_joint.length > 0 && (
          <PanelSection label="B2" title="Latent Space (Joint)">
            <SubplotGrid columns={3} gap="1px">
              {panelA_joint.map((f) => (
                <SubplotImage key={f} src={subplotPath(series, 2, f)} alt={f}
                  naturalWidth={sz(f).w} naturalHeight={sz(f).h} />
              ))}
            </SubplotGrid>
            {panelA_joint_legend && (
              <img
                src={subplotPath(series, 2, panelA_joint_legend)}
                alt="Joint UMAP legend"
                className="w-full h-auto mt-0"
                style={{ imageRendering: "auto" }}
                loading="eager"
              />
            )}
          </PanelSection>
        )}

        {/* Panel C: core boxplots (Topic: unified 6 metrics in 3 cols; DPMM: core only in 3 cols) */}
        <PanelSection label="C" title={series === "topic" ? "Metrics" : "Core Metrics"}>
          <SubplotGrid columns={3} gap="1px">
            {panelB.map((f) => (
              <SubplotImage key={f} src={subplotPath(series, 2, f)} alt={f}
                naturalWidth={sz(f).w} naturalHeight={sz(f).h} />
            ))}
          </SubplotGrid>
        </PanelSection>

        {/* Panel D: efficiency */}
        {panelD.length > 0 && (
          <PanelSection label="D" title="Efficiency">
            <SubplotGrid columns={3} gap="1px">
              {panelD.map((f) => (
                <SubplotImage key={f} src={subplotPath(series, 2, f)} alt={f}
                  naturalWidth={sz(f).w} naturalHeight={sz(f).h} />
              ))}
            </SubplotGrid>
          </PanelSection>
        )}
      </PanelColumns>

      {/* Row 3: Panel E — Extended metrics (DPMM only; Topic merges into Panel C) */}
      {panelC.length > 0 && series !== "topic" && (
        <PanelSection label="E" title="Extended Metrics">
          <SubplotGrid columns={8} gap="1px">
            {panelC.map((f) => (
              <SubplotImage key={f} src={subplotPath(series, 2, f)} alt={f}
                naturalWidth={sz(f).w} naturalHeight={sz(f).h} />
            ))}
          </SubplotGrid>
        </PanelSection>
      )}
    </FigureContainer>
  );
}

export default Figure2Content;
