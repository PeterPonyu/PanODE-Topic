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

/*  Figure 10: External Model Benchmark Comparison
 *
 *  Panel A: Workflow (selection → benchmark → comparison)
 *  Panel B: 6 core metric boxplots (NMI, ARI, ASW, DAV, DRE UMAP, LSE Overall)
 *           comparing 11 external models + Best-DPMM/Topic  (3-col grid)
 *           Internal best now has seed-level variance (up to 60 data points)
 *  Panel C: Extended metric boxplots (same suite as Fig 2 Panel D)  (8-col grid)
 *  Panel D: Efficiency metric boxplots (SecPerEpoch, PeakGPU, Params) (3-col)
 *  Panel E: Aggregate ranking bar chart + Wilcoxon/Cliff's δ significance heatmap
 *
 *  Uses a distinct color palette (warm red-blue-earth tones)
 *  to differentiate from internal model comparisons.
 */

interface Props {
  series: string;
}

interface Manifest {
  panelA?: string;
  panelB?: string[];
  panelC?: string[];
  panelD?: string[];
  panelE?: string;
  legend?: string;
  image_sizes?: Record<string, { w: number; h: number }>;
}

function Figure10Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig10/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setManifest)
      .catch((e) => setError(e.message));
  }, [series]);

  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!manifest) return <div className="p-4 text-gray-400">Loading…</div>;

  const coreBoxplots = manifest.panelB ?? [];
  const extBoxplots = manifest.panelC ?? [];
  const effBoxplots = manifest.panelD ?? [];
  const rankingFile = manifest.panelE;
  const legendFile = manifest.legend;
  const image_sizes = manifest.image_sizes ?? {};
  const sz = (f: string) => image_sizes[f] ?? {};

  /* Sort core boxplots by canonical order */
  const coreOrder = ["NMI", "ARI", "ASW", "DAV", "DRE_umap", "LSE_overall"];
  const sortedCore = [...coreBoxplots].sort((a, b) => {
    const ai = coreOrder.findIndex((m) => a.includes(m));
    const bi = coreOrder.findIndex((m) => b.includes(m));
    return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
  });

  return (
    <FigureContainer figureId="figure10" series={series}>
      {/* Panel A: Workflow */}
      <EnhancedWorkflowPanel figNum={10} series={series} />

      {/* Panel B: Core metric boxplots (6 metrics / 3 cols for dpmm; 4 metrics / 4 cols for topic) */}
      <PanelSection label="B" title="Core Metrics — External vs Internal">
        <SubplotGrid columns={series === "topic" ? 4 : 3} gap="3px">
          {sortedCore.map((f) => (
            <SubplotImage
              key={f}
              src={subplotPath(series, 10, f)}
              alt={f.replace("core_", "").replace(".png", "")}
              naturalWidth={sz(f).w}
              naturalHeight={sz(f).h}
            />
          ))}
        </SubplotGrid>
      </PanelSection>

      {/* Panel C: Extended metric boxplots (8-col / 24 metrics for dpmm; 2-col / 2 metrics for topic) */}
      {extBoxplots.length > 0 && (
        <PanelSection label="C" title="Extended Metrics">
          <SubplotGrid columns={series === "topic" ? 2 : 8} gap="2px">
            {extBoxplots.map((f) => (
              <SubplotImage
                key={f}
                src={subplotPath(series, 10, f)}
                alt={f.replace("ext_", "").replace(".png", "")}
                naturalWidth={sz(f).w}
                naturalHeight={sz(f).h}
              />
            ))}
          </SubplotGrid>
        </PanelSection>
      )}

      {/* Panels D + E side by side: D has efficiency, E has ranking + significance */}
      <PanelColumns columns={2} gap="4px">
        {/* Panel D: Efficiency metric boxplots */}
        {effBoxplots.length > 0 && (
          <PanelSection label="D" title="Efficiency">
            <SubplotGrid columns={2} gap="4px">
              {effBoxplots.map((f) => (
                <SubplotImage
                  key={f}
                  src={subplotPath(series, 10, f)}
                  alt={f.replace("eff_", "").replace(".png", "")}
                  naturalWidth={sz(f).w}
                  naturalHeight={sz(f).h}
                />
              ))}
            </SubplotGrid>
          </PanelSection>
        )}

        {/* Panel E: Aggregate ranking + significance heatmap */}
        <PanelSection label="E" title="Aggregate Ranking + Significance">
          <SubplotGrid columns={2} gap="4px">
            {rankingFile && (
              <SubplotImage
                src={subplotPath(series, 10, rankingFile)}
                alt="Aggregate ranking"
                naturalWidth={sz(rankingFile).w}
                naturalHeight={sz(rankingFile).h}
              />
            )}
            <SubplotImage
              src={`/statistical/wilcoxon_cliffs_delta_heatmap_${series}.png`}
              alt={`Wilcoxon + Cliff's delta heatmap (${series})`}
            />
          </SubplotGrid>
        </PanelSection>
      </PanelColumns>

      {/* Legend */}
      {legendFile && (
        <img
          src={subplotPath(series, 10, legendFile)}
          alt="Legend"
          className="w-full h-auto mt-0"
          style={{
            imageRendering: "auto",
            maxWidth: "70%",
            margin: "0 auto",
            display: "block",
          }}
          loading="eager"
        />
      )}
    </FigureContainer>
  );
}

export default Figure10Content;
