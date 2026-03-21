"use client";

import React, { useEffect, useState } from "react";
import {
  FigureContainer,
  PanelSection,
  SubplotGrid,
  SubplotImage,
  subplotPath,
} from "@/components/figures/FigureComponents";
import EnhancedWorkflowPanel from "@/components/figures/EnhancedWorkflowPanel";

/*  Figure 5: Cross-dataset scatter + convex-hull plots
 *
 *  Panel A: Workflow (rendered by WorkflowPanel)
 *  Panel B: Grid of metric-pair scatter plots with convex hulls.
 */

interface Props {
  series: string;
}

interface ScatterInfo {
  file: string;
  x_label: string;
  y_label: string;
}

interface Manifest {
  scatters?: ScatterInfo[];
  legend?: string;
  image_sizes?: Record<string, { w: number; h: number }>;
}

function Figure5Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig5/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setManifest)
      .catch((e) => setError(e.message));
  }, [series]);

  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!manifest) return <div className="p-4 text-gray-400">Loading…</div>;

  const scatters = manifest.scatters ?? [];
  const legendFile = manifest.legend;
  const image_sizes = manifest.image_sizes ?? {};
  const sz = (f: string) => image_sizes[f] ?? {};

  // Auto columns: 2 for ≤4, 3 for ≤9, 4 for many
  const cols = scatters.length <= 4 ? 2 : scatters.length > 9 ? 4 : 3;

  return (
    <FigureContainer figureId="figure5" series={series}>
      {/* Panel A: Workflow */}
      <EnhancedWorkflowPanel figNum={5} series={series} />

      <PanelSection label="B" title="Metric Trade-offs">
        <SubplotGrid columns={cols} gap="3px">
          {scatters.map((s) => (
            <SubplotImage
              key={s.file}
              src={subplotPath(series, 5, s.file)}
              alt={`${s.x_label} vs ${s.y_label}`}
              naturalWidth={sz(s.file).w}
              naturalHeight={sz(s.file).h}
            />
          ))}
        </SubplotGrid>
        {/* Shared legend row */}
        {legendFile && (
          <img
            src={subplotPath(series, 5, legendFile)}
            alt="Legend"
            className="w-full h-auto mt-0"
            style={{ imageRendering: "auto", maxWidth: "50%", margin: "0 auto", display: "block" }}
            loading="eager"
          />
        )}
      </PanelSection>
    </FigureContainer>
  );
}

export default Figure5Content;
