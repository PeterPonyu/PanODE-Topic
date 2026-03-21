"use client";

import React, { useEffect, useState } from "react";
import {
  FigureContainer,
  PanelSection,
  SubplotGrid,
  SubplotImage,
  subplotPath,
} from "@/components/figures/FigureComponents";
import BioWorkflowPanel from "@/components/figures/BioWorkflowPanel";

/*  Figure 7: Latent–Gene Pearson Correlation
 *
 *  Panel A: Workflow
 *  Panel B: Correlation heatmaps (datasets × models, 3-col per dataset)
 */

interface Props {
  series: string;
}

interface CorrEntry {
  file: string;
  model: string;
}

interface Manifest {
  panelA?: Record<string, CorrEntry[]>;
  models?: string[];
  datasets?: string[];
  image_sizes?: Record<string, { w: number; h: number }>;
}

function Figure7Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig7/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setManifest)
      .catch((e) => setError(e.message));
  }, [series]);

  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!manifest) return <div className="p-4 text-gray-400">Loading…</div>;

  const { panelA = {}, datasets = [] } = manifest;

  return (
    <FigureContainer figureId="figure7" series={series}>
      <BioWorkflowPanel figNum={7} series={series} />

      <PanelSection label="B" title="Latent–Gene Pearson Correlation">
        {datasets.map((ds) => {
          const entries = (panelA as Record<string, CorrEntry[]>)[ds] ?? [];
          if (entries.length === 0) return null;
          return (
            <div key={ds} className="mb-0.5">
              <div
                className="text-[7px] text-gray-500 pl-3 mb-0"
                style={{ fontFamily: "Arial, sans-serif" }}
              >
                {ds}
              </div>
              <SubplotGrid columns={Math.min(entries.length, 3)} gap="3px">
                {entries.map((e) => (
                  <SubplotImage
                    key={e.file}
                    src={subplotPath(series, 7, e.file)}
                    alt={`${e.model} — ${ds}`}
                  />
                ))}
              </SubplotGrid>
            </div>
          );
        })}
      </PanelSection>
    </FigureContainer>
  );
}

export default Figure7Content;
