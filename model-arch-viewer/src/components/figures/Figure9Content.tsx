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

/*  Figure 9: GO / BP Enrichment Analysis
 *
 *  Panel A: Workflow
 *  Panel B: GO enrichment dot plots (full width, 3-col per dataset)
 *
 *  Enrichment data: gradient saliency → top genes → Enrichr API → adj. p filter.
 *  Spacing increased between dataset groups for clarity.
 */

interface Props {
  series: string;
}

interface EnrichEntry {
  file: string;
  model: string;
}

interface Manifest {
  panelA?: Record<string, EnrichEntry[]>;
  panelBeta?: Record<string, EnrichEntry[]>;
  models?: string[];
  datasets?: string[];
  image_sizes?: Record<string, { w: number; h: number }>;
}

function Figure9Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig9/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setManifest)
      .catch((e) => setError(e.message));
  }, [series]);

  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!manifest) return <div className="p-4 text-gray-400">Loading…</div>;

  const { panelA = {}, panelBeta = {}, datasets = [] } = manifest;
  const betaDatasets = Object.keys(panelBeta);

  return (
    <FigureContainer figureId="figure9" series={series}>
      <BioWorkflowPanel figNum={9} series={series} />

      <PanelSection label="B" title="GO / BP Enrichment (Perturbation)">
        {datasets.map((ds) => {
          const entries = (panelA as Record<string, EnrichEntry[]>)[ds] ?? [];
          if (entries.length === 0) return null;
          return (
            <div key={ds} className="mb-0.5">
              <div
                className="text-[7px] text-gray-500 pl-3 mb-0"
                style={{ fontFamily: "Arial, sans-serif" }}
              >
                {ds}
              </div>
              <SubplotGrid columns={Math.min(entries.length, 3)} gap="2px">
                {entries.map((e) => (
                  <SubplotImage
                    key={e.file}
                    src={subplotPath(series, 9, e.file)}
                    alt={`${e.model} — ${ds}`}
                  />
                ))}
              </SubplotGrid>
            </div>
          );
        })}
      </PanelSection>

      {/* Panel C — Beta decoder enrichment (best Topic variant only) */}
      {betaDatasets.length > 0 && (
        <PanelSection label="C" title="GO / BP Enrichment (Topic β Decoder — Best Variant)">
          <SubplotGrid columns={betaDatasets.length} gap="4px">
            {betaDatasets.map((ds) => {
              const entries = panelBeta[ds] ?? [];
              if (entries.length === 0) return null;
              return entries.map((e) => (
                <div key={e.file}>
                  <div
                    className="text-[7px] text-gray-500 pl-1 mb-0.5"
                    style={{ fontFamily: "Arial, sans-serif" }}
                  >
                    {ds}
                  </div>
                  <SubplotImage
                    src={subplotPath(series, 9, e.file)}
                    alt={`${e.model} β — ${ds}`}
                  />
                </div>
              ));
            })}
          </SubplotGrid>
        </PanelSection>
      )}
    </FigureContainer>
  );
}

export default Figure9Content;
