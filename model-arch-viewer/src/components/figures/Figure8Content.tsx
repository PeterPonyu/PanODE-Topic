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

/*  Figure 8: Latent UMAP Projections
 *
 *  Panel A: Computational workflow (math-based, not assembly)
 *  Per dataset (setty, endo, dentate):
 *    Row 1 — Component projection intensity UMAP (3 models)
 *    Row 2 — Top positively correlated gene expression UMAP (3 models)
 *
 *  All 3 representative datasets × 3 models per series.
 *  Each subplot is a 2×3 grid showing 6 latent components.
 */

interface Props {
  series: string;
}

interface UmapEntry {
  file: string;
  model: string;
}

interface Manifest {
  comp_umap?: Record<string, UmapEntry[]>;
  gene_umap?: Record<string, UmapEntry[]>;
  models?: string[];
  datasets?: string[];
  image_sizes?: Record<string, { w: number; h: number }>;
}

const DS_LABELS: Record<string, string> = {
  setty: "Setty (Hematopoiesis)",
  endo: "Endo (Endocrinogenesis)",
  dentate: "Dentate (Dentate Gyrus)",
};

function Figure8Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig8/manifest.json`)
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
    comp_umap: compUmap = {},
    gene_umap: geneUmap = {},
    datasets = [],
  } = manifest;

  return (
    <FigureContainer figureId="figure8" series={series}>
      <BioWorkflowPanel figNum={8} series={series} />

      {/* Per-dataset paired panels: intensity + gene expression */}
      {datasets.map((ds, idx) => {
        const compEntries = (compUmap as Record<string, UmapEntry[]>)[ds] ?? [];
        const geneEntries = (geneUmap as Record<string, UmapEntry[]>)[ds] ?? [];
        if (compEntries.length === 0 && geneEntries.length === 0) return null;
        const panelLetter = String.fromCharCode(66 + idx * 2); // B, D, F
        const panelLetter2 = String.fromCharCode(67 + idx * 2); // C, E, G
        const dsLabel = DS_LABELS[ds] || ds;
        return (
          <div key={ds} className="mb-0.5">
            <div
              className="text-[8px] text-gray-700 font-semibold pl-1 mb-0"
              style={{ fontFamily: "Arial, sans-serif" }}
            >
              {dsLabel}
            </div>
            {/* Component intensity row */}
            {compEntries.length > 0 && (
              <PanelSection label={panelLetter} title="Component Projection Intensity">
                <SubplotGrid columns={3} gap="2px">
                  {compEntries.map((e) => (
                    <SubplotImage
                      key={e.file}
                      src={subplotPath(series, 8, e.file)}
                      alt={`${e.model} — ${ds} — component UMAP`}
                    />
                  ))}
                </SubplotGrid>
              </PanelSection>
            )}
            {/* Gene expression row */}
            {geneEntries.length > 0 && (
              <PanelSection label={panelLetter2} title="Top Positively Correlated Gene Expression">
                <SubplotGrid columns={3} gap="2px">
                  {geneEntries.map((e) => (
                    <SubplotImage
                      key={e.file}
                      src={subplotPath(series, 8, e.file)}
                      alt={`${e.model} — ${ds} — gene UMAP`}
                    />
                  ))}
                </SubplotGrid>
              </PanelSection>
            )}
          </div>
        );
      })}
    </FigureContainer>
  );
}

export default Figure8Content;
