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
import BioWorkflowPanel from "@/components/figures/BioWorkflowPanel";

/*  Figure 6: Biological Validation
 *
 *  Panel A: Workflow
 *  Panel B: Gene importance heatmaps (z-scored, 3-col per dataset)
 *  Panel C: Topic decoder β heatmaps (Topic series only)
 */

interface Props {
  series: string;
}

interface HeatEntry {
  file: string;
  model: string;
}

interface Manifest {
  panelA?: Record<string, HeatEntry[]>;
  panelC_beta?: Record<string, HeatEntry[]>;
  models?: string[];
  datasets?: string[];
  image_sizes?: Record<string, { w: number; h: number }>;
}

/* Helper: render a dataset→entries panel section with consistent styling */
function DatasetPanelGrid({
  data,
  datasets,
  series,
  columns = 3,
}: {
  data: Record<string, HeatEntry[]>;
  datasets: string[];
  series: string;
  columns?: number;
}) {
  return (
    <>
      {datasets.map((ds) => {
        const entries = data[ds] ?? [];
        if (entries.length === 0) return null;
        return (
          <div key={ds} className="mb-0.5">
            <div
              className="text-[7px] text-gray-500 pl-3 mb-0"
              style={{ fontFamily: "Arial, sans-serif" }}
            >
              {ds}
            </div>
            <SubplotGrid
              columns={Math.min(entries.length, columns)}
              gap="3px"
            >
              {entries.map((e) => (
                <SubplotImage
                  key={e.file}
                  src={subplotPath(series, 6, e.file)}
                  alt={`${e.model} — ${ds}`}
                />
              ))}
            </SubplotGrid>
          </div>
        );
      })}
    </>
  );
}

function Figure6Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig6/manifest.json`)
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
    panelA = {},
    panelC_beta: panelCBeta = {},
    datasets = [],
  } = manifest;

  // Use the canonical dataset order from manifest; for Panel C derive
  // available datasets from panelCBeta keys, falling back to datasets list.
  const betaDatasets = (
    ["setty", "endo", "dentate"].filter(
      (ds) => (panelCBeta as Record<string, HeatEntry[]>)[ds]?.length > 0
    )
  );
  // If betaDatasets is empty, fall back to datasets
  const panelCDatasetsToShow = betaDatasets.length > 0 ? betaDatasets : datasets;

  const hasBeta = betaDatasets.length > 0;

  return (
    <FigureContainer figureId="figure6" series={series}>
      <BioWorkflowPanel figNum={6} series={series} />

      {hasBeta ? (
        /* Topic series: B+C side-by-side to keep height ≤ 21cm */
        <PanelColumns columns={2} gap="4px">
          <PanelSection label="B" title="Gene Importance Heatmaps">
            <DatasetPanelGrid
              data={panelA as Record<string, HeatEntry[]>}
              datasets={datasets}
              series={series}
              columns={3}
            />
          </PanelSection>
          <PanelSection label="C" title="Topic Decoder β">
            <DatasetPanelGrid
              data={panelCBeta as Record<string, HeatEntry[]>}
              datasets={panelCDatasetsToShow}
              series={series}
              columns={3}
            />
          </PanelSection>
        </PanelColumns>
      ) : (
        /* DPMM series: single Panel B */
        <PanelSection label="B" title="Gene Importance Heatmaps">
          <DatasetPanelGrid
            data={panelA as Record<string, HeatEntry[]>}
            datasets={datasets}
            series={series}
            columns={3}
          />
        </PanelSection>
      )}
    </FigureContainer>
  );
}

export default Figure6Content;
