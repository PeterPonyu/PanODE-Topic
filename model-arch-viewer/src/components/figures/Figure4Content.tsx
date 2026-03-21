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

/*  Figure 4: Training / Sweep UMAP embeddings — 17:21 multi-column layout
 *
 *  Panel A: Workflow (rendered by WorkflowPanel)
 *  Panels B–K: 10 sweep parameters in 3-column rows.
 *  Each panel: dataset rows × sweep-value UMAPs in 4-col grid.
 */

interface Props {
  series: string;
}

interface SnapFile {
  file: string;
  label: string;
}

interface ParamInfo {
  source: string;
  datasets: Record<string, SnapFile[]>;
}

interface Manifest {
  params?: Record<string, ParamInfo>;
  image_sizes?: Record<string, { w: number; h: number }>;
}

function Figure4Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig4/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setManifest)
      .catch((e) => setError(e.message));
  }, [series]);

  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!manifest) return <div className="p-4 text-gray-400">Loading…</div>;

  const params = manifest.params ?? {};
  const image_sizes = manifest.image_sizes ?? {};
  const sz = (f: string) => image_sizes[f] ?? {};
  const paramNames = Object.keys(params);
  const labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  // Group params into triplets for 3-column layout
  const rows: (string | null)[][] = [];
  for (let i = 0; i < paramNames.length; i += 3) {
    rows.push([
      paramNames[i],
      paramNames[i + 1] ?? null,
      paramNames[i + 2] ?? null,
    ]);
  }

  const renderParam = (paramName: string, labelIdx: number) => {
    const info = params[paramName];
    const dsNames = Object.keys(info.datasets);
    return (
      <PanelSection
        label={labels[labelIdx] ?? ""}
        title={`${paramName} (${info.source})`}
      >
        {dsNames.map((dsName) => {
          const snaps = info.datasets[dsName];
          return (
            <div key={dsName} className="mb-0">
              <div
                className="text-[7px] text-gray-500 pl-3 mb-0"
                style={{ fontFamily: "Arial, sans-serif" }}
              >
                {dsName}
              </div>
              <SubplotGrid columns={Math.min(snaps.length, 4)} gap="1px">
                {snaps.map((snap) => (
                  <SubplotImage
                    key={snap.file}
                    src={subplotPath(series, 4, snap.file)}
                    alt={snap.label}
                    naturalWidth={sz(snap.file).w}
                    naturalHeight={sz(snap.file).h}
                  />
                ))}
              </SubplotGrid>
            </div>
          );
        })}
      </PanelSection>
    );
  };

  return (
    <FigureContainer figureId="figure4" series={series}>
      {/* Panel A: Workflow */}
      <EnhancedWorkflowPanel figNum={4} series={series} />

      {rows.map((group, rowIdx) => (
        <PanelColumns key={rowIdx} columns={3} gap="3px">
          {group.map((p, ci) =>
            p ? (
              <div key={p}>{renderParam(p, rowIdx * 3 + ci + 1)}</div>
            ) : (
              <div key={`empty-${ci}`} />
            )
          )}
        </PanelColumns>
      ))}
    </FigureContainer>
  );
}

export default Figure4Content;
