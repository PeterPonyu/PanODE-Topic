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

/*  Figure 3: Sensitivity Analysis — 17:21 multi-column panel layout
 *
 *  Panel A: Workflow (rendered by WorkflowPanel)
 *  Panels B–K: 10 sweep parameters in 3-column rows.
 *  Each sweep: 3-col metric boxplots.
 */

interface Props {
  series: string;
}

interface SweepInfo {
  source: string;
  plots: string[];
}

interface Manifest {
  sweeps?: Record<string, SweepInfo>;
  image_sizes?: Record<string, { w: number; h: number }>;
}

function Figure3Content({ series }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/subplots/${series}/fig3/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setManifest)
      .catch((e) => setError(e.message));
  }, [series]);

  if (error) return <div className="p-4 text-red-500">Error: {error}</div>;
  if (!manifest) return <div className="p-4 text-gray-400">Loading…</div>;

  const sweeps = manifest.sweeps ?? {};
  const image_sizes = manifest.image_sizes ?? {};
  const sz = (f: string) => image_sizes[f] ?? {};
  const sweepNames = Object.keys(sweeps);
  const labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

  // Group sweeps into triplets for 3-column panel layout
  const rows: (string | null)[][] = [];
  for (let i = 0; i < sweepNames.length; i += 3) {
    rows.push([
      sweepNames[i],
      sweepNames[i + 1] ?? null,
      sweepNames[i + 2] ?? null,
    ]);
  }

  return (
    <FigureContainer figureId="figure3" series={series}>
      {/* Panel A: Workflow */}
      <EnhancedWorkflowPanel figNum={3} series={series} />

      {rows.map((group, rowIdx) => (
        <PanelColumns key={rowIdx} columns={3} gap="3px">
          {group.map((sweepName, ci) => {
            if (!sweepName) return <div key={`empty-${ci}`} />;
            const info = sweeps[sweepName];
            const labelIdx = rowIdx * 3 + ci + 1; // +1: Panel A is Workflow
            return (
              <PanelSection
                key={sweepName}
                label={labels[labelIdx] ?? ""}
                title={`${sweepName}`}
              >
                <SubplotGrid columns={series === "topic" ? 2 : 3} gap="1px">
                  {info.plots.map((f) => (
                    <SubplotImage
                      key={f}
                      src={subplotPath(series, 3, f)}
                      alt={f}
                      naturalWidth={sz(f).w}
                      naturalHeight={sz(f).h}
                    />
                  ))}
                </SubplotGrid>
              </PanelSection>
            );
          })}
        </PanelColumns>
      ))}
    </FigureContainer>
  );
}

export default Figure3Content;
