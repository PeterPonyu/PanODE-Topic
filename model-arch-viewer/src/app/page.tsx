"use client";

import React, { useRef, useState } from "react";
import { ALL_MODELS, type ModelArch } from "@/data/models";
import ArchDiagram from "@/components/ArchDiagram";
import ExportBar from "@/components/ExportBar";

/* ── colour accents per series ── */
const SERIES_ACCENT: Record<string, string> = {
  dpmm: "border-indigo-500",
  topic: "border-orange-500",
};
const VARIANT_BADGE: Record<string, string> = {
  base: "bg-gray-200 text-gray-700",
  transformer: "bg-sky-200 text-sky-800",
  contrastive: "bg-amber-200 text-amber-800",
};

export default function Home() {
  const [selected, setSelected] = useState<ModelArch>(ALL_MODELS[0]);
  const diagramRef = useRef<HTMLDivElement>(null);

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* ──────────── Sidebar ──────────── */}
      <aside className="w-64 bg-white border-r border-gray-200 p-3 shrink-0">
        <h2 className="text-lg font-bold text-gray-800 mb-1">PanODE</h2>
        <p className="text-xs text-gray-500 mb-4">
          Model Architecture Viewer — Fig.&nbsp;1
        </p>

        {/* Series groups */}
        {(["dpmm", "topic"] as const).map((series) => (
          <div key={series} className="mb-4">
            <div className="text-[10px] uppercase tracking-wider font-bold text-gray-400 mb-1">
              {series.toUpperCase()} Series
            </div>
            {ALL_MODELS.filter((m) => m.series === series).map((m) => (
              <button
                key={m.id}
                onClick={() => setSelected(m)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm mb-1
                  border-l-4 transition-all cursor-pointer
                  ${
                    selected.id === m.id
                      ? `${SERIES_ACCENT[series]} bg-gray-100 font-semibold`
                      : "border-transparent hover:bg-gray-50"
                  }`}
              >
                <span>{m.displayName}</span>
                <span
                  className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full ${
                    VARIANT_BADGE[m.variant]
                  }`}
                >
                  {m.variant}
                </span>
              </button>
            ))}
          </div>
        ))}

        <div className="border-t border-gray-200 pt-2 mt-2 text-[11px] text-gray-500 leading-snug">
          ODE module is intentionally hidden at this stage.
          This viewer focuses on Phase-1 encoder/prior/decoder architecture for paper figures.
        </div>
      </aside>

      {/* ──────────── Main area ──────────── */}
      <main className="flex-1 p-5 overflow-y-auto">
        <div className="max-w-[1400px] mx-auto">
          {/* Header */}
          <div className="mb-4 text-center">
            <h1 className="text-2xl font-bold text-gray-800">
              {selected.displayName}
            </h1>
            <div className="flex items-center justify-center gap-2 mt-1">
              <span
                className={`text-xs px-2 py-0.5 rounded-full ${
                  VARIANT_BADGE[selected.variant]
                }`}
              >
                {selected.variant}
              </span>
              <span className="text-xs text-gray-500">
                {selected.series.toUpperCase()} series
              </span>
              <span className="text-xs text-gray-400 font-mono">
                ~{selected.paramCount}
              </span>
            </div>
          </div>

          {/* Diagram export target */}
          <div
            ref={diagramRef}
            className="bg-white rounded-xl border border-gray-200 shadow-sm p-4"
          >
            <ArchDiagram model={selected} />
          </div>

          {/* Export buttons */}
          <ExportBar targetRef={diagramRef} fileName={selected.id} />

          {/* Key Comparison Table (short) */}
          <div className="mt-5 bg-white rounded-xl border border-gray-200 shadow-sm p-3 overflow-x-auto">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">
              Quick Comparison
            </h3>
            <table className="w-full text-xs text-left">
              <thead className="text-gray-400 uppercase text-[10px]">
                <tr>
                  <th className="pb-2">Model</th>
                  <th className="pb-2">Latent</th>
                  <th className="pb-2">Prior</th>
                  <th className="pb-2">Decoder</th>
                  <th className="pb-2">Params</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {ALL_MODELS.map((m) => (
                  <tr
                    key={m.id}
                    className={
                      m.id === selected.id ? "bg-indigo-50 font-medium" : ""
                    }
                  >
                    <td className="py-1.5 pr-2 whitespace-nowrap">
                      {m.displayName}
                    </td>
                    <td className="py-1.5 pr-2">{m.latentSpace.type}</td>
                    <td className="py-1.5 pr-2">{m.prior.name.split("(")[0]}</td>
                    <td className="py-1.5 pr-2">
                      {m.series === "topic" ? "β matrix" : "MLP"}
                    </td>
                    <td className="py-1.5 font-mono">{m.paramCount}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}
