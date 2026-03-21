"use client";
import React from "react";
import type { LayerDef, LossDef } from "@/data/models";

/* ── Compact layer block ─────────────────────────────────────── */
export function LayerBlock({ layer, idx }: { layer: LayerDef; idx: number }) {
  return (
    <div
      className={`${layer.color} border border-gray-200/80 rounded px-1.5 py-1
                   shadow-sm text-[10px] leading-tight`}
    >
      <div className="font-semibold text-gray-800">{layer.name}</div>
      <div className="text-gray-600 text-[8px] font-mono break-words">{layer.dims}</div>
      {layer.activation && (
        <div className="text-blue-700 text-[8px]">{layer.activation}</div>
      )}
      {layer.note && (
        <div className="text-gray-500 text-[8px] italic">{layer.note}</div>
      )}
    </div>
  );
}

/* ── Arrow between blocks ────────────────────────────────────── */
export function Arrow({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center py-0.5 text-gray-400">
      <svg width="12" height="18" viewBox="0 0 12 18">
        <line
          x1="6" y1="0" x2="6" y2="13"
          stroke="currentColor" strokeWidth="1.5"
        />
        <polygon
          points="2.5,11 6,17 9.5,11"
          fill="currentColor"
        />
      </svg>
      {label && <span className="text-[8px] text-gray-500">{label}</span>}
    </div>
  );
}

/* ── Latent space block ──────────────────────────────────────── */
export function LatentBlock({
  name,
  dims,
  type,
  color,
}: {
  name: string;
  dims: string;
  type: string;
  color: string;
}) {
  return (
    <div className={`${color} border-2 border-indigo-300 rounded-lg px-2 py-1.5 shadow-sm text-center`}>
      <div className="font-bold text-indigo-800 text-[10px]">{name}</div>
      <div className="text-[8px] text-gray-600 font-mono">{dims}</div>
      <span className="inline-block mt-0.5 text-[7px] bg-white/70 rounded px-1 py-0.5 text-indigo-700">
        {type === "simplex" ? "Simplex" : "ℝ^d"}
      </span>
    </div>
  );
}

/* ── Prior block ─────────────────────────────────────────────── */
export function PriorBlock({
  name,
  description,
  color,
}: {
  name: string;
  description: string;
  color: string;
}) {
  return (
    <div className={`${color} border border-gray-200 rounded px-2 py-1.5 text-center`}>
      <div className="font-semibold text-violet-800 text-[10px]">{name}</div>
      <div className="text-[8px] text-gray-600 leading-tight mt-0.5">{description}</div>
    </div>
  );
}

/* ── Loss tag ────────────────────────────────────────────────── */
export function LossTag({ loss }: { loss: LossDef }) {
  return (
    <div
      className={`${loss.color} border border-gray-200/80 rounded px-1.5 py-0.5
                   text-[9px] inline-flex flex-col items-center`}
    >
      <span className="font-semibold text-gray-700">{loss.name}</span>
      {loss.formula && (
        <span className="text-[7px] text-gray-500 font-mono">{loss.formula}</span>
      )}
    </div>
  );
}

/* ── Section header ──────────────────────────────────────────── */
export function SectionLabel({ text, color }: { text: string; color: string }) {
  return (
    <div className={`text-[8px] font-bold uppercase tracking-wider ${color} mb-0.5`}>
      {text}
    </div>
  );
}
