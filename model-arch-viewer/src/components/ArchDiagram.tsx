"use client";
import React from "react";
import type { ModelArch } from "@/data/models";
import {
  LayerBlock,
  LossTag,
  SectionLabel,
} from "./ArchBlock";

interface Props {
  model: ModelArch;
}

function HArrow({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center text-gray-400 shrink-0 px-0.5">
      <svg width="18" height="10" viewBox="0 0 18 10">
        <line x1="0" y1="5" x2="13" y2="5" stroke="currentColor" strokeWidth="1.5" />
        <polygon points="12,1.5 18,5 12,8.5" fill="currentColor" />
      </svg>
      {label && <span className="text-[7px] text-gray-500 -mt-0.5">{label}</span>}
    </div>
  );
}

export default function ArchDiagram({ model }: Props) {
  const encLayers = model.encoder;
  const decLayers = model.decoder;
  const useDoubleEnc = encLayers.length > 3;
  const useDoubleDec = decLayers.length > 3;

  return (
    <div className="w-full">
      {/* Main pipeline */}
      <div className="flex items-center gap-0.5">
        {/* Input */}
        <div className="bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 text-center shrink-0 w-[70px]">
          <div className="font-bold text-gray-700 text-[9px] leading-none">Input</div>
          <div className="text-[7px] text-gray-500 font-mono leading-tight mt-0.5">
            {model.series === "topic" ? "raw counts" : "log1p norm"}
          </div>
        </div>

        <HArrow />

        {/* Encoder */}
        <div className="flex-1 min-w-0">
          <SectionLabel text="Encoder" color="text-blue-700" />
          {useDoubleEnc ? (
            <div className="grid grid-cols-2 gap-0.5">
              {encLayers.map((l, i) => (
                <LayerBlock key={i} layer={l} idx={i} />
              ))}
            </div>
          ) : (
            <div className="space-y-0.5">
              {encLayers.map((l, i) => (
                <LayerBlock key={i} layer={l} idx={i} />
              ))}
            </div>
          )}
        </div>

        <HArrow label="enc" />

        {/* Latent Space */}
        <div className="shrink-0 w-[100px]">
          <SectionLabel text="Latent" color="text-indigo-700" />
          <div className={`${model.latentSpace.color} border border-indigo-300 rounded px-1 py-0.5 shadow-sm text-center`}>
            <div className="font-bold text-indigo-800 text-[8px] leading-tight">{model.latentSpace.name}</div>
            <div className="text-[7px] text-gray-600 font-mono leading-none mt-0.5">{model.latentSpace.dims}</div>
          </div>
        </div>

        <HArrow />

        {/* Prior */}
        <div className="shrink-0 w-[120px]">
          <SectionLabel text="Prior" color="text-violet-700" />
          <div className={`${model.prior.color} border border-violet-300 rounded px-1 py-0.5 text-center`}>
            <div className="font-semibold text-violet-800 text-[8px] leading-tight">{model.prior.name}</div>
            <div className="text-[6px] text-gray-600 leading-tight mt-0.5 break-words">
              {model.prior.description}
            </div>
          </div>
        </div>

        <HArrow label="dec" />

        {/* Decoder */}
        <div className="flex-1 min-w-0">
          <SectionLabel text="Decoder" color="text-sky-700" />
          {useDoubleDec ? (
            <div className="grid grid-cols-2 gap-0.5">
              {decLayers.map((l, i) => (
                <LayerBlock key={i} layer={l} idx={i} />
              ))}
            </div>
          ) : (
            <div className="space-y-0.5">
              {decLayers.map((l, i) => (
                <LayerBlock key={i} layer={l} idx={i} />
              ))}
            </div>
          )}
        </div>

        <HArrow />

        {/* Output */}
        <div className="bg-slate-50 border border-slate-200 rounded px-1.5 py-0.5 text-center shrink-0 w-[70px]">
          <div className="font-bold text-gray-700 text-[9px] leading-none">Output</div>
          <div className="text-[7px] text-gray-500 font-mono leading-tight mt-0.5">recon x̂</div>
        </div>
      </div>

      {/* Extras & Loss - Compact Row */}
      <div className="mt-1 flex gap-2 items-start border-t border-gray-100 pt-0.5">
         {/* Extras */}
        {model.extras && model.extras.length > 0 && (
          <div className="flex-1">
             <div className="text-[7px] font-bold text-violet-700 mb-0.5">AUXILIARY</div>
             <div className="flex flex-wrap gap-0.5">
                {model.extras.map((l, i) => (
                  <LayerBlock key={i} layer={l} idx={i} />
                ))}
            </div>
          </div>
        )}

        {/* Loss Functions */}
        <div className="flex-1">
           <div className="text-[7px] font-bold text-slate-600 mb-0.5">OBJECTIVES</div>
           <div className="flex flex-wrap gap-0.5">
            {model.losses.map((l, i) => (
              <LossTag key={i} loss={l} />
            ))}
          </div>
        </div>
      </div>

      {/* Footer Notes */}
      <div className="mt-0.5 flex justify-between items-end">
        <div className="text-[7px] text-gray-500 leading-none">
           {model.notes.join(" • ")}
        </div>
        <div className="text-[7px] text-gray-400 font-mono">
          ~{model.paramCount} params
        </div>
      </div>
    </div>
  );
}
