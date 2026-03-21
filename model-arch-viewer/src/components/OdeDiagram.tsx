"use client";

import React from "react";
import { ODE_MODULE } from "@/data/models";

interface Props {
  series: "dpmm" | "topic";
}

export default function OdeDiagram({ series }: Props) {
  const ode = ODE_MODULE[series];
  return (
    <div className="flex flex-col items-center w-full max-w-md mx-auto mt-6 p-4
                    bg-gradient-to-b from-cyan-50 to-white border border-cyan-200 rounded-xl shadow">
      <div className="text-xs font-bold uppercase tracking-wider text-cyan-700 mb-2">
        Phase 2 — ODE Dynamics (Frozen Encoder/Decoder)
      </div>

      <div className="bg-cyan-100 border border-cyan-300 rounded-lg px-3 py-2 text-center w-full">
        <div className="font-semibold text-cyan-800 text-sm">{ode.name}</div>
        <div className="text-xs text-gray-600 font-mono mt-1">{ode.layers}</div>
      </div>

      <svg width="16" height="24" viewBox="0 0 16 24" className="my-1 text-gray-400">
        <line x1="8" y1="0" x2="8" y2="18" stroke="currentColor" strokeWidth="2" />
        <polygon points="3,16 8,23 13,16" fill="currentColor" />
      </svg>

      <div className="bg-cyan-50 border border-cyan-200 rounded-lg px-3 py-2 text-center w-full">
        <div className="font-semibold text-cyan-700 text-sm">Time Head</div>
        <div className="text-xs text-gray-600 font-mono mt-0.5">{ode.timeHead}</div>
      </div>

      <svg width="16" height="24" viewBox="0 0 16 24" className="my-1 text-gray-400">
        <line x1="8" y1="0" x2="8" y2="18" stroke="currentColor" strokeWidth="2" />
        <polygon points="3,16 8,23 13,16" fill="currentColor" />
      </svg>

      <div className="bg-white border border-cyan-200 rounded-lg px-3 py-2 text-center w-full">
        <div className="font-semibold text-cyan-700 text-sm">ODE Integration</div>
        <div className="text-[10px] text-gray-500 font-mono">{ode.integration}</div>
        <div className="text-[10px] text-gray-500 mt-0.5">
          Solves dz/dt = f(z, t) from t₀ → t₁
        </div>
      </div>
    </div>
  );
}
