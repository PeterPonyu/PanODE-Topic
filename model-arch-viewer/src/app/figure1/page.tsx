"use client";

import React from "react";
import Link from "next/link";
import ArchDiagram from "@/components/ArchDiagram";
import { DPMM_MODELS, TOPIC_MODELS } from "@/data/models";

export default function FigureOnePage() {
  const models = [...DPMM_MODELS, ...TOPIC_MODELS];

  return (
    <div className="min-h-screen bg-white p-5">
      <div className="max-w-[2200px] mx-auto">
        <h1 className="text-2xl font-bold text-gray-800 text-center">Figure 1 — Model Architectures</h1>
        <p className="text-sm text-gray-500 text-center mt-1 mb-4">
          Left-to-right Phase-1 architecture flow for 3 DPMM and 3 Topic variants
        </p>

        <div className="flex gap-4 justify-center mb-4">
          <Link href="/figure1/dpmm" className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">
            DPMM Series (3×1)
          </Link>
          <Link href="/figure1/topic" className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700">
            Topic Series (3×1)
          </Link>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {models.map((model) => (
            <section key={model.id} className="border border-gray-200 rounded-xl p-3 bg-gray-50/40">
              <div className="text-sm font-semibold text-gray-800 mb-2">{model.displayName}</div>
              <ArchDiagram model={model} />
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
