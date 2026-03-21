"use client";

import React, { useCallback, useRef } from "react";
import { toPng, toSvg } from "html-to-image";

interface Props {
  targetRef: React.RefObject<HTMLDivElement | null>;
  fileName: string;
}

export default function ExportBar({ targetRef, fileName }: Props) {
  const downloadLink = useRef<HTMLAnchorElement>(null);

  const exportPng = useCallback(async () => {
    if (!targetRef.current) return;
    const url = await toPng(targetRef.current, {
      pixelRatio: 3,
      backgroundColor: "#ffffff",
    });
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName}.png`;
    a.click();
  }, [targetRef, fileName]);

  const exportSvg = useCallback(async () => {
    if (!targetRef.current) return;
    const url = await toSvg(targetRef.current, { backgroundColor: "#ffffff" });
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName}.svg`;
    a.click();
  }, [targetRef, fileName]);

  return (
    <div className="flex gap-2 justify-center mt-4">
      <button
        onClick={exportPng}
        className="px-3 py-1.5 text-xs font-medium rounded-md
                   bg-blue-600 text-white hover:bg-blue-700 transition-colors
                   cursor-pointer shadow-sm"
      >
        Export PNG (3×)
      </button>
      <button
        onClick={exportSvg}
        className="px-3 py-1.5 text-xs font-medium rounded-md
                   bg-green-600 text-white hover:bg-green-700 transition-colors
                   cursor-pointer shadow-sm"
      >
        Export SVG
      </button>
      <a ref={downloadLink} className="hidden" />
    </div>
  );
}
