"use client";

import React from "react";
import UnifiedArchDiagram from "@/components/UnifiedArchDiagram";

export default function Figure1TopicPage() {
  return (
    <div className="bg-white py-0.5">
      <div id="figure1-root" className="max-w-167.5 mx-auto text-center">
        <UnifiedArchDiagram series="topic" />
      </div>
    </div>
  );
}
