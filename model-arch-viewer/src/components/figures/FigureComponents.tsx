"use client";

import React from "react";

/* ╔══════════════════════════════════════════════════════════════════════════╗
   ║  Shared figure-composition components for Figures 2-6.                 ║
   ║                                                                        ║
   ║  <FigureContainer>  — outer wrapper with id for Playwright screenshot  ║
   ║  <PanelSection>     — titled panel row/grid                            ║
   ║  <PanelLabel>       — bold letter label (A), (B), …                   ║
   ║  <SubplotImage>     — single subplot image with optional caption       ║
   ║  <SubplotGrid>      — CSS-grid of subplot images                       ║
   ╚══════════════════════════════════════════════════════════════════════════╝ */

// ─── FigureContainer ─────────────────────────────────────────────────────────

interface FigureContainerProps {
  figureId: string;       // e.g. "figure2"
  series: string;         // "dpmm" | "topic"
  maxWidthPx?: number;    // default 670 (≈17cm at 96dpi, maximum print width)
  minHeightPx?: number;   // default 0 — height is content-fitted; max is 21cm at DPR=3
  children: React.ReactNode;
}

export function FigureContainer({
  figureId,
  series,
  maxWidthPx = 670,
  minHeightPx = 0,
  children,
}: FigureContainerProps) {
  const rootId = `${figureId}-root`;
  return (
    <div className="bg-white py-1">
      <div
        id={rootId}
        className="mx-auto text-center"
        style={{
          maxWidth: `${maxWidthPx}px`,
          ...(minHeightPx > 0 ? { minHeight: `${minHeightPx}px` } : {}),
        }}
      >
        {children}
      </div>
    </div>
  );
}

// ─── PanelLabel ──────────────────────────────────────────────────────────────

interface PanelLabelProps {
  label: string;      // "A", "B", etc.
  className?: string;
}

export function PanelLabel({ label, className = "" }: PanelLabelProps) {
  return (
    <span
      className={`inline-block font-bold text-[11px] leading-none mr-1 ${className}`}
      style={{ fontFamily: "Arial, sans-serif" }}
    >
      ({label})
    </span>
  );
}

// ─── PanelSection ────────────────────────────────────────────────────────────

interface PanelSectionProps {
  label?: string;      // e.g. "A"
  title?: string;      // optional section title
  children: React.ReactNode;
  className?: string;
}

export function PanelSection({
  label,
  title,
  children,
  className = "",
}: PanelSectionProps) {
  return (
    <div className={`mb-1 ${className}`}>
      <div className="flex items-baseline gap-0.5 mb-0.5 px-0.5">
        {label && <PanelLabel label={label} />}
        {title && (
          <span
            className="text-[9px] text-gray-600 font-medium"
            style={{ fontFamily: "Arial, sans-serif" }}
          >
            {title}
          </span>
        )}
      </div>
      {children}
    </div>
  );
}

// ─── SubplotImage ────────────────────────────────────────────────────────────

interface SubplotImageProps {
  src: string;          // image URL path
  alt?: string;
  caption?: string;
  className?: string;
  style?: React.CSSProperties;
  naturalWidth?: number;  // unused — kept for API compat
  naturalHeight?: number; // unused — kept for API compat
}

export function SubplotImage({
  src,
  alt = "subplot",
  caption,
  className = "",
  style,
}: SubplotImageProps) {
  // Images are generated with bbox_inches='tight' so all text is included.
  // Each image displays at its natural proportions via w-full h-auto.
  return (
    <div className={`inline-block text-center ${className}`} style={style}>
      <img
        src={src}
        alt={alt}
        className="w-full h-auto"
        style={{ imageRendering: "auto" }}
        loading="eager"
      />
      {caption && (
        <div
          className="text-[7px] text-gray-500 mt-0.5 leading-tight"
          style={{ fontFamily: "Arial, sans-serif" }}
        >
          {caption}
        </div>
      )}
    </div>
  );
}

// ─── SubplotGrid ─────────────────────────────────────────────────────────────

interface SubplotGridProps {
  columns: number;
  gap?: string;          // default "4px"
  children: React.ReactNode;
  className?: string;
}

export function SubplotGrid({
  columns,
  gap = "3px",
  children,
  className = "",
}: SubplotGridProps) {
  return (
    <div
      className={`grid ${className}`}
      style={{
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap,
        alignItems: "start",
      }}
    >
      {children}
    </div>
  );
}

// ─── Helper: build static subplot path ──────────────────────────────────────

export function subplotPath(
  series: string,
  figNum: number,
  filename: string,
): string {
  return `/subplots/${series}/fig${figNum}/${filename}`;
}

// ─── PanelColumns ───────────────────────────────────────────────────────────
// Multi-column layout for placing panels side-by-side (e.g. A|B)

interface PanelColumnsProps {
  columns?: number;     // default 2
  gap?: string;         // default "4px"
  children: React.ReactNode;
  className?: string;
}

export function PanelColumns({
  columns = 2,
  gap = "4px",
  children,
  className = "",
}: PanelColumnsProps) {
  return (
    <div
      className={`grid ${className}`}
      style={{
        gridTemplateColumns: `repeat(${columns}, 1fr)`,
        gap,
        alignItems: "start",
      }}
    >
      {children}
    </div>
  );
}

// ─── Manifest type ──────────────────────────────────────────────────────────

export interface ManifestData {
  [key: string]: any;
}
