"use client";
import React, { type ReactNode } from "react";

/* ══════════════════════════════════════════════════════════════
   UnifiedArchDiagram v3 — compact, Phase-1 only
   architecture diagram for DPMM / Topic series.

   Verified against source code in models/*.py (Jun 2025).

   Layout:
     Row 1 — Main pipeline (Input → Encoder → Latent → Prior → Decoder → Output)
     Row 2 — [Contrastive MoCo] | [Loss Functions] | [Parameters]
   Phase 2 (ODE) removed — no corresponding experiments conducted.
   ══════════════════════════════════════════════════════════════ */

const FONT = "'Arial','Helvetica Neue','Liberation Sans','DejaVu Sans',sans-serif";

/* ── Typography helpers ─────────────────────────────────── */
function Sup({ children }: { children: ReactNode }) {
  return <sup style={{ fontSize: "0.68em", lineHeight: 0, verticalAlign: "super" }}>{children}</sup>;
}
function Sub({ children }: { children: ReactNode }) {
  return <sub style={{ fontSize: "0.68em", lineHeight: 0, verticalAlign: "sub" }}>{children}</sub>;
}
function M({ children }: { children: ReactNode }) {
  // Italic using system sans-serif to avoid unusual/unavailable fonts
  return <span style={{ fontStyle: "italic" }}>{children}</span>;
}

/* ── SVG Arrow ────────────────────────────────────────── */
function HArr({ label, width = 28 }: { label?: string; width?: number }) {
  return (
    <div className="flex flex-col items-center justify-center shrink-0 px-0">
      <svg width={width} height="14" viewBox={`0 0 ${width} 14`}>
        <line x1="0" y1="6" x2={width - 12} y2="6" stroke="#94A3B8" strokeWidth="1.6" />
        <polygon points={`${width - 10},2 ${width},6.5 ${width - 10},11`} fill="#94A3B8" />
      </svg>
      {label && <span style={{ fontSize: 11, color: "#94A3B8", marginTop: -1 }}>{label}</span>}
    </div>
  );
}

/* ── Badge ──────────────────────────────────────────── */
function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span className={`inline-block font-bold px-2 py-0.5 rounded-sm leading-none ${color}`}
      style={{ fontSize: 10.5, color: '#1f2937' }}>
      {text}
    </span>
  );
}

/* ── Section header ─────────────────────────────────── */
function SHead({ text, color, icon }: { text: string; color: string; icon?: ReactNode }) {
  void color; // colors reserved for backgrounds/borders only
  return (
    <div className="font-bold uppercase tracking-wider text-center text-gray-800 flex items-center justify-center gap-1"
      style={{ fontSize: 11.5, marginBottom: 1, letterSpacing: "0.06em" }}>
      {icon}
      {text}
    </div>
  );
}

/* ── Mini inline icons for section headers (14×14) ────── */
const SIcons = {
  input: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <circle cx="7" cy="8" r="3" fill="#6A9FD8" opacity="0.6" stroke="#4A7CBF" strokeWidth="0.7"/>
      <circle cx="13" cy="8" r="3" fill="#6A9FD8" opacity="0.6" stroke="#4A7CBF" strokeWidth="0.7"/>
      <circle cx="10" cy="14" r="3" fill="#6A9FD8" opacity="0.6" stroke="#4A7CBF" strokeWidth="0.7"/>
    </svg>
  ),
  encoder: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <circle cx="4" cy="6" r="1.8" fill="#457B9D" stroke="#1D3557" strokeWidth="0.5"/>
      <circle cx="4" cy="10" r="1.8" fill="#457B9D" stroke="#1D3557" strokeWidth="0.5"/>
      <circle cx="4" cy="14" r="1.8" fill="#457B9D" stroke="#1D3557" strokeWidth="0.5"/>
      <circle cx="11" cy="8" r="1.8" fill="#F1FAEE" stroke="#457B9D" strokeWidth="0.5"/>
      <circle cx="11" cy="12" r="1.8" fill="#F1FAEE" stroke="#457B9D" strokeWidth="0.5"/>
      <circle cx="17" cy="10" r="1.8" fill="#E63946" stroke="#A8DADC" strokeWidth="0.5"/>
      <line x1="6" y1="6" x2="9" y2="8" stroke="#457B9D" strokeWidth="0.4" opacity="0.5"/>
      <line x1="6" y1="10" x2="9" y2="8" stroke="#457B9D" strokeWidth="0.4" opacity="0.5"/>
      <line x1="6" y1="10" x2="9" y2="12" stroke="#457B9D" strokeWidth="0.4" opacity="0.5"/>
      <line x1="6" y1="14" x2="9" y2="12" stroke="#457B9D" strokeWidth="0.4" opacity="0.5"/>
      <line x1="13" y1="8" x2="15" y2="10" stroke="#E63946" strokeWidth="0.5" opacity="0.6"/>
      <line x1="13" y1="12" x2="15" y2="10" stroke="#E63946" strokeWidth="0.5" opacity="0.6"/>
    </svg>
  ),
  latent: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <rect x="2" y="4" width="6" height="12" rx="0.8" fill="none" stroke="#457B9D" strokeWidth="0.7"/>
      <line x1="5" y1="7" x2="5" y2="13" stroke="#457B9D" strokeWidth="0.5" strokeDasharray="0.8,0.8"/>
      <line x1="11" y1="10" x2="16" y2="6" stroke="#E63946" strokeWidth="0.6"/>
      <line x1="11" y1="10" x2="16" y2="14" stroke="#2A9D8F" strokeWidth="0.6"/>
      <circle cx="16" cy="6" r="1.5" fill="#E63946" opacity="0.6"/>
      <circle cx="16" cy="14" r="1.5" fill="#2A9D8F" opacity="0.6"/>
      <circle cx="11" cy="10" r="1.8" fill="#457B9D" opacity="0.5" stroke="#1D3557" strokeWidth="0.5"/>
    </svg>
  ),
  prior: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <ellipse cx="10" cy="10" rx="8" ry="5" fill="none" stroke="#7B68AE" strokeWidth="0.8" strokeDasharray="1.5,1"/>
      <circle cx="6" cy="10" r="2" fill="#E76F51" opacity="0.5"/>
      <circle cx="10" cy="9" r="2.5" fill="#457B9D" opacity="0.5"/>
      <circle cx="14" cy="10" r="1.8" fill="#2A9D8F" opacity="0.5"/>
    </svg>
  ),
  decoder: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <rect x="3" y="4" width="5" height="12" rx="1" fill="#E76F51" opacity="0.3" stroke="#D62828" strokeWidth="0.7"/>
      <rect x="9" y="6" width="4" height="8" rx="0.8" fill="#457B9D" opacity="0.3" stroke="#1D3557" strokeWidth="0.7"/>
      <rect x="14" y="8" width="3" height="4" rx="0.6" fill="#2A9D8F" opacity="0.3" stroke="#1D7874" strokeWidth="0.7"/>
      <line x1="5.5" y1="7" x2="5.5" y2="10" stroke="#D62828" strokeWidth="0.5"/>
      <line x1="11" y1="8.5" x2="11" y2="11.5" stroke="#1D3557" strokeWidth="0.5"/>
    </svg>
  ),
  output: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <rect x="3" y="5" width="14" height="10" rx="1.5" fill="none" stroke="#457B9D" strokeWidth="0.8"/>
      <line x1="6" y1="8" x2="14" y2="8" stroke="#2A9D8F" strokeWidth="0.6"/>
      <line x1="6" y1="11" x2="12" y2="11" stroke="#2A9D8F" strokeWidth="0.6"/>
      <circle cx="15" cy="12" r="1.2" fill="#E63946" opacity="0.7"/>
    </svg>
  ),
  loss: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <polyline points="3,15 7,8 11,12 15,4 18,6" fill="none" stroke="#E63946" strokeWidth="1" opacity="0.7"/>
      <circle cx="15" cy="4" r="1.5" fill="#E63946" opacity="0.5"/>
    </svg>
  ),
  params: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <line x1="3" y1="7" x2="17" y2="7" stroke="#457B9D" strokeWidth="0.8"/>
      <circle cx="12" cy="7" r="2" fill="#E76F51" stroke="#D62828" strokeWidth="0.6"/>
      <line x1="3" y1="13" x2="17" y2="13" stroke="#457B9D" strokeWidth="0.8"/>
      <circle cx="7" cy="13" r="2" fill="#2A9D8F" stroke="#1D7874" strokeWidth="0.6"/>
    </svg>
  ),
  moco: (
    <svg width="14" height="14" viewBox="0 0 20 20" style={{ display: "inline-block", verticalAlign: "middle" }}>
      <circle cx="6" cy="10" r="4" fill="none" stroke="#7B68AE" strokeWidth="0.8"/>
      <circle cx="14" cy="10" r="4" fill="none" stroke="#457B9D" strokeWidth="0.8"/>
      <line x1="10" y1="7" x2="10" y2="13" stroke="#E63946" strokeWidth="0.6" strokeDasharray="1,1"/>
    </svg>
  ),
};

/* ── Variant sub-block ──────────────────────────────── */
function VarBlock({
  name, tagColor, bgColor, layers, isLast = false,
}: { name: string; tagColor: string; bgColor: string; layers: ReactNode[]; isLast?: boolean }) {
  return (
    <div className={`${bgColor} px-2 py-0.5 ${isLast ? "" : "border-b border-dashed border-gray-300/60 pb-0.5"}`}>
      <Badge text={name} color={tagColor} />
      <div className="mt-0 space-y-0">
        {layers.map((l, i) => (
          <div key={i} className="text-gray-700 leading-snug pl-1" style={{ fontSize: 11 }}>{l}</div>
        ))}
      </div>
    </div>
  );
}

/* ── Loss row ────────────────────────────────────────── */
function LossRow({ name, eq, scope, bg }: { name: ReactNode; eq: ReactNode; scope: string; bg: string }) {
  return (
    <div className={`${bg} border border-gray-200/80 rounded px-2 py-0.5 flex items-baseline gap-1`}>
      <span className="font-semibold text-gray-800 shrink-0" style={{ fontSize: 11 }}>{name}</span>
      <span className="text-gray-400" style={{ fontSize: 10 }}>=</span>
      <span className="text-gray-600 flex-1" style={{ fontSize: 11 }}>{eq}</span>
      <span className="text-gray-400 shrink-0" style={{ fontSize: 10 }}>[{scope}]</span>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════
   Main Component
   ══════════════════════════════════════════════════════════════ */
interface Props {
  series: "dpmm" | "topic";
}

export default function UnifiedArchDiagram({ series }: Props) {
  const d = series === "dpmm";

  /* ── Palette ─────────────────────────────── */
  const enc = d
    ? { tag1: "bg-blue-200 text-gray-900", bg1: "bg-blue-50/70",
        tag2: "bg-indigo-200 text-gray-900", bg2: "bg-indigo-50/50",
        border: "border-blue-200" }
    : { tag1: "bg-cyan-200 text-gray-900", bg1: "bg-cyan-50/70",
        tag2: "bg-teal-200 text-gray-900", bg2: "bg-teal-50/50",
        border: "border-cyan-200" };

  return (
    <div style={{ fontFamily: FONT }} className="w-full">
      {/* ═══ Title ═══ */}
      <h2 style={{ fontSize: 14, letterSpacing: "-0.01em" }} className="font-bold text-gray-800 text-center mb-0">
        {d ? "DPMM Series" : "Topic Series"} — Unified Architecture
      </h2>
      <p style={{ fontSize: 10.5 }} className="text-gray-500 text-center mb-1">
        {d
          ? "Deterministic AE backbone · DPMM (Bayesian GMM) clustering prior · Mish activation"
          : "VAE backbone · Dirichlet → Logistic-Normal prior · Probability simplex latent"}
        {" · Variants: "}
        <b>Base</b>, <b>Transformer</b>, <b>Contrastive</b>
      </p>

      {/* ═══ Row 1a: Input → Encoder → Latent Space ═══ */}
      <div className="flex items-start gap-0.5 justify-center">
        {/* ──── Input ──── */}
        <div className="bg-slate-50 border border-slate-300 rounded px-2 py-1 text-center shrink-0" style={{ width: 90 }}>
          <div className="font-bold text-gray-700 flex items-center justify-center gap-0.5" style={{ fontSize: 13 }}>{SIcons.input}Input</div>
          <div className="text-gray-800 leading-tight mt-0" style={{ fontSize: 12 }}>
            <M>x</M> ∈ ℝ<Sup><M>n</M></Sup>
          </div>
          <div className="text-gray-500 leading-tight mt-0" style={{ fontSize: 11 }}>
            <M>n</M> = 3 000 HVGs
          </div>
          <div className={`mt-0 text-[10px] font-medium rounded px-1 py-0 inline-block ${d ? "bg-blue-50 text-gray-800" : "bg-cyan-50 text-gray-800"}`}>
            {d ? "log₁p normalised" : "raw counts → log₁p"}
          </div>
        </div>

        <HArr />

        {/* ──── Encoder ──── */}
        <div className="flex-[2.6] min-w-0">
          <SHead text="Encoder" color={d ? "text-blue-700" : "text-cyan-700"} icon={SIcons.encoder} />
          <div className={`border ${enc.border} rounded-lg overflow-hidden`}>
            <VarBlock
              name="Base / Contrastive"
              tagColor={enc.tag1}
              bgColor={enc.bg1}
              layers={d
                ? [
                    <>Linear(<M>n</M>→256), BN→Mish→Drop(0.15)</>,
                    <>Linear(256→128), BN→Mish→Drop(0.15)</>,
                    <>Linear(128→<M>d</M>) <span className="text-gray-400 text-[10px]">— det. AE head</span></>,
                  ]
                : [
                    <>Linear(<M>n</M>→128), BN→ReLU→Drop(0.0)</>,
                    <>Linear(128→128), BN→ReLU→Drop(0.0)</>,
                    <><M>μ</M>-head: Linear(128→<M>K</M>) <span className="text-gray-400 text-[10px]">— LN mean</span></>,
                    <><M>σ</M><Sup>2</Sup>-head: Linear(128→<M>K</M>), Softplus+10<Sup>−4</Sup></>,
                  ]
              }
            />
            <VarBlock
              name="Transformer"
              tagColor={enc.tag2}
              bgColor={enc.bg2}
              isLast
              layers={d
                ? [
                    <>8×Linear(<M>n</M>→<M>d</M><Sub>m</Sub>)→LN→GELU→Drop(0.1)</>,
                    <>+Learnable token emb ∈ ℝ<Sup>8×128</Sup></>,
                    <>TransEnc(2L,4H,<M>d</M><Sub>m</Sub>=128,ff=256,GELU)</>,
                    <>CrossAttn: MHA(<M>q</M><Sub>l</Sub>,<M>K</M>,<M>V</M>)→LN</>,
                    <>Linear(128→<M>d</M>) <span className="text-gray-400 text-[10px]">— AE head</span></>,
                  ]
                : [
                    <>8×Linear(<M>n</M>→<M>d</M><Sub>m</Sub>)→LN→GELU→Drop(0.1)</>,
                    <>+Learnable token emb ∈ ℝ<Sup>8×128</Sup></>,
                    <>TransEnc(2L,4H,<M>d</M><Sub>m</Sub>=128,ff=256,GELU)</>,
                    <>CrossAttn: MHA(<M>q</M><Sub>l</Sub>,<M>K</M>,<M>V</M>)→LN</>,
                    <><M>μ</M>-head: Linear(128→<M>K</M>)</>,
                    <><M>σ</M><Sup>2</Sup>-head: Linear(128→<M>K</M>), Softplus+10<Sup>−4</Sup></>,
                  ]
              }
            />
          </div>
        </div>

        <HArr label="encode" />

        {/* ──── Latent Space ──── */}
        <div className="shrink-0" style={{ width: 130 }}>
          <SHead text="Latent Space" color="text-indigo-700" icon={SIcons.latent} />
          <div className={`border-2 rounded-lg px-2 py-1 shadow-sm text-center ${d ? "border-blue-300 bg-blue-50/80" : "border-cyan-300 bg-cyan-50/80"}`}>
            <div className="font-bold leading-tight" style={{ fontSize: 13, color: "#1f2937" }}>
              {d ? <><M>z</M> ∈ ℝ<Sup><M>d</M></Sup></> : <><M>θ</M> ∈ Δ<Sup><M>K</M>−1</Sup></>}
            </div>
            {!d && (
              <div className="text-gray-600 mt-0" style={{ fontSize: 11 }}>
                <M>θ</M> = softmax(log <M>θ</M>)
              </div>
            )}
            <div className="text-gray-600 mt-0" style={{ fontSize: 11 }}>
              {d
                ? <><M>z</M> = Enc(<M>x</M>)  <span className="text-gray-400">(det.)</span></>
                : <>log <M>θ</M> = <M>μ</M> + <M>σ</M> ⊙ <M>ε</M>,  <M>ε</M> ~ 𝒩(0, <M>I</M>)</>}
            </div>
            <div className="text-gray-500 mt-0.5" style={{ fontSize: 11 }}>
              {d ? <><M>d</M> = 10</> : <><M>K</M> = 10 topics</>}
            </div>
            <span className={`inline-block mt-0.5 rounded px-2 py-0.5 font-semibold ${d ? "bg-blue-100 text-gray-800" : "bg-cyan-100 text-gray-800"}`}
              style={{ fontSize: 10.5 }}>
              {d ? "Continuous ℝ^d" : "Probability Simplex"}
            </span>
          </div>
        </div>
      </div>

      {/* ── Connecting arrow between pipeline rows ── */}
      <div className="flex justify-center" style={{ margin: "1px 0" }}>
        <svg width="14" height="14" viewBox="0 0 16 16">
          <line x1="7.5" y1="0" x2="7.5" y2="10" stroke="#94A3B8" strokeWidth="1.6" />
          <polygon points="2.5,10 7.5,16 12.5,10" fill="#94A3B8" />
        </svg>
      </div>

      {/* ═══ Row 1b: Prior → Decoder → Output ═══ */}
      <div className="flex items-start gap-0.5 justify-center">

        {/* ──── Prior ──── */}
        <div className="shrink-0" style={{ width: 145 }}>
          <SHead text="Prior" color={d ? "text-violet-700" : "text-teal-700"} icon={SIcons.prior} />
          <div className={`border rounded px-2 py-0.5 text-center ${d ? "border-violet-300 bg-violet-50/80" : "border-teal-300 bg-teal-50/80"}`}>
            <div className="font-semibold leading-tight" style={{ fontSize: 12, color: "#1f2937" }}>
              {d ? "DPMM (BayesianGMM)" : "Dirichlet → Logistic-Normal"}
            </div>
            <div className="text-gray-600 leading-snug mt-0.5 text-center" style={{ fontSize: 11 }}>
              {d ? (
                <>
                  <div><M>K</M> = 50 components, diag. Σ</div>
                  <div><M>p</M>(<M>z</M>) = Σ<Sub><M>k</M></Sub> <M>π</M><Sub><M>k</M></Sub> 𝒩(<M>z</M>|<M>μ</M><Sub><M>k</M></Sub>, Σ<Sub><M>k</M></Sub>)</div>
                  <div className="text-gray-400 text-[10px]">weight_conc_prior=1.0</div>
                  <div className="text-gray-400 text-[10px]">mean_prec_prior=0.1</div>
                  <div className="text-gray-400 text-[10px]">warmup ratio=0.6</div>
                  <div className="text-gray-400 text-[10px]">refit every 10 ep post-warmup</div>
                </>
              ) : (
                <>
                  <div>Dir(<M>α</M>) ≈ Logistic-Normal</div>
                  <div><M>α</M><Sub><M>k</M></Sub> = 1/<M>K</M> · <M>s</M>,  <M>s</M> = 10</div>
                  <div><M>μ</M><Sub>p</Sub> = log <M>α</M> − mean(log <M>α</M>)</div>
                  <div><M>σ</M><Sup>2</Sup><Sub>p</Sub> = (1−2/<M>K</M>)/<M>α</M> + Σ(1/<M>α</M>)/<M>K</M><Sup>2</Sup></div>
                  <div className="text-gray-400 text-[10px]">free bits = 0.1 nats/dim</div>
                  <div className="text-gray-400 text-[10px]">penalty weight = 50</div>
                </>
              )}
            </div>
          </div>
        </div>

        <HArr label="decode" />

        {/* ──── Decoder ──── */}
        <div className="flex-[1.8] min-w-0">
          <SHead text="Decoder" color="text-sky-700" icon={SIcons.decoder} />
          {d ? (
            <div className="border border-sky-200 rounded-lg overflow-hidden">
              <VarBlock name="Base / Contrastive" tagColor="bg-sky-200 text-gray-900" bgColor="bg-sky-50/70"
                layers={[
                  <>Linear(<M>d</M> → 128), BN → Mish → Drop(0.15)</>,
                  <>Linear(128 → 256), BN → Mish → Drop(0.15)</>,
                  <>Linear(256 → <M>n</M>) <span className="text-gray-400 text-[10px]">— no output act.</span></>,
                ]}
              />
              <VarBlock name="Transformer" tagColor="bg-indigo-200 text-gray-900" bgColor="bg-indigo-50/50" isLast
                layers={[<>MLP(<M>d</M>→128→256→<M>n</M>), LN→GELU→Drop(0.1)</>]}
              />
            </div>
          ) : (
            <div className="border border-teal-200 rounded-lg bg-teal-50/40 px-2 py-1">
              <Badge text="All Variants" color="bg-teal-200 text-gray-900" />
              <div className="mt-0.5 space-y-0 text-gray-700 pl-1" style={{ fontSize: 11 }}>
                <div><M>β</M> ∈ ℝ<Sup><M>K</M>×<M>n</M></Sup> <span className="text-gray-400 text-[10px]">(learnable topic-word logits)</span></div>
                <div><M>β̃</M><Sub><M>k</M></Sub> = softmax(<M>β</M><Sub><M>k</M></Sub>) <span className="text-gray-400 text-[10px]">— per-topic word dist.</span></div>
                <div>log <M>p</M>(<M>w</M>|<M>θ</M>) = LSE<Sub><M>k</M></Sub>[log <M>θ</M><Sub><M>k</M></Sub> + log <M>β̃</M><Sub><M>kv</M></Sub>]</div>
                <div><M>x̂</M> = exp(log <M>p</M>)</div>
              </div>
            </div>
          )}
        </div>

        <HArr />

        {/* ──── Output ──── */}
        <div className="bg-slate-50 border border-slate-300 rounded px-2 py-1 text-center shrink-0" style={{ width: 90 }}>
          <div className="font-bold text-gray-700 flex items-center justify-center gap-0.5" style={{ fontSize: 13 }}>{SIcons.output}Output</div>
          <div className="text-gray-800 leading-tight mt-0" style={{ fontSize: 12 }}>
            <M>x̂</M> ∈ ℝ<Sup><M>n</M></Sup>
          </div>
          <div className="text-gray-500 mt-0" style={{ fontSize: 11 }}>
            {d ? "MSE reconstruction" : "topic-mixture"}
          </div>
        </div>
      </div>

      {/* ═══ Row 2: Contrastive | Losses | ODE + Params ═══ */}
      <div className="mt-1 space-y-0.5">

        {/* ── Contrastive MoCo ── */}
        <div className={`border border-dashed rounded-lg p-0.5 ${d ? "border-violet-300 bg-violet-50/30" : "border-purple-300 bg-purple-50/30"}`}>
          <div className="flex items-center gap-1 mb-0.5">
            <Badge text="Contrastive Only" color={d ? "bg-violet-200 text-gray-900" : "bg-purple-200 text-gray-900"} />
            {SIcons.moco}
            <span className="font-bold uppercase tracking-wide text-gray-700" style={{ fontSize: 11 }}>
              MoCo Pipeline
            </span>
          </div>
          {/* Augmentation */}
          <div className="text-gray-600 mb-0.5" style={{ fontSize: 11 }}>
            <span className="font-semibold text-gray-700">Augmentation:</span>
          </div>
          <div className="space-y-0 mb-0.5 pl-1" style={{ fontSize: 10 }}>
            {(d
              ? [
                  "FeatDrop(p=0.2) — Bernoulli mask",
                  "GaussNoise(σ=0.1, prob=0.2)",
                  "FeatMask(p=0.1)",
                  "RandScale ~ U(0.8, 1.2)",
                ]
              : [
                  "FeatDrop(p=0.2) — Bernoulli mask",
                  "GaussNoise(σ=0.1, prob=0.2)",
                  "FeatMask(p=0.1)",
                  "ExtraDrop(p=0.05)",
                ]
            ).map((t, i) => (
              <div key={i} className="text-gray-500">{t}</div>
            ))}
          </div>
          {/* Projectors */}
          <div className="space-y-px" style={{ fontSize: 11 }}>
            {(d
              ? [
                  ["Q proj", <>MLP(<M>d</M>→<M>d</M>→128), BN→ReLU</>],
                  ["K proj", <>EMA copy, <M>m</M>=0.999</>],
                  ["Queue", <>4096 × 128, <M>τ</M>=0.2</>],
                  ["Proto", "10 × 128 (learnable, Xavier)"],
                ]
              : [
                  ["Q proj", <>MLP(<M>K</M>→<M>K</M>→64), BN→ReLU</>],
                  ["K proj", <>EMA copy, <M>m</M>=0.999</>],
                  ["Queue", <>4096 × 64, <M>τ</M>=0.2</>],
                  ["Proto", "10 × 64 (learnable, Xavier)"],
                ]
            ).map(([lb, desc], i) => (
              <div key={i} className="flex gap-1 justify-center">
                <span className="font-semibold shrink-0 text-gray-700" style={{ width: 52, textAlign: "left" }}>{lb}:</span>
                <span className="text-gray-600 text-left" style={{ minWidth: 170 }}>{desc}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Loss Functions + Parameters side by side ── */}
        <div className="grid grid-cols-2 gap-0.5">
          {/* Loss Functions */}
          <div className="border border-gray-200 rounded-lg p-0.5 bg-gray-50/50">
            <SHead text="Loss Functions" color="text-slate-600" icon={SIcons.loss} />
            <div className="space-y-0.5 mt-0.5">
              {d ? (
                <>
                  <LossRow
                    name={<>ℒ<Sub>recon</Sub></>}
                    eq={<>MSE(<M>x</M>, <M>x̂</M>) = ‖<M>x</M>−<M>x̂</M>‖<Sup>2</Sup><Sub>2</Sub>/<M>n</M></>}
                    scope="All" bg="bg-slate-100"
                  />
                  <LossRow
                    name={<>ℒ<Sub>DPMM</Sub></>}
                    eq={<>−(1/<M>N</M>)Σ<Sub><M>i</M></Sub> log Σ<Sub><M>k</M></Sub> <M>π</M><Sub><M>k</M></Sub>𝒩(<M>z</M><Sub><M>i</M></Sub>|<M>μ</M><Sub><M>k</M></Sub>,Σ<Sub><M>k</M></Sub>)</>}
                    scope="All" bg="bg-violet-50"
                  />
                  <LossRow
                    name={<>ℒ<Sub>CL</Sub></>}
                    eq={<>InfoNCE(<M>q</M>·<M>k</M><Sup>+</Sup>/<M>τ</M>) + 0.5·Sym + 0.3·Proto</>}
                    scope="Contr." bg="bg-indigo-50"
                  />
                  <div className="text-gray-500 pl-1 mt-0.5" style={{ fontSize: 10 }}>
                    ℒ = ℒ<Sub>recon</Sub> + <M>λ</M><Sub>dpmm</Sub>·ℒ<Sub>DPMM</Sub> + <M>λ</M><Sub>moco</Sub>·ℒ<Sub>CL</Sub>
                    <span className="text-gray-400 ml-1">(<M>λ</M><Sub>dpmm</Sub>=1.0, <M>λ</M><Sub>moco</Sub>=0.5)</span>
                  </div>
                </>
              ) : (
                <>
                  <LossRow
                    name={<>ℒ<Sub>recon</Sub></>}
                    eq={<>−(1/<M>N</M>)Σ<Sub><M>i,v</M></Sub> <M>x̄</M><Sub><M>iv</M></Sub> log(<M>x̂</M><Sub><M>iv</M></Sub>+<M>ε</M>)</>}
                    scope="All" bg="bg-slate-100"
                  />
                  <LossRow
                    name={<>ℒ<Sub>KL</Sub></>}
                    eq={<><M>D</M><Sub>KL</Sub>(<M>q</M>(<M>θ</M>|<M>x</M>) ‖ LN(<M>μ</M><Sub>p</Sub>,<M>σ</M><Sub>p</Sub>)) + 50·penalty</>}
                    scope="All" bg="bg-teal-50"
                  />
                  <LossRow
                    name={<>ℒ<Sub>CL</Sub></>}
                    eq={<>0.1·InfoNCE + 0.05·MSE-sym + 0.05·Proto</>}
                    scope="Contr." bg="bg-indigo-50"
                  />
                  <LossRow
                    name={<>ℒ<Sub>reg</Sub></>}
                    eq={<>0.01·<M>H</M><Sub>sparsity</Sub> + 0.01·<M>H</M><Sub>diversity</Sub></>}
                    scope="Contr." bg="bg-purple-50"
                  />
                  <div className="text-gray-500 pl-1 mt-0.5" style={{ fontSize: 10 }}>
                    ℒ = ℒ<Sub>recon</Sub> + <M>β</M>·ℒ<Sub>KL</Sub> + ℒ<Sub>CL</Sub> + ℒ<Sub>reg</Sub>
                    <span className="text-gray-400 ml-1">(<M>β</M>=0.01 Topic, 1.0 Pure-VAE)</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Parameters */}
          <div className="border border-gray-200 rounded-lg p-0.5 bg-gray-50/50">
            <SHead text="Parameters" color="text-slate-600" icon={SIcons.params} />
            <div className="space-y-0.5 mt-0.5">
              {(d
                ? [
                    { v: "Base", c: "~1.62M", ep: "400ep" },
                    { v: "Transformer", c: "~4.21M", ep: "400ep" },
                    { v: "Contrastive", c: "~1.62M", ep: "400ep" },
                  ]
                : [
                    { v: "Base", c: "~433K", ep: "1000ep" },
                    { v: "Transformer", c: "~3.44M", ep: "1000ep" },
                    { v: "Contrastive", c: "~436K", ep: "1000ep" },
                  ]
              ).map((p, i) => (
                <div key={i} className="flex items-center gap-1 text-gray-600 justify-center" style={{ fontSize: 11 }}>
                  <span className="font-semibold text-gray-700" style={{ width: 72 }}>{p.v}</span>
                  <span className="font-mono text-gray-600">{p.c}</span>
                  <span className="text-gray-400" style={{ width: 48, textAlign: "right" }}>{p.ep}</span>
                </div>
              ))}
            </div>
            <div className="mt-0.5 text-gray-500 leading-snug" style={{ fontSize: 10 }}>
              AdamW · lr = 10<Sup>−3</Sup> · wd = {d ? "0" : "10⁻³"} · batch = 128
              <br/>
              {d ? <>dropout = 0.2 · DPMM warmup = 0.6</> : <>dropout = 0.0 · <M>K</M> = 10 topics</>}
              <br/>
              grad clip = 10.0 · seed = 42 · HVG = 3k
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
