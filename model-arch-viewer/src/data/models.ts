/* ── Model architecture definitions for PanODE-LAB ── */
/* Unified cool-tone palette: blue → indigo → violet → sky → slate */

export interface LayerDef {
  name: string;
  dims: string;       // e.g. "3000→256"
  activation?: string;
  note?: string;
  color: string;       // tailwind bg class
}

export interface LossDef {
  name: string;
  formula?: string;
  color: string;
}

export interface ModelArch {
  id: string;
  displayName: string;
  series: "dpmm" | "topic";
  variant: "base" | "transformer" | "contrastive";
  encoder: LayerDef[];
  latentSpace: {
    name: string;
    dims: string;
    type: string; // "continuous" | "simplex"
    color: string;
  };
  prior: {
    name: string;
    description: string;
    color: string;
  };
  decoder: LayerDef[];
  losses: LossDef[];
  extras?: LayerDef[];  // MoCo, augmentation, etc.
  paramCount: string;
  notes: string[];
}

const INPUT_DIM = "n_genes (3000)";
const LATENT_DPMM = "latent_dim (10)";
const LATENT_TOPIC = "n_topics (10)";

// ═══════════════════════════════════════════════════════════════
// DPMM Series — blue / indigo / violet palette
// ═══════════════════════════════════════════════════════════════

export const DPMM_BASE: ModelArch = {
  id: "dpmm-base",
  displayName: "DPMM-Base",
  series: "dpmm",
  variant: "base",
  encoder: [
    { name: "Linear", dims: `${INPUT_DIM} → 256`, activation: "BN → Mish → Dropout(0.15)", color: "bg-blue-50" },
    { name: "Linear", dims: "256 → 128", activation: "BN → Mish → Dropout(0.15)", color: "bg-blue-100" },
    { name: "Linear", dims: `128 → ${LATENT_DPMM}`, note: "AE projection head", color: "bg-blue-200" },
  ],
  latentSpace: {
    name: "z ∈ ℝ^d",
    dims: LATENT_DPMM,
    type: "continuous",
    color: "bg-indigo-100",
  },
  prior: {
    name: "DPMM (BayesianGMM)",
    description: "n_components=50, diag cov, periodic refit after warmup (ratio=0.6)",
    color: "bg-violet-100",
  },
  decoder: [
    { name: "Linear", dims: `${LATENT_DPMM} → 128`, activation: "BN → Mish → Dropout(0.15)", color: "bg-sky-50" },
    { name: "Linear", dims: "128 → 256", activation: "BN → Mish → Dropout(0.15)", color: "bg-sky-100" },
    { name: "Linear", dims: `256 → ${INPUT_DIM}`, note: "No output activation", color: "bg-sky-200" },
  ],
  losses: [
    { name: "Reconstruction", formula: "MSE(x, x̂) = ‖x − x̂‖²₂ / n", color: "bg-slate-100" },
    { name: "DPMM", formula: "−(1/N) Σᵢ log Σₖ πₖ 𝒩(zᵢ|μₖ,Σₖ)", color: "bg-violet-50" },
  ],
  paramCount: "1.62M",
  notes: [
    "Deterministic AE + DPMM (BayesianGMM) prior; Mish activation",
    "DPMM refit every 10 ep after warmup_ratio(0.6) × total_epochs",
    "weight_conc_prior=1.0, mean_prec_prior=0.1, K=50 components, diag Σ",
  ],
};

export const DPMM_TRANSFORMER: ModelArch = {
  id: "dpmm-transformer",
  displayName: "DPMM-Transformer",
  series: "dpmm",
  variant: "transformer",
  encoder: [
    { name: "8× Projection Heads", dims: `${INPUT_DIM} → d_model(128)`, activation: "LN → GELU → Dropout", color: "bg-blue-50" },
    { name: "Token Embeddings", dims: "1 × 8 × 128", note: "Learnable positions", color: "bg-blue-50" },
    { name: "TransformerEncoder", dims: "2L, 4H, d=128, ff=256", activation: "GELU", color: "bg-blue-100" },
    { name: "Cross-Attention", dims: "query → MHA(Q,K,V)", note: "Aggregation", color: "bg-blue-200" },
    { name: "LN → Linear", dims: `128 → ${LATENT_DPMM}`, note: "AE head", color: "bg-blue-200" },
  ],
  latentSpace: {
    name: "z ∈ ℝ^d",
    dims: LATENT_DPMM,
    type: "continuous",
    color: "bg-indigo-100",
  },
  prior: {
    name: "DPMM (BayesianGMM)",
    description: "Same as DPMM-Base with warmup_ratio=0.6",
    color: "bg-violet-100",
  },
  decoder: [
    { name: "MLP", dims: `${LATENT_DPMM} → 128 → 256 → ${INPUT_DIM}`, activation: "LN → GELU → Dropout(0.1)", color: "bg-sky-100" },
  ],
  losses: [
    { name: "Reconstruction", formula: "MSE(x, x̂) = ‖x − x̂‖²₂ / n", color: "bg-slate-100" },
    { name: "DPMM", formula: "−(1/N) Σᵢ log Σₖ πₖ 𝒩(zᵢ|μₖ,Σₖ)", color: "bg-violet-50" },
  ],
  paramCount: "4.21M",
  notes: [
    "8 multi-head projections + learnable token-type embeddings ∈ ℝ^{8×128}",
    "TransformerEnc(2L, 4H, d_model=128, ff=256, GELU) → Cross-Attn aggregation",
  ],
};

export const DPMM_CONTRASTIVE: ModelArch = {
  id: "dpmm-contrastive",
  displayName: "DPMM-Contrastive",
  series: "dpmm",
  variant: "contrastive",
  encoder: [
    { name: "Linear", dims: `${INPUT_DIM} → 256`, activation: "BN → Mish → Dropout(0.15)", color: "bg-blue-50" },
    { name: "Linear", dims: "256 → 128", activation: "BN → Mish → Dropout(0.15)", color: "bg-blue-100" },
    { name: "Linear", dims: `128 → ${LATENT_DPMM}`, note: "AE projection head", color: "bg-blue-200" },
  ],
  latentSpace: {
    name: "z ∈ ℝ^d",
    dims: LATENT_DPMM,
    type: "continuous",
    color: "bg-indigo-100",
  },
  prior: {
    name: "DPMM (BayesianGMM)",
    description: "Same as DPMM-Base with warmup_ratio=0.6",
    color: "bg-violet-100",
  },
  decoder: [
    { name: "Linear", dims: `${LATENT_DPMM} → 128`, activation: "BN → Mish → Dropout(0.15)", color: "bg-sky-50" },
    { name: "Linear", dims: "128 → 256", activation: "BN → Mish → Dropout(0.15)", color: "bg-sky-100" },
    { name: "Linear", dims: `256 → ${INPUT_DIM}`, note: "No output activation", color: "bg-sky-200" },
  ],
  extras: [
    { name: "Augmentation", dims: "FeatDrop(0.2)+Noise(σ=0.1)+Mask(0.1)+Scale", color: "bg-violet-50" },
    { name: "Query Proj", dims: `${LATENT_DPMM} → ${LATENT_DPMM} → 128`, activation: "BN → ReLU", color: "bg-violet-100" },
    { name: "Key Proj (m)", dims: "Same arch, m=0.999", note: "Momentum", color: "bg-violet-100" },
    { name: "Memory Queue", dims: "4096 × 128", note: "τ=0.2", color: "bg-violet-200" },
    { name: "Prototypes", dims: "10 × 128", note: "Learnable", color: "bg-violet-200" },
  ],
  losses: [
    { name: "Reconstruction", formula: "MSE(x, x̂) = ‖x − x̂‖²₂ / n", color: "bg-slate-100" },
    { name: "DPMM", formula: "−(1/N) Σᵢ log Σₖ πₖ 𝒩(zᵢ|μₖ,Σₖ)", color: "bg-violet-50" },
    { name: "InfoNCE", formula: "−log(exp(q·k⁺/τ) / Σexp(q·kⱼ/τ))", color: "bg-indigo-50" },
    { name: "Symmetric CL", formula: "Bidirectional InfoNCE  ×0.5", color: "bg-indigo-50" },
    { name: "Prototype CL", formula: "Prototype-cluster CL  ×0.3", color: "bg-indigo-50" },
  ],
  paramCount: "1.62M",
  notes: [
    "FeatDrop(0.2) + GaussNoise(σ=0.1) + Mask(0.1) + RandScale(0.8–1.2)",
    "ℒ = ℒ_recon + λ_dpmm·ℒ_DPMM + λ_moco·(InfoNCE + 0.5·Sym + 0.3·Proto)",
    "Queue=4096×128, τ=0.2, m=0.999, prototypes=10×128, λ_moco=0.5",
  ],
};

// ═══════════════════════════════════════════════════════════════
// Topic Series — cyan / teal / slate cool palette
// ═══════════════════════════════════════════════════════════════

export const TOPIC_BASE: ModelArch = {
  id: "topic-base",
  displayName: "Topic-Base",
  series: "topic",
  variant: "base",
  encoder: [
    { name: "Linear", dims: `${INPUT_DIM} → 128`, activation: "BN → ReLU → Dropout(0.0)", color: "bg-cyan-50" },
    { name: "Linear", dims: "128 → 128", activation: "BN → ReLU → Dropout(0.0)", color: "bg-cyan-100" },
    { name: "μ-head", dims: `128 → ${LATENT_TOPIC}`, note: "Mean (logistic-normal)", color: "bg-cyan-200" },
    { name: "σ²-head", dims: `128 → ${LATENT_TOPIC}`, note: "softplus(·) + 1e-4", color: "bg-cyan-200" },
  ],
  latentSpace: {
    name: "θ = softmax(log_θ) on Δ^{K-1}",
    dims: LATENT_TOPIC,
    type: "simplex",
    color: "bg-cyan-100",
  },
  prior: {
    name: "Dirichlet → Logistic-Normal",
    description: "Dir(α) ≈ LN: α_k=s/K (s=10), μ_p=logα−mean(logα), σ²_p=(1−2/K)/α+Σ(1/α)/K²",
    color: "bg-teal-100",
  },
  decoder: [
    { name: "Topic-Word β", dims: `${LATENT_TOPIC} × ${INPUT_DIM}`, note: "Learnable", color: "bg-sky-50" },
    { name: "softmax(β, dim=1)", dims: "Topic→word dist.", note: "Per-topic probs", color: "bg-sky-100" },
    { name: "x̂ = Σ θ_k · β_k", dims: INPUT_DIM, note: "Mixture recon.", color: "bg-sky-100" },
  ],
  losses: [
    { name: "Reconstruction", formula: "−(1/N) Σᵢᵥ x̄ᵢᵥ log(x̂ᵢᵥ+ε)  [x̄ = L1-norm]", color: "bg-slate-100" },
    { name: "KL Divergence", formula: "D_KL(q(θ|x) ‖ LN(μ_p,σ_p)) + 50·Σmax(0, 0.1−KL_d)", color: "bg-teal-50" },
  ],
  paramCount: "433K",
  notes: [
    "log θ = μ + σ⊙ε, θ=softmax(log θ) — reparameterised on simplex",
    "TopicDecoder: β̃_k = softmax(β_k), log p(w|θ) = LSE_k[log θ_k + log β̃_kv]",
    "Free bits: 0.1 nats/dim, penalty_weight=50 prevents posterior collapse",
  ],
};

export const TOPIC_TRANSFORMER: ModelArch = {
  id: "topic-transformer",
  displayName: "Topic-Transformer",
  series: "topic",
  variant: "transformer",
  encoder: [
    { name: "8× Proj Heads", dims: `${INPUT_DIM} → d_model(128)`, activation: "LN → GELU → Drop", color: "bg-cyan-50" },
    { name: "Token Embs", dims: "1 × 8 × 128", note: "Learnable positions", color: "bg-cyan-50" },
    { name: "TransformerEnc", dims: "2L, 4H, d=128, ff=256", activation: "GELU", color: "bg-cyan-100" },
    { name: "Cross-Attn", dims: "query → MHA(Q,K,V)", note: "Aggregation", color: "bg-cyan-100" },
    { name: "LayerNorm", dims: "128", color: "bg-cyan-200" },
    { name: "μ-head", dims: `128 → ${LATENT_TOPIC}`, note: "Mean", color: "bg-cyan-200" },
    { name: "σ²-head", dims: `128 → ${LATENT_TOPIC}`, note: "softplus(·)+1e-4", color: "bg-cyan-200" },
  ],
  latentSpace: {
    name: "θ = softmax(log_θ) on Δ^{K-1}",
    dims: LATENT_TOPIC,
    type: "simplex",
    color: "bg-cyan-100",
  },
  prior: {
    name: "Dirichlet → Logistic-Normal",
    description: "Same Dir(α)→LN prior; free bits=0.1, pen_weight=10·Σmax(0,fb−KL_d)²",
    color: "bg-teal-100",
  },
  decoder: [
    { name: "Topic-Word β", dims: `${LATENT_TOPIC} × ${INPUT_DIM}`, note: "Learnable", color: "bg-sky-50" },
    { name: "softmax(β, dim=1)", dims: "Topic→word dist.", color: "bg-sky-100" },
    { name: "x̂ = Σ θ_k · β_k", dims: INPUT_DIM, note: "Mixture recon.", color: "bg-sky-100" },
  ],
  losses: [
    { name: "Reconstruction", formula: "−(1/N) Σᵢᵥ x̄ᵢᵥ log(x̂ᵢᵥ+ε)", color: "bg-slate-100" },
    { name: "KL Divergence", formula: "D_KL(q‖LN) + 10·Σmax(0,0.1−KL_d)²", color: "bg-teal-50" },
  ],
  paramCount: "3.44M",
  notes: [
    "Transformer encoder + same parametric β decoder as Base",
    "Squared free-bits penalty (pen_weight=10) for transformer variant",
  ],
};

export const TOPIC_CONTRASTIVE: ModelArch = {
  id: "topic-contrastive",
  displayName: "Topic-Contrastive",
  series: "topic",
  variant: "contrastive",
  encoder: [
    { name: "Linear", dims: `${INPUT_DIM} → 128`, activation: "BN → ReLU → Dropout(0.0)", color: "bg-cyan-50" },
    { name: "Linear", dims: "128 → 128", activation: "BN → ReLU → Dropout(0.0)", color: "bg-cyan-100" },
    { name: "μ-head", dims: `128 → ${LATENT_TOPIC}`, color: "bg-cyan-200" },
    { name: "σ²-head", dims: `128 → ${LATENT_TOPIC}`, note: "softplus(·)+1e-4", color: "bg-cyan-200" },
  ],
  latentSpace: {
    name: "θ = softmax(log_θ) on Δ^{K-1}",
    dims: LATENT_TOPIC,
    type: "simplex",
    color: "bg-cyan-100",
  },
  prior: {
    name: "Dirichlet → Logistic-Normal",
    description: "Same Dir(α)→LN prior with free bits",
    color: "bg-teal-100",
  },
  decoder: [
    { name: "Topic-Word β", dims: `${LATENT_TOPIC} × ${INPUT_DIM}`, note: "Learnable", color: "bg-sky-50" },
    { name: "softmax(β, dim=1)", dims: "Topic→word dist.", color: "bg-sky-100" },
    { name: "x̂ = Σ θ_k · β_k", dims: INPUT_DIM, note: "Mixture recon.", color: "bg-sky-100" },
  ],
  extras: [
    { name: "Augmentation", dims: "FeatDrop(0.2)+Noise(σ=0.1)+Mask(0.1)+ExtraDrop(0.05)", color: "bg-purple-50" },
    { name: "Query Proj", dims: `${LATENT_TOPIC} → ${LATENT_TOPIC} → 64`, activation: "BN → ReLU", color: "bg-purple-100" },
    { name: "Key Proj (m)", dims: "Same arch, m=0.999", note: "Momentum EMA", color: "bg-purple-100" },
    { name: "Memory Queue", dims: "4096 × 64", note: "τ=0.2", color: "bg-purple-200" },
    { name: "Prototypes", dims: "10 × 64", note: "Learnable, Xavier init", color: "bg-purple-200" },
  ],
  losses: [
    { name: "Reconstruction", formula: "−(1/N) Σᵢᵥ x̄ᵢᵥ log(x̂ᵢᵥ+ε)", color: "bg-slate-100" },
    { name: "KL Divergence", formula: "D_KL(q(θ|x) ‖ LN(μ_p,σ_p)) + 50·free_bits_penalty", color: "bg-teal-50" },
    { name: "InfoNCE", formula: "0.1 × CE(logits, labels)", color: "bg-indigo-50" },
    { name: "Symmetric CL", formula: "0.05 × MSE(log θ_q, log θ_k)   [NOT InfoNCE]", color: "bg-indigo-50" },
    { name: "Prototype CL", formula: "0.05 × Prototype CL", color: "bg-indigo-50" },
    { name: "Sparsity", formula: "0.01 × (−Σ θ_k log θ_k)", color: "bg-purple-50" },
    { name: "Diversity", formula: "0.01 × Σ θ̄_k log θ̄_k", color: "bg-purple-50" },
  ],
  paramCount: "436K",
  notes: [
    "Fundamentally different CL from DPMM: embedding=64, hardcoded weights 0.1/0.05/0.05",
    "Sym CL = MSE(log θ_q, log θ_k), NOT bidirectional InfoNCE",
    "Additional topic-specific sparsity + diversity regularisation (wt=0.01 each)",
  ],
};

// ═══════════════════════════════════════════════════════════════
// Grouped by series
// ═══════════════════════════════════════════════════════════════

export const ALL_MODELS: ModelArch[] = [
  DPMM_BASE, DPMM_TRANSFORMER, DPMM_CONTRASTIVE,
  TOPIC_BASE, TOPIC_TRANSFORMER, TOPIC_CONTRASTIVE,
];

export const DPMM_MODELS = [DPMM_BASE, DPMM_TRANSFORMER, DPMM_CONTRASTIVE];
export const TOPIC_MODELS = [TOPIC_BASE, TOPIC_TRANSFORMER, TOPIC_CONTRASTIVE];

// ODE module (shared, Phase 2 — frozen encoder/decoder)
export const ODE_MODULE = {
  dpmm: {
    name: "TimeConditionedODE",
    layers: "dz/dt = W₂ ELU(W₁[z; t]),  h=25",
    timeHead: "Linear(d → 1) → Sigmoid",
    integration: "torchdiffeq.odeint (dopri5)",
    lossWeights: "α_consist=1.0, α_recon=1.0",
  },
  topic: {
    name: "TopicODEFunction",
    layers: "d(log θ)/dt = W₂ ELU(W₁[log θ; t]),  h=64",
    timeHead: "Linear(K → 64) → ELU → Linear(64 → 1) → σ",
    integration: "torchdiffeq.odeint (dopri5)",
    lossWeights: "α_consist=1.0, α_recon=1.0",
  },
};
