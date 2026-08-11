# PanODE-Topic Audit Report

Date: 2026-03-22

## Scope
This report audits the repository structure, major subfolders, training pipeline, data pipeline, evaluation pipeline, and the local result artifacts currently present in this checkout.

This audit is based on:
- Source review of key files in `models/`, `utils/`, `benchmarks/`, `eval_lib/`, `experiments/`, `scripts/`, `refined_figures/`, `article/`, `src/`, `vcd/`, and `model-arch-viewer/`.
- Presence checks on local output directories such as `benchmarks/benchmark_results/` and `experiments/results/`.
- Spot checks of local artifacts such as per-dataset CSVs and the generated statistical report.

This audit does not include:
- Re-running any training.
- Recomputing metrics.
- Verifying raw external datasets under `$PANODE_DATASETS_ROOT` (or `~/datasets`).
- Reviewing every generated artifact file one by one.

## Highest-Priority Findings

1. [High] Repository completeness is broken for the benchmark registry.
   - `benchmarks/model_registry.py` imports `models.dpmm_base`, `models.dpmm_transformer`, `models.dpmm_contrastive`, and `models.pure_ae`, but these files are not present in `models/`.
   - `python -c "import benchmarks.model_registry"` fails with `ModuleNotFoundError: No module named 'models.dpmm_base'`.
   - Impact: core benchmark entry points are not runnable as-is from this checkout.

2. [High] Training-length control is ambiguous and can invalidate epoch sweeps.
   - `benchmarks/train_utils.py` uses `fit_epochs = params.pop("fit_epochs", epochs)`.
   - If model params contain `fit_epochs`, runner-supplied `epochs=` can be ignored.
   - Impact: some reported training-sweep conclusions may not reflect the requested epoch count.

3. [High] Validation-based model selection is not consistently applied.
   - `utils/base_model.py` supports validation loss, LR scheduling, and validation-based early stopping.
   - `models/topic_base.py` and `models/pure_vae.py` use custom `fit()` loops that stop on training loss instead.
   - Impact: "trained enough" is not strongly justified by validation behavior for major model families.

4. [High] Label handling is inconsistent across datasets.
   - `benchmarks/dataset_registry.py` defines dataset-specific `label_key`.
   - `benchmarks/runners/benchmark_base.py` only checks for `clusters` or `cell_type`.
   - Some datasets use keys like `celltype`, `Clusters`, or `Cell type`.
   - Impact: annotated labels can be missed, with fallback to pseudo-labels or inconsistent evaluation.

5. [High] The data pipeline is broad but not self-contained.
   - Dataset roots should be supplied via `PANODE_DATASETS_ROOT` (see `benchmarks/dataset_registry.py`).
   - The repository does not ship the `.h5ad` files or a full versioned dataset manifest.
   - Impact: external auditors cannot independently verify data sufficiency from the repo alone.

6. [Medium] Results are locally abundant but weakly portable.
   - Local output directories contain thousands of generated files.
   - `.gitignore` excludes most benchmark outputs, experiment results, figures, and biological results.
   - Impact: internal evidence exists, but the audit trail is not portable through git.

7. [Medium] Statistical reporting is usable but not yet audit-grade.
   - The local `statistical_report.md` exists and is structured.
   - The script uses one-sided Wilcoxon tests on dataset-level means and marks significance at `p < 0.05` without global multiplicity correction.
   - Impact: result claims should be treated as provisional rather than final.

8. [Medium] Documentation and naming drift reduce audit readiness.
   - Root README under-describes the current tree.
   - Some scripts expect a `docs/` folder that is absent in this checkout.
   - Local manifests refer to `PanODE-LAB` while this repo is `PanODE-Topic`.
   - Impact: future refinement is slower and more error-prone.

## Executive Answers to the Main Audit Questions

### Is the current model training sufficient?

Verdict: partially supported for exploratory benchmarking, but not strong enough yet for a strict "training sufficiency confirmed" claim.

Why:
- There is a canonical configuration with `epochs = 1000`, `batch_size = 128`, `latent_dim = 10`, and fixed output directories.
- Local benchmark artifacts are present in large volume, which suggests the training pipeline has been exercised extensively.
- However, model-specific training loops do not consistently use validation-based stopping, and runner-requested epoch counts can be silently overridden by `fit_epochs`.
- Because of those issues, current evidence is better described as "many runs were executed" than "training sufficiency was rigorously proven".

Upgrade path to a stronger claim:
- Make epoch precedence explicit and test it.
- Use validation-based selection consistently.
- Save and compare multi-seed convergence summaries at the metric level, not just the loss level.

### Is the current training data sufficient?

Verdict: broad enough for relative benchmarking, not strong enough alone for broad biological or atlas-scale claims.

Why:
- The registry covers a large set of single-cell datasets across development, cancer, disease, and perturbation settings.
- That breadth is a real strength.
- But the default benchmark path subsamples to `3000` cells and `3000` HVGs, many datasets rely on Leiden-based pseudo-labels, and the raw datasets are external and machine-specific.
- This supports comparative testing under a fixed protocol, not a full claim that the models were trained on sufficiently rich data for every downstream conclusion.

Upgrade path to a stronger claim:
- Record the exact dataset list, source/version, label provenance, and split statistics.
- Report results on both the default 3k-cell protocol and a higher-scale no-subsample protocol.
- Separate author-annotated labels from Leiden pseudo-label datasets in the results summary.

### Are the current results clear enough?

Verdict: clear enough for internal iteration, not yet clear enough for external or archival auditing.

Why:
- Result tables, merged experiment outputs, paper figures, and statistical exports all exist locally and use fairly clear directory structures.
- Per-dataset tables such as `experiments/results/topic_vs_external/tables/*.csv` are machine-readable and standardized.
- A generated statistical report also exists locally.
- However, these outputs are mostly gitignored, local manifests show naming/path drift, and the statistical pipeline still has inferential weaknesses.

Upgrade path to a stronger claim:
- Publish a frozen result bundle or release artifact.
- Save run manifests with repository hash, environment hash, dataset manifest hash, and exact source CSVs used for statistics.
- Add multiple-testing correction and seed-level paired analysis.

## Audited Directory Tree

Project-relevant tree observed in this checkout:

```text
PanODE-Topic/
├── article/
│   └── topic/
├── benchmarks/
│   ├── benchmark_results/          # present locally, gitignored
│   ├── biological_validation/
│   ├── external_models/
│   ├── figure_generators/
│   ├── paper_figures/              # present locally, gitignored
│   ├── runners/
│   └── training_dynamics_results/  # present locally, gitignored
├── eval_lib/
│   ├── baselines/
│   ├── experiment/
│   ├── metrics/
│   └── viz/
├── experiments/
│   └── results/                    # present locally, gitignored
├── fonts/
├── model-arch-viewer/
│   ├── public/
│   ├── scripts/
│   └── src/
├── models/
├── refined_figures/
├── scripts/
├── src/
│   └── visualization/
├── utils/
└── vcd/
```

Notable absent or incomplete areas:
- `docs/` is absent, even though some scripts expect it.
- `refined_figures/output/` is absent in this checkout.
- Several `models/*.py` files expected by `benchmarks/model_registry.py` are missing.

## Local Artifact Snapshot

Local generated outputs currently present on disk:

- `benchmarks/benchmark_results/`: 88 subdirectories, 2569 files
- `benchmarks/training_dynamics_results/`: 0 subdirectories, 9 files
- `benchmarks/biological_validation/results/`: 3 subdirectories, 568 files
- `benchmarks/paper_figures/`: 22 subdirectories, 855 files
- `experiments/results/`: 27 subdirectories, 374 files
- `refined_figures/output/`: not present
- `docs/`: not present

Interpretation:
- The project has substantial local evidence of benchmarking and result generation.
- That evidence is real, but it is not portable by default because the outputs are gitignored.

## Per-Folder Review

### `README.md`, `.gitignore`, and root-level files

Status: needs cleanup for audit readiness.

- `README.md` gives a useful overview of the intended project layout, model variants, and canonical hyperparameters.
- The README is stale relative to the actual tree. It does not fully cover `vcd/`, the full `scripts/` surface, or the current local-output reality.
- `.gitignore` is very aggressive. It keeps the repo clean, but it also hides nearly all audit evidence from version control.
- `merge_external_results.py` is important to the evaluation story but sits at the repo root instead of inside `scripts/`, which makes the operational flow less obvious.

Recommended follow-up:
- Update the README tree and add a short "audit and reproduction" section.
- Move or document root-level operational scripts more clearly.
- Keep generated artifacts ignored, but track a lightweight manifest of what was produced.

### `article/` and `article/topic/`

Status: good structure, moderate evidence risk.

- This area is clearly organized as manuscript source, with tracked tables under `article/topic/tables/` and build support in `article/build.sh`.
- It is a strong narrative anchor for the paper side of the project.
- The main risk is that article assets look publication-ready while the upstream raw outputs that feed them are mostly not versioned.

Recommended follow-up:
- Add a note in the article build path that lists which generated inputs must exist before building.
- Link manuscript tables back to the source result files used to create them.

### `models/`

Status: high priority remediation needed.

- This folder contains topic and flow-matching model implementations and is central to the scientific contribution.
- The folder is incomplete relative to `benchmarks/model_registry.py`. Missing modules include `dpmm_base.py`, `dpmm_transformer.py`, `dpmm_contrastive.py`, and `pure_ae.py`.
- Because of that mismatch, registry-driven benchmarks are not runnable as-is from this checkout.
- The training semantics also need alignment: `topic_base.py` and `pure_vae.py` use custom train-loss-based early stopping, while `utils/base_model.py` supports validation-based stopping and scheduling.

Recommended follow-up:
- Restore or remove missing model modules so `benchmarks.model_registry` can import cleanly.
- Unify training semantics around one clearly defined `fit()` contract.
- Add tests that assert all registry models import and train for at least a smoke-test epoch.

### `utils/`

Status: solid base, but not fully enforced by callers.

- `utils/base_model.py` provides the cleanest training contract in the repo: validation-aware training, LR scheduling, and checkpoint behavior.
- `utils/data.py` offers a generic data-splitting pipeline, but the split logic is random rather than stratified.
- The main issue is not that `utils/` is weak; it is that the higher-level model implementations do not fully adhere to the base abstractions.

Recommended follow-up:
- Make `BaseModel.fit()` the required baseline behavior unless a model documents a justified exception.
- Add stratified split support where labels are available.

### `benchmarks/`

Status: high value, moderate-to-high risk.

- This is the operational heart of the repo: config, dataset registry, model registry, runners, figures, and local outputs.
- It is a strength that so much of the benchmark workflow is centralized here.
- The tradeoff is that code, local cache, figures, and results all live under one large tree, which increases drift risk.

Recommended follow-up:
- Keep `benchmarks/` as the home for operational benchmarking, but document its internal contract more clearly.
- Separate source-only documentation from generated-output conventions.

### `benchmarks/runners/`

Status: needs careful cleanup before strong benchmarking claims.

- Coverage is broad: `benchmark_base`, `benchmark_training`, `benchmark_crossdata`, `benchmark_scalability`, `benchmark_preprocessing`, `benchmark_joint`, `benchmark_transfer`, and more.
- This is good experimental coverage.
- The main audit risks are runner inconsistency:
  - Requested `epochs=` can be overridden by model params carrying `fit_epochs`.
  - `benchmark_base.py` does not consistently honor dataset registry `label_key`.
  - Different runners may expose slightly different training semantics or data assumptions.

Recommended follow-up:
- Standardize a single runner contract for training length, label resolution, seeds, and manifest writing.
- Add a small regression suite that checks runner behavior on one toy dataset.

### `benchmarks/external_models/`

Status: strong comparison breadth, high maintenance surface.

- This folder provides many external baseline wrappers and associated distribution helpers.
- It is valuable for comparison coverage and supports the "vs external" benchmarking story.
- The downside is maintenance complexity, version drift, and duplicated logic relative to `eval_lib/baselines/`.

Recommended follow-up:
- Document which external models are actively supported and which are archival.
- Reduce duplicated distribution/helper code where possible.

### `benchmarks/figure_generators/`

Status: good publication-engineering layer.

- The figure generator stack is mature and clearly intended for paper-ready composition.
- This folder helps turn raw CSV outputs into standardized subplots and final panels.
- The main fragility is that it assumes stable local naming conventions and available generated inputs.

Recommended follow-up:
- Add a manifest file per figure that records the exact input CSVs and output files used.
- Fail fast with clear messages when required result files are missing.

### `benchmarks/biological_validation/`

Status: useful and reasonably well documented.

- This area includes a README, specialized downstream analyses, and a populated local results directory.
- It adds biological interpretability beyond generic clustering or embedding metrics.
- The main limitation is coverage: the current wiring appears stronger for core datasets than for the full expanded registry.

Recommended follow-up:
- Explicitly document which datasets are supported for each biological validation step.
- Add a coverage matrix for core vs expanded datasets.

### `benchmarks/benchmark_results/`

Status: strong local evidence, weak versioned evidence.

- This directory is present locally with 88 subdirectories and 2569 files.
- It contains logs, CSVs, models, statistical exports, and benchmark-specific subtrees.
- This is one of the strongest signs that the project has actually been run at scale.
- The audit risk is portability: none of this evidence is versioned, and the local `run_manifest.jsonl` references `PanODE-LAB` paths instead of this repo path.

Recommended follow-up:
- Preserve the full directory as local scratch output, but add a tracked summary manifest or release bundle.
- Normalize path recording so manifests refer to the current repository name or use relative paths.

### `benchmarks/paper_figures/`

Status: good evidence of figure production, but local-only.

- This directory is present locally with 22 subdirectories and 855 files.
- It shows that the paper-figure production pipeline has been used.
- The artifact count is encouraging, but none of it is tracked in git.

Recommended follow-up:
- Track a minimal figure index that maps figure names to source CSVs and generation scripts.
- Consider publishing a compressed artifact bundle for each major paper refresh.

### `benchmarks/training_dynamics_results/`

Status: useful supporting evidence, but limited.

- This directory exists locally and contains 9 files.
- It suggests that training-dynamics analysis has been run at least in some form.
- On its own, this is not enough to claim training sufficiency. It is supporting material rather than decisive evidence.

Recommended follow-up:
- Save convergence summaries per model, dataset, and seed in a stable tabular format.

### `eval_lib/`

Status: one of the best-structured areas in the repo.

- The library cleanly separates metrics, baselines, experiment utilities, and visualization/statistics.
- It is easier to audit than many other areas because the responsibilities are more modular.
- The main risk is not organization, but the inference strength of the downstream statistical scripts that consume these utilities.

Recommended follow-up:
- Keep `eval_lib/` as the stable core and move more one-off logic out of ad hoc scripts into this layer.

### `eval_lib/metrics/`

Status: strong metric centralization.

- This folder is a major strength because it defines a canonical metric battery and column naming scheme.
- Result tables are clearer because metrics are named and grouped consistently.
- The caution is that a wide metric surface increases the burden on statistical control and interpretation.

Recommended follow-up:
- Keep a tracked metric glossary that explains which metrics are primary, secondary, and exploratory.

### `eval_lib/experiment/`

Status: good merge/config infrastructure.

- The experiment configuration and merge utilities give the repo a coherent output schema.
- Templates in `eval_lib/experiment/templates/` are helpful for reusing the structure.
- This area helps result clarity and future refinement.

Recommended follow-up:
- Record the exact merged input sources for every merged output directory.

### `eval_lib/baselines/`

Status: valuable, but overlap should be managed.

- This folder provides reusable baseline wrappers and its own registry.
- It is useful for keeping external comparison logic available outside the benchmark scripts.
- The main risk is overlap with `benchmarks/external_models/`, which can cause maintenance drift.

Recommended follow-up:
- Document the relationship between `eval_lib/baselines/` and `benchmarks/external_models/`.
- Consolidate where feasible.

### `eval_lib/viz/`

Status: strong analysis/presentation layer with statistical caveats upstream.

- This folder supports rigorous experiment visualization and comparative analysis.
- It improves clarity of outputs and is one of the more mature presentation layers.
- However, visualization rigor does not automatically imply statistical rigor; the weakest points are still in how source tests are selected and interpreted.

Recommended follow-up:
- Surface test assumptions and multiplicity decisions directly in generated figure legends or captions.

### `experiments/`

Status: thin but useful orchestration layer.

- This folder provides focused entry points such as `run_external_benchmark.py` and `merge_and_visualize.py`.
- It is small, which helps readability.
- It depends heavily on the rest of the pipeline being intact and the result directories being available.

Recommended follow-up:
- Add a short folder README that explains when to use `experiments/` instead of `benchmarks/`.

### `experiments/results/`

Status: locally strong, externally weak.

- This directory is present locally with 27 subdirectories and 374 files.
- The layout is clear: examples include `tables/`, `series/`, and `figures/`.
- A sampled file such as `experiments/results/topic_vs_external/tables/setty_df.csv` shows a consistent metric schema.
- The main weakness is again portability: the directory is gitignored and therefore not part of the tracked audit record.

Recommended follow-up:
- Keep the directory structure, but export a tracked manifest or release asset for each major experiment refresh.

### `refined_figures/`

Status: clear figure intent, incomplete local output state.

- The folder is well organized around one script per figure plus a dispatcher.
- This is a good pattern for paper refinement.
- In this checkout, `refined_figures/output/` is absent, so the code is present but the corresponding local output folder is not.

Recommended follow-up:
- Record which benchmark outputs are prerequisites for each refined figure.
- Save figure-generation manifests alongside final outputs.

### `scripts/`

Status: operationally important and under-documented.

- This folder is larger and more capable than the root README suggests.
- It includes dataset prep, figure refresh, combined CSV generation, statistical reporting, and parallel execution helpers.
- The folder adds significant operational power, but it also carries several audit risks:
  - Some scripts expect a `docs/` directory that is absent.
  - There is no root `requirements.txt` or `pyproject.toml` documenting the Python environment.
  - `scripts/statistical_analysis.py` selects the source combined CSV by sorted filename rather than an explicit manifest-based choice.
  - The same statistical script uses one-sided Wilcoxon tests on dataset-level means and flags `p < 0.05` without global correction.

Recommended follow-up:
- Add a top-level environment file.
- Add a `scripts/README.md` summarizing purpose, prerequisites, and output locations.
- Tighten the statistical methodology and record which source CSV is being analyzed.

### `model-arch-viewer/`

Status: useful presentation tool, lightly documented.

- This is a separate Next.js application for architecture views and figure pages.
- Its purpose is inferable from the code and routes, but there is no dedicated folder README and the `package.json` description is empty.
- It is not a scientific blocker, but it is part of the overall results-presentation story.

Recommended follow-up:
- Add a short README explaining how it relates to the paper figures and local subplot assets.

### `src/` and `src/visualization/`

Status: clear support layer with minor documentation drift.

- The `src/visualization/` package gives the repo a reusable layout and style layer for figures.
- This is a good sign that presentation logic is being centralized rather than scattered.
- The main issue is documentation drift: the root README under-describes what is actually present here.

Recommended follow-up:
- Expand the README section for `src/visualization/` and document the expected consumers of this package.

### `vcd/`

Status: valuable quality-control layer.

- The Visual Conflict Detector is a good addition because it targets layout, legend, text, and perceptual figure issues.
- This improves result clarity and publication polish.
- It does not answer questions about model validity, training sufficiency, or statistical sufficiency on its own.

Recommended follow-up:
- Keep `vcd/` focused on visual quality and avoid treating it as scientific validation.
- Add a short note in the audit docs describing exactly what VCD does and does not guarantee.

### `fonts/`

Status: low risk, operational asset folder.

- This folder exists for figure rendering assets.
- It is not central to the scientific audit.
- No major issue beyond the fact that it is ignored and therefore not portable by default.

Recommended follow-up:
- Document the required fonts or provide a fallback path.

### `docs/` (expected but absent)

Status: missing.

- Some scripts refer to blueprint files under `docs/`, but the folder is not present in this checkout.
- This is a direct documentation gap and slows subsequent auditing/refinement.

Recommended follow-up:
- Either restore `docs/` or remove the dependency on it from scripts.
- If the folder is intentionally external, document that clearly.

## Training Audit Notes

What is already good:
- Canonical benchmark configuration exists in `benchmarks/config.py`.
- Global seeding is implemented.
- Output directories are centralized.
- Local result volume suggests the team did substantial experimentation.

What is not yet strong enough:
- Training control is split across multiple `fit()` implementations.
- Validation-based stopping/checkpointing is not consistently honored.
- The repo does not currently prove that the chosen epoch budgets are sufficient across datasets and seeds.

Audit conclusion:
- Training is active and substantial.
- Training sufficiency is not yet demonstrated strongly enough for a final audit sign-off.

## Data Audit Notes

What is already good:
- Dataset coverage is broad for a single-cell benchmarking repo.
- The registry captures task type, species, and label-key metadata.
- Dedicated prep scripts exist for large dataset sets.

What limits the claim:
- Default benchmarking uses `max_cells = 3000` and `hvg_top_genes = 3000`.
- Many datasets rely on Leiden-based pseudo-labels rather than only author annotations.
- Dataset paths are local and raw data are not shipped.

Audit conclusion:
- Data breadth is good for comparative benchmarking.
- Data sufficiency is only moderate for strong claims about generality, biology, or scalability.

## Results Audit Notes

What is already good:
- Local results are present in large volume.
- Per-dataset tables and merged experiment outputs use consistent schemas.
- Statistical reports and paper figures exist locally.

What limits the claim:
- Outputs are gitignored and not portable.
- Statistical significance is not corrected globally across many comparisons.
- Run manifests show naming/path drift.
- The repo does not yet produce a compact, tracked audit bundle.

Audit conclusion:
- Results are clear enough for internal iteration and refinement.
- Results are not yet fully clear enough for long-term external auditing without additional packaging and statistical cleanup.

## Recommended Refinement Order

1. Restore repository completeness.
   - Make all registry-imported model files present, or remove dead imports.
   - Add an import smoke test for `benchmarks.model_registry`.

2. Fix training control semantics.
   - Decide whether runner `epochs` or model `fit_epochs` takes precedence.
   - Use one documented rule everywhere.

3. Standardize validation behavior.
   - Align model-specific `fit()` loops with `BaseModel.fit()` or document justified exceptions.

4. Standardize dataset label resolution.
   - Always use registry `label_key`.
   - Log which label source was used for each run.

5. Make the dataset and environment reproducible.
   - Add a root Python environment file.
   - Add a tracked dataset manifest with source, version, label provenance, and local override instructions.

6. Improve statistical audit quality.
   - Use family-wise or FDR correction where appropriate.
   - Prefer seed-level paired analysis instead of collapsing to dataset-level means before testing.
   - Record the exact input CSV chosen for each statistical report.

7. Add a tracked audit artifact bundle.
   - Save a lightweight manifest of runs, datasets, hashes, and result sources.
   - Optionally publish frozen CSV/figure bundles for major refreshes.

8. Repair documentation drift.
   - Update README tree.
   - Add `scripts/README.md`.
   - Restore or document the expected `docs/` folder.
   - Unify `PanODE-Topic`, `PanODE-LAB`, and `PanODE` naming where possible.

## Re-Audit Checklist

Use this list for the next audit cycle:

- [ ] `benchmarks.model_registry` imports successfully in a clean environment.
- [ ] Every registry model has a smoke-test training run.
- [ ] Runner-specified epoch sweeps are verified to affect actual training length.
- [ ] Validation-based selection behavior is consistent across all core models.
- [ ] Dataset label provenance is logged for every benchmark run.
- [ ] Dataset and environment manifests are tracked in git.
- [ ] A frozen result bundle exists for the latest benchmark refresh.
- [ ] Statistical reports document correction strategy and exact input files.
- [ ] README and script documentation match the real tree.
- [ ] Naming is consistent across manifests, scripts, and output paths.

## Final Audit Verdict

Repository structure:
- Good core separation of concerns, but documentation drift and missing expected files lower audit readiness.

Model training sufficiency:
- Not yet fully proven.

Training data sufficiency:
- Broad for benchmarking, limited for strong general claims.

Results clarity:
- Good locally, not yet external-audit grade.

Overall:
- The repository is promising and actively used, but it still needs completeness fixes, stronger reproducibility metadata, and tighter statistical/training controls before it should be treated as fully audit-ready.
