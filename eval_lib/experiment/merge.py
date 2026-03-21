"""merge — Portable cross-experiment result merger.

Provides ``MergedExperimentConfig`` which combines result CSVs from
multiple experiment folders into a single unified output, so that
REA / visualise_experiment.py can treat them as one experiment.

This module is intentionally dependency-light (only ``pandas`` +
stdlib) to simplify portability across projects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class MergedExperimentConfig:
    """Combine methods from multiple experiment result folders into one view.

    Each *source* is a ``(tables_dir, series_dir, method_names)`` triple.
    The merge operation reads per-dataset CSVs from each source folder,
    filters to the requested methods, concatenates, and writes the merged
    CSVs to a new output folder — letting REA / ``visualize_experiment.py``
    treat them as a single experiment.

    Example
    -------
    >>> merged = MergedExperimentConfig(
    ...     name="dpmm_vs_external",
    ...     sources=[
    ...         {"tables": "experiments/results/ablation_dpmm/tables",
    ...          "series": "experiments/results/ablation_dpmm/series",
    ...          "methods": ["DPMM-Base", "DPMM-Contrastive"]},
    ...         {"tables": "experiments/results/external_benchmark/tables",
    ...          "series": "experiments/results/external_benchmark/series",
    ...          "methods": ["CellBLAST", "GMVAE", "SCALEX"]},
    ...     ],
    ... )
    >>> merged.build_merged_tables()    # writes merged CSVs
    """

    name: str = "merged"
    sources: List[Dict] = field(default_factory=list)
    output_root: Path = Path("experiments/results")
    description: str = ""

    @property
    def tables_dir(self) -> Path:
        return self.output_root / self.name / "tables"

    @property
    def series_dir(self) -> Path:
        return self.output_root / self.name / "series"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / self.name / "figures"

    @property
    def method_names(self) -> List[str]:
        """All methods in source order."""
        result = []
        for src in self.sources:
            for m in src.get("methods", []):
                if m not in result:
                    result.append(m)
        return result

    def build_merged_tables(self) -> Path:
        """Read CSVs from each source, merge by dataset, write to tables_dir.

        Returns the path to the merged tables directory.
        """
        import pandas as pd

        tables_dir = self.tables_dir
        series_dir = self.series_dir
        tables_dir.mkdir(parents=True, exist_ok=True)
        series_dir.mkdir(parents=True, exist_ok=True)

        # Discover all dataset keys across sources
        all_ds_keys: Dict[str, List[Tuple[Path, Optional[List[str]]]]] = {}
        for src in self.sources:
            src_tables = Path(src["tables"])
            methods_filter = src.get("methods")  # None = all methods

            if not src_tables.exists():
                print(f"  WARNING: source tables dir not found: {src_tables}")
                continue

            for csv_path in sorted(src_tables.glob("*_df.csv")):
                ds_key = csv_path.stem[:-3]  # strip "_df"
                if ds_key not in all_ds_keys:
                    all_ds_keys[ds_key] = []
                all_ds_keys[ds_key].append((csv_path, methods_filter))

        # Also collect series files
        all_ds_series: Dict[str, List[Tuple[Path, Optional[List[str]]]]] = {}
        for src in self.sources:
            src_series = Path(src.get("series", ""))
            methods_filter = src.get("methods")
            if src_series.exists():
                for csv_path in sorted(src_series.glob("*_dfs.csv")):
                    ds_key = csv_path.stem[:-4]  # strip "_dfs"
                    if ds_key not in all_ds_series:
                        all_ds_series[ds_key] = []
                    all_ds_series[ds_key].append((csv_path, methods_filter))

        merged_count = 0
        for ds_key, source_list in sorted(all_ds_keys.items()):
            parts = []
            for csv_path, methods_filter in source_list:
                df = pd.read_csv(csv_path)
                if methods_filter is not None and "method" in df.columns:
                    df = df[df["method"].isin(methods_filter)]
                parts.append(df)

            if not parts:
                continue

            merged_df = pd.concat(parts, ignore_index=True)
            # Reorder to match method_names order
            desired_order = self.method_names
            if "method" in merged_df.columns:
                ordered = []
                for m in desired_order:
                    match = merged_df[merged_df["method"] == m]
                    if not match.empty:
                        ordered.append(match)
                leftover = merged_df[~merged_df["method"].isin(desired_order)]
                if not leftover.empty:
                    ordered.append(leftover)
                if ordered:
                    merged_df = pd.concat(ordered, ignore_index=True)

            out_csv = tables_dir / f"{ds_key}_df.csv"
            merged_df.to_csv(out_csv, index=False)
            merged_count += 1

        # Merge series too
        series_count = 0
        for ds_key, source_list in sorted(all_ds_series.items()):
            parts = []
            for csv_path, methods_filter in source_list:
                df = pd.read_csv(csv_path)
                if methods_filter is not None and "hue" in df.columns:
                    df = df[df["hue"].isin(methods_filter)]
                parts.append(df)

            if not parts:
                continue
            merged_df = pd.concat(parts, ignore_index=True)
            out_csv = series_dir / f"{ds_key}_dfs.csv"
            merged_df.to_csv(out_csv, index=False)
            series_count += 1

        print(f"  Merged {merged_count} dataset tables, {series_count} series "
              f"-> {tables_dir}")
        return tables_dir

    def summary(self) -> str:
        lines = [
            f"Merged Experiment: {self.name}",
            f"Description: {self.description}",
            f"Methods: {self.method_names}",
            f"Sources: {len(self.sources)}",
        ]
        for i, src in enumerate(self.sources):
            lines.append(f"  [{i+1}] {src.get('tables', '?')} -> {src.get('methods', 'ALL')}")
        return "\n".join(lines)
