import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics import pairwise_distances  # type: ignore
from sklearn.manifold import trustworthiness as sklearn_trustworthiness  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore
from scipy.stats import spearmanr, pearsonr  # type: ignore
import warnings
from typing import Dict, Tuple


class ExtendedDimensionalityReductionEvaluator:
    """
    Extended dimensionality reduction quality evaluator.

    Complements the base DRE evaluator with additional geometry-oriented
    metrics that do not require clustering labels.

    Core metrics:
    - trustworthiness: how well low-D preserves high-D neighborhoods
    - continuity: how well high-D neighborhoods are continued in low-D
    - distance_spearman: Spearman rank correlation of pairwise distances
    - distance_pearson: Pearson correlation of pairwise distances
    - local_scale_quality: local scale distortion quality
    - neighborhood_symmetry: bidirectional neighborhood consistency

    Features:
    - Efficient, vectorized computation
    - Automatic subsampling for large datasets
    - Focus on complementary evaluation to DRE (co-ranking based)
    - All metrics in [0, 1] range (higher is better)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the evaluator.

        Args:
            verbose: whether to print detailed information
        """
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def _validate_inputs(self, X_high: np.ndarray, X_low: np.ndarray):
        """Validate input parameters."""
        if not isinstance(X_high, np.ndarray) or not isinstance(X_low, np.ndarray):
            raise TypeError("Input data must be NumPy arrays.")

        if X_high.shape[0] != X_low.shape[0]:
            raise ValueError(
                f"High- and low-dimensional data must have the same number of samples: "
                f"{X_high.shape[0]} vs {X_low.shape[0]}"
            )

        if X_high.ndim != 2 or X_low.ndim != 2:
            raise ValueError("Input data must be 2D arrays.")

        if X_high.shape[0] < 10:
            raise ValueError("Need at least 10 samples for stable extended DRE metrics.")

    @staticmethod
    def _safe_float(value: float, default: float = 0.0) -> float:
        """Safely convert to float, replacing NaN/Inf with *default*."""
        try:
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return default
            return value
        except Exception:
            return default

    @staticmethod
    def _subsample_aligned(
        x_high: np.ndarray,
        x_low: np.ndarray,
        max_samples: int = 2000,
        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample both arrays in the same way for tractable computation."""
        if x_high.shape[0] <= max_samples:
            return x_high, x_low
        rng = np.random.default_rng(random_state)
        idx = rng.choice(x_high.shape[0], max_samples, replace=False)
        return x_high[idx], x_low[idx]

    # ==================== 1. Trustworthiness ====================

    def trustworthiness_score(
        self, X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 15
    ) -> float:
        """
        Trustworthiness: how well the low-D embedding preserves high-D
        neighborhood structure.

        Args:
            X_high: high-dimensional data
            X_low: low-dimensional data
            n_neighbors: number of neighbors

        Returns:
            float: trustworthiness score in [0, 1] (higher is better)
        """
        try:
            k = min(n_neighbors, X_high.shape[0] - 2)
            return float(sklearn_trustworthiness(X_high, X_low, n_neighbors=k))
        except Exception as e:
            warnings.warn(f"Error computing trustworthiness: {e}")
            return 0.0

    # ==================== 2. Continuity ====================

    def continuity_score(
        self, X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 15
    ) -> float:
        """
        Continuity (complementary to trustworthiness).
        Penalizes true high-dim neighbors missed in low-dim neighborhood.

        Args:
            X_high: high-dimensional data
            X_low: low-dimensional data
            n_neighbors: number of neighbors

        Returns:
            float: continuity score in [0, 1] (higher is better)
        """
        try:
            n = X_high.shape[0]
            k = min(max(2, n_neighbors), n - 2)

            d_high = pairwise_distances(X_high)
            d_low = pairwise_distances(X_low)

            rank_low = np.argsort(np.argsort(d_low, axis=1), axis=1)
            high_nn = np.argsort(d_high, axis=1)[:, 1 : k + 1]

            penalties = 0.0
            for i in range(n):
                ranks = rank_low[i, high_nn[i]]
                miss = ranks[ranks > k]
                if miss.size > 0:
                    penalties += np.sum(miss - k)

            denom = n * k * (2 * n - 3 * k - 1)
            if denom <= 0:
                return 0.0
            return float(1.0 - (2.0 / denom) * penalties)

        except Exception as e:
            warnings.warn(f"Error computing continuity: {e}")
            return 0.0

    # ==================== 3. Distance correlations ====================

    def distance_spearman_score(
        self, X_high: np.ndarray, X_low: np.ndarray
    ) -> float:
        """
        Spearman rank correlation between pairwise distances in high- and
        low-dimensional spaces.

        Args:
            X_high: high-dimensional data
            X_low: low-dimensional data

        Returns:
            float: Spearman correlation (higher is better)
        """
        try:
            d_high = pairwise_distances(X_high)
            d_low = pairwise_distances(X_low)

            tri = np.triu_indices_from(d_high, k=1)
            dh = d_high[tri]
            dl = d_low[tri]

            return self._safe_float(spearmanr(dh, dl).correlation)
        except Exception as e:
            warnings.warn(f"Error computing distance Spearman: {e}")
            return 0.0

    def distance_pearson_score(
        self, X_high: np.ndarray, X_low: np.ndarray
    ) -> float:
        """
        Pearson correlation between pairwise distances in high- and
        low-dimensional spaces.

        Args:
            X_high: high-dimensional data
            X_low: low-dimensional data

        Returns:
            float: Pearson correlation (higher is better)
        """
        try:
            d_high = pairwise_distances(X_high)
            d_low = pairwise_distances(X_low)

            tri = np.triu_indices_from(d_high, k=1)
            dh = d_high[tri]
            dl = d_low[tri]

            return self._safe_float(pearsonr(dh, dl)[0])
        except Exception as e:
            warnings.warn(f"Error computing distance Pearson: {e}")
            return 0.0

    # ==================== 4. Local scale quality ====================

    def local_scale_quality_score(
        self, X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 15
    ) -> float:
        """
        Local scale distortion via neighborhood distance-ratio mismatch.
        Lower distortion → higher quality score.

        Args:
            X_high: high-dimensional data
            X_low: low-dimensional data
            n_neighbors: number of neighbors

        Returns:
            float: local scale quality in [0, 1] (higher is better)
        """
        try:
            n = X_high.shape[0]
            k = min(max(2, n_neighbors), n - 1)

            nn_high = NearestNeighbors(n_neighbors=k + 1).fit(X_high)
            nn_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low)

            dh, _ = nn_high.kneighbors(X_high)
            dl, _ = nn_low.kneighbors(X_low)

            dh = dh[:, 1:]  # drop self-distance
            dl = dl[:, 1:]

            dh_norm = dh / (np.mean(dh, axis=1, keepdims=True) + 1e-12)
            dl_norm = dl / (np.mean(dl, axis=1, keepdims=True) + 1e-12)

            distortion = np.mean(np.abs(np.log((dl_norm + 1e-12) / (dh_norm + 1e-12))))
            return float(np.exp(-distortion))

        except Exception as e:
            warnings.warn(f"Error computing local scale quality: {e}")
            return 0.0

    # ==================== 5. Neighborhood symmetry ====================

    def neighborhood_symmetry_score(
        self, X_high: np.ndarray, X_low: np.ndarray, n_neighbors: int = 15
    ) -> float:
        """
        Bidirectional neighborhood consistency: average of trustworthiness
        and continuity.

        Args:
            X_high: high-dimensional data
            X_low: low-dimensional data
            n_neighbors: number of neighbors

        Returns:
            float: neighborhood symmetry in [0, 1] (higher is better)
        """
        t = self.trustworthiness_score(X_high, X_low, n_neighbors)
        c = self.continuity_score(X_high, X_low, n_neighbors)
        return 0.5 * (t + c)

    # ==================== 6. Comprehensive evaluation ====================

    def comprehensive_evaluation(
        self,
        X_high: np.ndarray,
        X_low: np.ndarray,
        n_neighbors: int = 15,
        max_samples: int = 2000,
        random_state: int = 42) -> Dict[str, float]:
        """
        Comprehensive evaluation of extended DR quality metrics.

        Args:
            X_high: high-dimensional data, shape = (n_samples, n_features_high)
            X_low:  low-dimensional data, shape = (n_samples, n_features_low)
            n_neighbors: number of neighbors for kNN-based metrics
            max_samples: subsample to this many points for tractability
            random_state: random seed for subsampling

        Returns:
            dict: dictionary with all DREX evaluation metrics
        """
        self._validate_inputs(X_high, X_low)

        self._log(
            f"Starting extended DR evaluation "
            f"(n_samples={X_high.shape[0]}, k={n_neighbors})..."
        )

        # Subsample for tractability
        xh, xl = self._subsample_aligned(X_high, X_low, max_samples, random_state)

        results: Dict[str, float] = {}

        # 1. Trustworthiness
        self._log("Computing trustworthiness...")
        results['trustworthiness'] = self.trustworthiness_score(xh, xl, n_neighbors)

        # 2. Continuity
        self._log("Computing continuity...")
        results['continuity'] = self.continuity_score(xh, xl, n_neighbors)

        # 3. Distance correlations
        self._log("Computing distance correlations...")
        results['distance_spearman'] = self.distance_spearman_score(xh, xl)
        results['distance_pearson'] = self.distance_pearson_score(xh, xl)

        # 4. Local scale quality
        self._log("Computing local scale quality...")
        results['local_scale_quality'] = self.local_scale_quality_score(xh, xl, n_neighbors)

        # 5. Neighborhood symmetry
        results['neighborhood_symmetry'] = self._safe_float(
            0.5 * (results['trustworthiness'] + results['continuity'])
        )

        # 6. Overall quality
        results['overall_quality'] = float(np.mean([
            results['trustworthiness'],
            results['continuity'],
            results['distance_spearman'],
            results['distance_pearson'],
            results['local_scale_quality'],
            results['neighborhood_symmetry'],
        ]))

        if self.verbose:
            self._print_results(results)

        return results

    def _print_results(self, results: Dict[str, float]):
        """Print a formatted summary of evaluation results."""

        print("\n" + "=" * 60)
        print("       Extended Dimensionality Reduction Quality (DREX)")
        print("=" * 60)

        print("\n[Neighborhood Preservation]")
        print(f"  Trustworthiness:        {results['trustworthiness']:.4f} ★")
        print("    └─ Low-D preserves high-D neighborhoods")
        print(f"  Continuity:             {results['continuity']:.4f} ★")
        print("    └─ High-D neighborhoods continued in low-D")
        print(f"  Neighborhood symmetry:  {results['neighborhood_symmetry']:.4f}")
        print("    └─ Bidirectional consistency (mean of above)")

        print("\n[Distance Preservation]")
        print(f"  Distance Spearman:      {results['distance_spearman']:.4f} ★")
        print("    └─ Rank correlation of pairwise distances")
        print(f"  Distance Pearson:       {results['distance_pearson']:.4f}")
        print("    └─ Linear correlation of pairwise distances")

        print("\n[Local Geometry]")
        print(f"  Local scale quality:    {results['local_scale_quality']:.4f} ★")
        print("    └─ Consistency of local distance ratios")

        overall = results['overall_quality']

        print("\n[Overall Assessment]")
        print(f"  Mean quality score:     {overall:.4f}")

        if overall >= 0.8:
            quality_level = "Excellent"
        elif overall >= 0.6:
            quality_level = "Good"
        elif overall >= 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Needs improvement"

        print(f"  Quality level:          {quality_level}")
        print("=" * 60)

    def compare_methods(
        self,
        method_results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        n_neighbors: int = 15,
        max_samples: int = 2000) -> pd.DataFrame:
        """
        Compare multiple dimensionality reduction methods using DREX metrics.

        Args:
            method_results_dict: mapping {method_name: (X_high, X_low)}
            n_neighbors: number of neighbors
            max_samples: subsample size

        Returns:
            DataFrame: comparison table of methods
        """
        comparison_results = []

        for method_name, (X_high, X_low) in method_results_dict.items():
            self._log(f"\nEvaluating method: {method_name}")

            original_verbose = self.verbose
            self.verbose = False

            results = self.comprehensive_evaluation(
                X_high, X_low, n_neighbors, max_samples
            )

            self.verbose = original_verbose

            comparison_results.append({
                'Method': method_name,
                'Trustworthiness': results['trustworthiness'],
                'Continuity': results['continuity'],
                'Dist_Spearman': results['distance_spearman'],
                'Dist_Pearson': results['distance_pearson'],
                'Local_Scale': results['local_scale_quality'],
                'Nbr_Symmetry': results['neighborhood_symmetry'],
                'Overall_Quality': results['overall_quality'],
            })

        df = pd.DataFrame(comparison_results)
        df = df.sort_values('Overall_Quality', ascending=False)

        if self.verbose:
            self._print_comparison_table(df)

        return df

    def _print_comparison_table(self, df: pd.DataFrame):
        """Print a formatted comparison table of methods."""

        print(f"\n{'=' * 100}")
        print("              Extended DR Method Comparison (DREX)")
        print('=' * 100)

        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        print(df.to_string(index=False))

        print(
            f"\nBest method: {df.iloc[0]['Method']} "
            f"(Overall score: {df.iloc[0]['Overall_Quality']:.4f})"
        )

        print('=' * 100)


# ==================== Convenience functions ====================

def evaluate_extended_dimensionality_reduction(
    X_high: np.ndarray,
    X_low: np.ndarray,
    n_neighbors: int = 15,
    max_samples: int = 2000,
    random_state: int = 42,
    verbose: bool = True) -> Dict[str, float]:
    """
    Convenience function to evaluate extended DR quality.

    Complements ``evaluate_dimensionality_reduction()`` from DRE.py with
    geometry-oriented metrics (trustworthiness, continuity, distance
    correlations, local scale quality, neighborhood symmetry).

    Args:
        X_high: high-dimensional data (e.g. latent space)
        X_low: low-dimensional data (e.g. UMAP 2-D projection)
        n_neighbors: number of neighbors for kNN-based metrics
        max_samples: subsample to this many points
        random_state: random seed for subsampling
        verbose: whether to print detailed output

    Returns:
        dict: evaluation results with keys:
            trustworthiness, continuity, distance_spearman,
            distance_pearson, local_scale_quality,
            neighborhood_symmetry, overall_quality
    """
    evaluator = ExtendedDimensionalityReductionEvaluator(verbose=verbose)
    return evaluator.comprehensive_evaluation(
        X_high, X_low, n_neighbors, max_samples, random_state
    )


def compare_extended_dr_methods(
    method_results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_neighbors: int = 15,
    max_samples: int = 2000,
    verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to compare multiple DR methods using DREX metrics.

    Args:
        method_results_dict: mapping {method_name: (X_high, X_low)}
        n_neighbors: number of neighbors
        max_samples: subsample size
        verbose: whether to print detailed output

    Returns:
        DataFrame: comparison results
    """
    evaluator = ExtendedDimensionalityReductionEvaluator(verbose=verbose)
    return evaluator.compare_methods(method_results_dict, n_neighbors, max_samples)
