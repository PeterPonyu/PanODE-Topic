import numpy as np
import pandas as pd  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore
import warnings
from typing import Dict


class ExtendedLatentSpaceEvaluator:
    """
    Extended latent space quality evaluator.

    Complements the base LSE evaluator with additional intrinsic geometry
    metrics that do not require clustering labels or a high-dimensional
    reference.

    Core metrics:
    - two_hop_connectivity: kNN 2-hop graph reachability
    - radial_concentration_quality: radial distribution health
    - local_curvature_linearity: PCA dominance in local neighborhoods
    - neighbor_entropy_stability: uniformity of local distance distributions

    Features:
    - Focus on intrinsic latent geometry (no external reference needed)
    - Automatic subsampling for large datasets
    - Complementary to LSE (spectral / participation-based)
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

    def _validate_inputs(self, latent_space: np.ndarray):
        """Validate input parameters."""
        if not isinstance(latent_space, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")

        if latent_space.ndim != 2:
            raise ValueError("Input data must be a 2D array.")

        if latent_space.shape[0] < 10:
            raise ValueError("Need at least 10 samples for stable extended LSE metrics.")

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
    def _subsample(
        z: np.ndarray,
        max_samples: int = 2000,
        random_state: int = 42) -> np.ndarray:
        """Subsample latent space for tractable computation."""
        if z.shape[0] <= max_samples:
            return z
        rng = np.random.default_rng(random_state)
        idx = rng.choice(z.shape[0], max_samples, replace=False)
        return z[idx]

    # ==================== 1. Two-hop connectivity ====================

    def two_hop_connectivity_score(
        self, latent_space: np.ndarray, n_neighbors: int = 15
    ) -> float:
        """
        Two-hop neighborhood reachability in the kNN graph.

        Captures local manifold connectivity: what fraction of the
        population is reachable from any point within two kNN hops.

        Args:
            latent_space: latent coordinates, shape (n_samples, n_dims)
            n_neighbors: number of neighbors for the kNN graph

        Returns:
            float: reachability fraction in [0, 1] (higher is better)
        """
        try:
            n = latent_space.shape[0]
            k = min(max(2, n_neighbors), n - 1)

            nn = NearestNeighbors(n_neighbors=k + 1).fit(latent_space)
            _, idx = nn.kneighbors(latent_space)
            idx = idx[:, 1:]  # drop self

            # Dense boolean adjacency matrix (efficient for n <= 2000)
            adj = np.zeros((n, n), dtype=bool)
            rows = np.repeat(np.arange(n), k)
            cols = idx.reshape(-1)
            adj[rows, cols] = True

            two_hop = adj @ adj
            two_hop = two_hop > 0

            reachable = adj | two_hop
            np.fill_diagonal(reachable, False)

            return float(np.mean(np.sum(reachable, axis=1) / max(1, n - 1)))

        except Exception as e:
            warnings.warn(f"Error computing two-hop connectivity: {e}")
            return 0.0

    # ==================== 2. Radial concentration quality ====================

    def radial_concentration_score(self, latent_space: np.ndarray) -> float:
        """
        Radial concentration quality: not collapsed, not wildly dispersed.

        Measures whether the coefficient of variation of radial distances
        from the centroid is near a healthy target (~0.8). Too low may
        indicate mode collapse; too high indicates excessive spread.

        Args:
            latent_space: latent coordinates, shape (n_samples, n_dims)

        Returns:
            float: radial health score in [0, 1] (higher is better)
        """
        try:
            center = np.mean(latent_space, axis=0, keepdims=True)
            r = np.linalg.norm(latent_space - center, axis=1)
            mean_r = np.mean(r) + 1e-12
            std_r = np.std(r)
            cv = std_r / mean_r

            # Prefer moderate CV; too low → collapse, too high → noisy spread
            target = 0.8
            score = np.exp(-abs(cv - target))

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            warnings.warn(f"Error computing radial concentration: {e}")
            return 0.0

    # ==================== 3. Local curvature linearity ====================

    def local_curvature_score(
        self, latent_space: np.ndarray, n_neighbors: int = 25
    ) -> float:
        """
        Local curvature proxy via PCA dominance in neighborhoods.

        For each point, fit PCA on its k-nearest neighborhood and measure
        how much variance the first PC explains relative to the second.
        High dominance ⇒ locally linear (trajectory-like) patches.

        Args:
            latent_space: latent coordinates, shape (n_samples, n_dims)
            n_neighbors: number of neighbors per local patch

        Returns:
            float: curvature linearity score in [0, 1] (higher is better)
        """
        try:
            n = latent_space.shape[0]
            k = min(max(5, n_neighbors), n - 1)

            nn = NearestNeighbors(n_neighbors=k + 1).fit(latent_space)
            _, idx = nn.kneighbors(latent_space)
            idx = idx[:, 1:]  # drop self

            scores = []
            for i in range(n):
                patch = latent_space[idx[i]]
                patch = patch - np.mean(patch, axis=0, keepdims=True)
                if patch.shape[0] < 5:
                    continue
                pca = PCA(n_components=min(3, patch.shape[1], patch.shape[0]))
                pca.fit(patch)
                evr = pca.explained_variance_ratio_
                if evr.size >= 2:
                    dominance = evr[0] / (evr[1] + 1e-12)
                    scores.append(dominance / (1.0 + dominance))

            if not scores:
                return 0.0

            return float(np.mean(scores))

        except Exception as e:
            warnings.warn(f"Error computing local curvature: {e}")
            return 0.0

    # ==================== 4. Neighbor entropy stability ====================

    def entropy_stability_score(
        self, latent_space: np.ndarray, n_neighbors: int = 15
    ) -> float:
        """
        Entropy stability of neighbor distance distributions.

        For each point, compute the entropy of its k-nearest distances
        (after normalisation to a distribution). Low dispersion of
        per-point entropies ⇒ uniform local geometry.

        Args:
            latent_space: latent coordinates, shape (n_samples, n_dims)
            n_neighbors: number of neighbors

        Returns:
            float: entropy stability score in [0, 1] (higher is better)
        """
        try:
            n = latent_space.shape[0]
            k = min(max(3, n_neighbors), n - 1)

            nn = NearestNeighbors(n_neighbors=k + 1).fit(latent_space)
            d, _ = nn.kneighbors(latent_space)
            d = d[:, 1:]  # drop self-distance

            p = d / (np.sum(d, axis=1, keepdims=True) + 1e-12)
            ent = -np.sum(p * np.log(p + 1e-12), axis=1)

            mean_ent = np.mean(ent) + 1e-12
            cv_ent = np.std(ent) / mean_ent

            return float(np.exp(-cv_ent))

        except Exception as e:
            warnings.warn(f"Error computing entropy stability: {e}")
            return 0.0

    # ==================== 5. Comprehensive evaluation ====================

    def comprehensive_evaluation(
        self,
        latent_space: np.ndarray,
        n_neighbors: int = 15,
        max_samples: int = 2000,
        random_state: int = 42) -> Dict:
        """
        Comprehensive evaluation of extended latent space quality metrics.

        Args:
            latent_space: latent coordinates, shape (n_samples, n_dims)
            n_neighbors: number of neighbors for kNN-based metrics
            max_samples: subsample to this many points for tractability
            random_state: random seed for subsampling

        Returns:
            dict: dictionary with all LSEX evaluation metrics and
                  an ``interpretation`` sub-dict
        """
        self._validate_inputs(latent_space)

        self._log(
            f"Starting extended latent space evaluation "
            f"(n_samples={latent_space.shape[0]}, k={n_neighbors})..."
        )

        # Subsample for tractability
        z = self._subsample(latent_space, max_samples, random_state)

        results: Dict = {}

        # 1. Two-hop connectivity
        self._log("Computing two-hop connectivity...")
        results['two_hop_connectivity'] = self.two_hop_connectivity_score(
            z, n_neighbors
        )

        # 2. Radial concentration
        self._log("Computing radial concentration quality...")
        results['radial_concentration_quality'] = self.radial_concentration_score(z)

        # 3. Local curvature linearity
        curvature_k = max(10, n_neighbors + 5)
        self._log("Computing local curvature linearity...")
        results['local_curvature_linearity'] = self.local_curvature_score(
            z, curvature_k
        )

        # 4. Neighbor entropy stability
        self._log("Computing neighbor entropy stability...")
        results['neighbor_entropy_stability'] = self.entropy_stability_score(
            z, n_neighbors
        )

        # 5. Overall quality
        core_values = [
            results['two_hop_connectivity'],
            results['radial_concentration_quality'],
            results['local_curvature_linearity'],
            results['neighbor_entropy_stability'],
        ]
        results['overall_quality'] = float(np.mean(core_values))

        # 6. Interpretation
        results['interpretation'] = self._generate_interpretation(results)

        if self.verbose:
            self._print_comprehensive_results(results)

        return results

    # ==================== Interpretation ====================

    def _generate_interpretation(self, results: Dict) -> Dict:
        """Generate a qualitative interpretation of the results."""

        interpretation: Dict = {
            'quality_level': '',
            'strengths': [],
            'weaknesses': [],
            'recommendations': [],
        }

        overall = results['overall_quality']

        # Quality level
        if overall >= 0.8:
            interpretation['quality_level'] = "Excellent"
        elif overall >= 0.6:
            interpretation['quality_level'] = "Good"
        elif overall >= 0.4:
            interpretation['quality_level'] = "Fair"
        else:
            interpretation['quality_level'] = "Needs improvement"

        thresholds = {'high': 0.7, 'medium': 0.5, 'low': 0.3}

        # Strengths
        if results['two_hop_connectivity'] > thresholds['high']:
            interpretation['strengths'].append(
                "High neighborhood graph connectivity"
            )
        if results['radial_concentration_quality'] > thresholds['high']:
            interpretation['strengths'].append(
                "Healthy radial distribution (no mode collapse)"
            )
        if results['local_curvature_linearity'] > thresholds['high']:
            interpretation['strengths'].append(
                "Locally linear structure (trajectory-like patches)"
            )
        if results['neighbor_entropy_stability'] > thresholds['high']:
            interpretation['strengths'].append(
                "Uniform local geometry (stable entropy)"
            )

        # Weaknesses
        if results['two_hop_connectivity'] < thresholds['medium']:
            interpretation['weaknesses'].append(
                "Low graph connectivity — fragmented manifold"
            )
        if results['radial_concentration_quality'] < thresholds['medium']:
            interpretation['weaknesses'].append(
                "Poor radial concentration — possible mode collapse or explosion"
            )
        if results['local_curvature_linearity'] < thresholds['medium']:
            interpretation['weaknesses'].append(
                "Weak local linearity — noisy or highly curved local patches"
            )
        if results['neighbor_entropy_stability'] < thresholds['medium']:
            interpretation['weaknesses'].append(
                "Unstable local geometry — uneven neighborhood structure"
            )

        # Recommendations
        if overall < 0.6:
            interpretation['recommendations'].append(
                "Consider adjusting latent dimensionality or model capacity"
            )
        if results['two_hop_connectivity'] < 0.4:
            interpretation['recommendations'].append(
                "Increase neighborhood size or use stronger contrastive loss"
            )
        if results['radial_concentration_quality'] < 0.4:
            interpretation['recommendations'].append(
                "Check for mode collapse; consider KL annealing or dropout"
            )
        if results['neighbor_entropy_stability'] < 0.4:
            interpretation['recommendations'].append(
                "Regularise latent space for more uniform local geometry"
            )

        return interpretation

    # ==================== Printing ====================

    def _print_comprehensive_results(self, results: Dict):
        """Print human-readable summary of the evaluation."""

        print("\n" + "=" * 70)
        print("       Extended Latent Space Quality Evaluation (LSEX)")
        print("=" * 70)

        print("\n[Graph Connectivity]")
        print(f"  Two-hop connectivity:          {results['two_hop_connectivity']:.4f} ★")
        print("    └─ kNN 2-hop reachability fraction")

        print("\n[Radial Distribution]")
        print(f"  Radial concentration quality:  {results['radial_concentration_quality']:.4f} ★")
        print("    └─ Healthy spread around centroid (no collapse / explosion)")

        print("\n[Local Geometry]")
        print(f"  Local curvature linearity:     {results['local_curvature_linearity']:.4f} ★")
        print("    └─ PCA dominance in local neighborhoods")
        print(f"  Neighbor entropy stability:    {results['neighbor_entropy_stability']:.4f} ★")
        print("    └─ Uniformity of local distance distributions")

        overall = results['overall_quality']

        print("\n[Overall Assessment]")
        print(f"  Mean quality score:            {overall:.4f}")

        interp = results['interpretation']
        print(f"  Quality level:                 {interp['quality_level']}")

        if interp['strengths']:
            print(f"\n  Strengths: {', '.join(interp['strengths'])}")

        if interp['weaknesses']:
            print(f"  Weaknesses: {', '.join(interp['weaknesses'])}")

        if interp['recommendations']:
            print(f"  Recommendations: {', '.join(interp['recommendations'])}")

        print("=" * 70)

    # ==================== Method comparison ====================

    def compare_methods(
        self,
        method_results_dict: Dict[str, np.ndarray],
        n_neighbors: int = 15,
        max_samples: int = 2000) -> pd.DataFrame:
        """
        Compare multiple latent space methods using LSEX metrics.

        Args:
            method_results_dict: mapping {method_name: latent_space}
            n_neighbors: number of neighbors
            max_samples: subsample size

        Returns:
            DataFrame: comparison table of methods
        """
        comparison_results = []

        for method_name, latent_space in method_results_dict.items():
            self._log(f"\nEvaluating method: {method_name}")

            original_verbose = self.verbose
            self.verbose = False

            results = self.comprehensive_evaluation(
                latent_space, n_neighbors, max_samples
            )

            self.verbose = original_verbose

            comparison_results.append({
                'Method': method_name,
                'TwoHop_Connectivity': results['two_hop_connectivity'],
                'Radial_Concentration': results['radial_concentration_quality'],
                'Curvature_Linearity': results['local_curvature_linearity'],
                'Entropy_Stability': results['neighbor_entropy_stability'],
                'Overall_Quality': results['overall_quality'],
                'Quality_Level': results['interpretation']['quality_level'],
            })

        df = pd.DataFrame(comparison_results)
        df = df.sort_values('Overall_Quality', ascending=False)

        if self.verbose:
            self._print_comparison_table(df)

        return df

    def _print_comparison_table(self, df: pd.DataFrame):
        """Print a formatted comparison table of methods."""

        print(f"\n{'=' * 100}")
        print("              Extended Latent Space Method Comparison (LSEX)")
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

def evaluate_extended_latent_space(
    latent_space: np.ndarray,
    n_neighbors: int = 15,
    max_samples: int = 2000,
    random_state: int = 42,
    verbose: bool = True) -> Dict:
    """
    Convenience function to evaluate extended latent space quality.

    Complements ``evaluate_single_cell_latent_space()`` from LSE.py with
    intrinsic geometry metrics (two-hop connectivity, radial concentration,
    local curvature linearity, neighbor entropy stability).

    Args:
        latent_space: latent coordinates (e.g. autoencoder bottleneck)
        n_neighbors: number of neighbors for kNN-based metrics
        max_samples: subsample to this many points
        random_state: random seed for subsampling
        verbose: whether to print detailed output

    Returns:
        dict: evaluation results with keys:
            two_hop_connectivity, radial_concentration_quality,
            local_curvature_linearity, neighbor_entropy_stability,
            overall_quality, interpretation
    """
    evaluator = ExtendedLatentSpaceEvaluator(verbose=verbose)
    return evaluator.comprehensive_evaluation(
        latent_space, n_neighbors, max_samples, random_state
    )


def compare_extended_latent_methods(
    method_results_dict: Dict[str, np.ndarray],
    n_neighbors: int = 15,
    max_samples: int = 2000,
    verbose: bool = True) -> pd.DataFrame:
    """
    Convenience function to compare multiple methods using LSEX metrics.

    Args:
        method_results_dict: mapping {method_name: latent_space}
        n_neighbors: number of neighbors
        max_samples: subsample size
        verbose: whether to print detailed output

    Returns:
        DataFrame: comparison results
    """
    evaluator = ExtendedLatentSpaceEvaluator(verbose=verbose)
    return evaluator.compare_methods(method_results_dict, n_neighbors, max_samples)
